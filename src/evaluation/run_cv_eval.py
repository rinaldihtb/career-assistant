"""Set eval Cocokkan CV (tests/eval/cv_cases.yaml): pilih MatchConfig dengan gold per CV.

    make eval-cv ARGS="--label nama"      semua CV fiktif (butuh .env)
    make eval-cv ARGS="--only cv01"       sebagian CV (prefix ID)
    make eval-cv ARGS="--rescore tests/eval/results/<file>.json"
                                          hitung ulang dari kandidat tersimpan, tanpa API

Biaya per CV: 1 panggilan LLM (profil), 2 embedding (ringkasan profil dan teks CV), 2 query
Qdrant (k = ``MatchConfig().k``). Grid, kalibrasi, dan metrik dihitung di Python dari kandidat
yang tersimpan, jadi menambah konfigurasi atau mengoreksi gold tidak menambah biaya API.

Alur per CV:
1. ``extract_cv_text`` -> ``extract_cv_profile`` -> nilai profil terhadap gold (recall skill,
   skill yang dikarang, jenjang pendidikan, jumlah pengalaman) dan cek kutipan pencapaian
   terhadap teks CV (tanpa gold).
2. Per vektor query: semua lowongan kandidat disimpan dengan cosine ``max`` dan ``mean`` serta
   jumlah skill cocok/disebut. Yang disimpan hanya angka, ID, dan profil (CV-nya fiktif);
   teks CV tidak disimpan.

Analisis:
1. Kalibrasi ``semantic_floor``/``semantic_ceiling`` per (query, aggregation) dari distribusi
   cosine semua CV (lihat ``calibrate``). Dilakukan sebelum grid karena rentang ceiling - floor
   ikut menentukan bobot efektif skor semantik terhadap skor skill, jadi ikut mengubah urutan.
2. Grid query x aggregation x skill_prior x semantic_weight, ditambah pembanding "semantik
   saja" (bobot 1.0) yang tidak ikut dipilih. Metrik: precision@5 dan @10 terhadap gold.
3. Pilih: rata-rata p@5 tertinggi; seri -> rata-rata p@10, lalu p@5 terburuk per CV, lalu yang
   paling dekat dengan default ``MatchConfig()`` (tanpa perubahan kode = paling sederhana).

Urutan dihitung ulang dengan fungsi yang sama dengan ``rank_jobs`` (``scale_semantic``,
``skill_score``, ``combine_score``); kesamaan urutannya dijaga test.
"""

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.agents.rag_agent import MIN_CHUNK_CHARS
from src.cv.match import (
    JobSkillIndex,
    MatchConfig,
    aggregate_job_scores,
    combine_score,
    load_job_skill_index,
    profile_query_text,
    scale_semantic,
    skill_score,
)
from src.cv.pdf_text import CVRejectedError, extract_cv_text
from src.cv.profile import Achievement, CVProfile, extract_cv_profile
from src.cv.skills import CVSkills, build_cv_skills, match_skills
from src.repositories.job_repository import connect_readonly

CASES_PATH = Path("tests/eval/cv_cases.yaml")
RESULTS_DIR = Path("tests/eval/results")
TRACKED_FILES = (
    "src/cv/pdf_text.py",
    "src/cv/profile.py",
    "src/cv/skills.py",
    "src/cv/match.py",
    "src/agents/prompts.py",
    "src/evaluation/run_cv_eval.py",
)
QUERIES = ("profile", "cv_text")
AGGREGATIONS = ("max", "mean")
SKILL_PRIORS = (0.0, 1.0, 2.0, 3.0)
SEMANTIC_WEIGHTS = (0.4, 0.5, 0.6, 0.7, 0.8)
REFERENCE_WEIGHT = 1.0  # semantik saja: pembanding, tidak ikut dipilih
CALIBRATION_STEP = 0.05
TOP_SCORE_RANGE = (0.70, 0.90)  # skor teratas yang wajar ditampilkan di kartu hasil
DEFAULT = MatchConfig()
CONFIG_FIELDS = {
    "query",
    "aggregation",
    "skill_prior",
    "semantic_weight",
    "semantic_floor",
    "semantic_ceiling",
}
SHOW_ROWS = 10
TOP_N = 10


def load_cases(path: Path = CASES_PATH, only: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Baca kasus dari YAML; ``only`` menyaring berdasarkan prefix ID (mis. "cv01")."""
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]
    if only:
        cases = [case for case in cases if case["id"].startswith(tuple(only))]
    return cases


def load_titles(db_path: str | Path) -> dict[int, str]:
    """``{id: job_title}`` semua lowongan (473 baris), dibaca read-only."""
    with closing(connect_readonly(db_path)) as conn:
        return dict(conn.execute("SELECT id, job_title FROM jobs"))


def gold_job_ids(case: Mapping[str, Any], titles: Mapping[int, str]) -> list[int]:
    """ID lowongan relevan: judul cocok ``title_include`` dan tidak cocok ``title_exclude``,
    ditambah ``extra_ids`` (lowongan relevan menurut deskripsi yang judulnya tidak tertangkap
    regex, dinilai manual; ID yang tidak ada di database diabaikan).
    """
    include = re.compile(case["title_include"], re.IGNORECASE)
    exclude = (
        re.compile(case["title_exclude"], re.IGNORECASE) if case.get("title_exclude") else None
    )
    by_title = {
        job_id
        for job_id, title in titles.items()
        if include.search(title) and not (exclude and exclude.search(title))
    }
    extra = {job_id for job_id in case.get("extra_ids") or () if job_id in titles}
    return sorted(by_title | extra)


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def same_skill(a: str, b: str) -> bool:
    """Dua penulisan skill dianggap sama: identik setelah normalisasi, atau satu memuat yang lain
    utuh per kata.

    Yang dinilai adalah skill-nya tertangkap, bukan ejaannya: LLM boleh memecah
    "Rekrutmen & Seleksi" menjadi "Rekrutmen" atau meringkas "Microsoft Excel (VBA)" menjadi
    "Microsoft Excel". Batas kata mencegah "Excel" cocok dengan "Excellent".
    """
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    return re.search(rf"(?<!\w){re.escape(short)}(?!\w)", long) is not None


def check_skills(
    extracted: Sequence[str], gold: Sequence[str | Sequence[str]], cv_text: str
) -> dict[str, Any]:
    """Nilai skill profil terhadap gold.

    Args:
        extracted: ``CVProfile.skills`` hasil LLM.
        gold: Item gold; item berupa daftar = varian yang dianggap sama (Inggris/English).
        cv_text: Teks CV hasil ``extract_cv_text``.

    Returns:
        ``gold``, ``found``, ``recall``, ``missed`` (item gold yang tidak tertangkap),
        ``invented`` (skill hasil LLM yang tidak ada di teks CV dan tidak cocok dengan gold:
        dikarang), ``extra`` (ada di teks CV tetapi di luar gold, mis. dari bagian Sertifikasi;
        hanya informasi).
    """
    variants = [[item] if isinstance(item, str) else list(item) for item in gold]
    missed = [
        names[0]
        for names in variants
        if not any(same_skill(name, skill) for name in names for skill in extracted)
    ]
    text = _norm(cv_text)
    invented: list[str] = []
    extra: list[str] = []
    for skill in extracted:
        if any(same_skill(name, skill) for names in variants for name in names):
            continue
        (extra if _norm(skill) in text else invented).append(skill)
    found = len(variants) - len(missed)
    return {
        "gold": len(variants),
        "found": found,
        "recall": found / len(variants) if variants else None,
        "missed": missed,
        "invented": invented,
        "extra": extra,
    }


# Tanda hubung dan kutip tipografis di CV diseragamkan dengan versi ASCII yang biasa ditulis LLM.
_PUNCTUATION = str.maketrans(
    {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
     "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
)  # fmt: skip
_BULLETS = frozenset("•▪●◦‣∙·")
_CONTEXT_PARTS = re.compile(r"\s+[-|]\s+|,\s*")


def squash(text: str) -> str:
    """Bentuk pembanding kutipan: NFKC, huruf kecil, tanda baca diseragamkan, tanpa spasi dan
    simbol bullet.

    Spasi dibuang seluruhnya karena teks pypdf memotong kalimat per baris dan judul bagian
    template Canva ditulis satu huruf per baris.
    """
    text = unicodedata.normalize("NFKC", text).casefold().translate(_PUNCTUATION)
    return "".join(
        ch
        for ch in text
        if not ch.isspace() and ch not in _BULLETS and unicodedata.category(ch) != "Co"
    )


def check_achievements(achievements: Sequence[Achievement], cv_text: str) -> dict[str, Any]:
    """Cek pencapaian hasil LLM terhadap teks CV (aturan 8 ``CV_PROFILE_SYSTEM``).

    Returns:
        ``count``, ``numeric`` (kutipan yang memuat angka), ``not_verbatim`` (kutipan yang tidak
        ada di teks CV setelah ``squash``: diterjemahkan, diringkas, atau dikarang), dan
        ``context_unknown`` (jumlah ``context`` yang salah satu bagiannya, dipisah koma atau
        " - ", tidak ada di teks CV).
    """
    text = squash(cv_text)
    not_verbatim = [item.text for item in achievements if squash(item.text) not in text]
    context_unknown = sum(
        1
        for item in achievements
        if item.context
        and any(squash(part) not in text for part in _CONTEXT_PARTS.split(item.context) if part)
    )
    return {
        "count": len(achievements),
        "numeric": sum(1 for item in achievements if re.search(r"\d", item.text)),
        "not_verbatim": not_verbatim,
        "context_unknown": context_unknown,
    }


def check_profile(profile: CVProfile, expected: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Jenjang pendidikan dan jumlah pengalaman: nilai harapan dan hasil ekstraksi."""
    return {
        "education_level": {
            "expected": expected.get("education_level"),
            "actual": profile.education_level,
        },
        "experiences": {
            "expected": expected.get("experiences"),
            "actual": len(profile.experiences),
        },
    }


def collect_candidates(
    query_vector: list[float],
    cv_skills: CVSkills,
    *,
    vector_store: Any,
    index: JobSkillIndex,
    config: MatchConfig = DEFAULT,
) -> list[dict[str, Any]]:
    """Semua lowongan kandidat untuk satu vektor query: 1 query Qdrant, tanpa embedding/LLM.

    Filter chunk dan agregasi sama dengan ``rank_jobs``. Bedanya, semua kandidat disimpan
    dengan kedua agregasi dan jumlah skill mentah, supaya grid bisa dihitung ulang tanpa API.

    Returns:
        Daftar dict ``job_id``, ``max``, ``mean`` (cosine), ``matched``, ``required``
        (jumlah skill), urut ``job_id``.
    """
    chunks = vector_store.similarity_search_with_score_by_vector(query_vector, k=config.k)
    hits = [
        (int(doc.metadata["job_id"]), float(score))
        for doc, score in chunks
        if doc.metadata.get("job_id") is not None
        and len(doc.page_content.strip()) >= MIN_CHUNK_CHARS
    ]
    by_max = aggregate_job_scores(hits, method="max")
    by_mean = aggregate_job_scores(hits, method="mean", top_chunks=config.top_chunks)
    candidates = []
    for job_id in sorted(by_max):
        if job_id not in index.vocab:
            continue  # vektor basi: ID tidak ada lagi di SQLite (sama dengan rank_jobs)
        skills = match_skills(
            cv_skills,
            index.vocab[job_id],
            index.descriptions[job_id],
            frequency=index.frequency,
        )
        candidates.append(
            {
                "job_id": job_id,
                "max": by_max[job_id],
                "mean": by_mean[job_id],
                "matched": len(skills.matched),
                "required": skills.required,
            }
        )
    return candidates


def rank_candidates(
    candidates: Sequence[Mapping[str, Any]], config: MatchConfig
) -> list[tuple[int, float]]:
    """Urutkan kandidat dengan rumus dan urutan seri yang sama dengan ``rank_jobs``.

    Returns:
        ``(job_id, skor)`` urut skor menurun, semua kandidat (tanpa ``limit``).
    """
    scored = []
    for candidate in candidates:
        raw = candidate[config.aggregation]
        scaled = scale_semantic(raw, config.semantic_floor, config.semantic_ceiling)
        skill = skill_score(candidate["matched"], candidate["required"], config.skill_prior)
        total = combine_score(scaled, skill, config.semantic_weight)
        scored.append((total, raw, candidate["job_id"]))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [(job_id, total) for total, _, job_id in scored]


def precision_at(ranked_ids: Sequence[int], gold: set[int] | frozenset[int], n: int) -> float:
    """Porsi ``n`` lowongan teratas yang termasuk gold; posisi kosong dihitung tidak relevan."""
    return sum(1 for job_id in ranked_ids[:n] if job_id in gold) / n


def quantile(values: Sequence[float], q: float) -> float:
    """Persentil dengan interpolasi linear: ``q=0.5`` median, ``q=1.0`` maksimum."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def calibrate(
    candidate_sets: Sequence[Sequence[Mapping[str, Any]]],
    aggregation: str,
    *,
    step: float = CALIBRATION_STEP,
) -> tuple[float, float]:
    """Batas kalibrasi semantik dari distribusi cosine semua CV, kelipatan ``step``.

    floor: median cosine kandidat terendah di antara CV, dibulatkan ke bawah. Untuk setiap CV,
    setidaknya separuh lowongan kandidatnya mendapat skor semantik di atas 0%.
    ceiling: cosine teratas tertinggi di antara CV, dibulatkan ke bawah. Lowongan terbaik CV
    yang paling cocok mendapat skor semantik sekitar 100%.

    Kelipatan 0,05 karena 5 CV tidak cukup untuk presisi 0,01. Aturan pertama (median gabungan
    dan maksimum, presisi 0,01) pada set eval memberi [0.50, 0.71] dan skor
    teratas cv04 66%, di luar 70-90%; aturan ini memberi [0.45, 0.70] dan 71-89% dengan
    precision@5 dan @10 yang sama (1.00).
    """
    sets = [candidates for candidates in candidate_sets if candidates]
    medians = [quantile([c[aggregation] for c in candidates], 0.5) for candidates in sets]
    tops = [max(c[aggregation] for c in candidates) for candidates in sets]

    def down(value: float) -> float:
        # round(..., 6) dulu: 0.7 / 0.05 = 13.999999999999998 akan terbulatkan ke 13.
        return round(math.floor(round(value / step, 6)) * step, 2)

    floor = down(min(medians))
    return floor, max(down(max(tops)), round(floor + step, 2))


def grid_configs(
    calibration: Mapping[str, tuple[float, float]],
) -> list[tuple[MatchConfig, bool]]:
    """Semua konfigurasi grid; ``True`` = boleh dipilih, ``False`` = pembanding semantik saja."""
    configs: list[tuple[MatchConfig, bool]] = []
    for query in QUERIES:
        for aggregation in AGGREGATIONS:
            floor, ceiling = calibration[f"{query}/{aggregation}"]
            base = {
                "query": query,
                "aggregation": aggregation,
                "semantic_floor": floor,
                "semantic_ceiling": ceiling,
            }
            for prior in SKILL_PRIORS:
                for weight in SEMANTIC_WEIGHTS:
                    config = MatchConfig(**base, skill_prior=prior, semantic_weight=weight)
                    configs.append((config, True))
            reference = MatchConfig(**base, semantic_weight=REFERENCE_WEIGHT)
            configs.append((reference, False))
    return configs


def evaluate_config(
    config: MatchConfig, runs: Sequence[Mapping[str, Any]], *, selectable: bool = True
) -> dict[str, Any]:
    """Precision@5, precision@10, dan skor teratas per CV untuk satu konfigurasi.

    Args:
        runs: Per CV: ``id``, ``gold`` (set ID), ``candidates`` (``{query: kandidat}``).
    """
    p5: dict[str, float] = {}
    p10: dict[str, float] = {}
    top: dict[str, float | None] = {}
    for run in runs:
        ranked = rank_candidates(run["candidates"][config.query], config)
        ids = [job_id for job_id, _ in ranked]
        p5[run["id"]] = precision_at(ids, run["gold"], 5)
        p10[run["id"]] = precision_at(ids, run["gold"], 10)
        top[run["id"]] = round(ranked[0][1], 4) if ranked else None
    return {
        "config": config.model_dump(include=CONFIG_FIELDS),
        "selectable": selectable,
        "p5": p5,
        "p10": p10,
        "mean_p5": round(statistics.mean(p5.values()), 4),
        "mean_p10": round(statistics.mean(p10.values()), 4),
        "min_p5": min(p5.values()),
        "top_score": top,
    }


def choose_config(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Pilih baris grid: p@5 rata-rata tertinggi, lalu p@10 rata-rata, lalu p@5 terburuk,
    lalu yang paling dekat dengan default ``MatchConfig()``.

    "Paling dekat dengan default" dipakai sebagai ukuran "paling sederhana": konfigurasi yang
    sama dengan default tidak butuh perubahan kode. Urutan kedekatan: query, aggregation,
    selisih skill_prior, selisih semantic_weight.
    """

    def key(row: Mapping[str, Any]) -> tuple:
        config = row["config"]
        return (
            -row["mean_p5"],
            -row["mean_p10"],
            -row["min_p5"],
            config["query"] != DEFAULT.query,
            config["aggregation"] != DEFAULT.aggregation,
            abs(config["skill_prior"] - DEFAULT.skill_prior),
            abs(config["semantic_weight"] - DEFAULT.semantic_weight),
        )

    return min((row for row in rows if row["selectable"]), key=key)


def analyze(
    records: Sequence[Mapping[str, Any]],
    titles: Mapping[int, str],
) -> dict[str, Any]:
    """Kalibrasi, grid, dan pilihan konfigurasi dari record yang punya kandidat; tanpa API.

    Returns:
        Ringkasan untuk JSON: ``calibration``, ``grid`` (semua baris), ``chosen``,
        ``current_default`` (default sekarang apa adanya), ``top`` (``TOP_N`` teratas per CV
        untuk konfigurasi terpilih), ``out_of_range`` (CV dengan skor teratas di luar
        ``TOP_SCORE_RANGE``), dan ringkasan profil.
    """
    runs = [
        {
            "id": r["id"],
            "gold": set(r["gold_ids"]),
            "judged": set(r["gold_ids"]) | set(r.get("not_relevant_ids") or ()),
            "candidates": r["candidates"],
        }
        for r in records
        if not r.get("error") and set(QUERIES) <= set(r.get("candidates") or {})
    ]
    if not runs:
        return {"error": "Tidak ada CV dengan kandidat lengkap"}

    calibration = {
        f"{query}/{aggregation}": calibrate([run["candidates"][query] for run in runs], aggregation)
        for query in QUERIES
        for aggregation in AGGREGATIONS
    }
    grid = [
        evaluate_config(config, runs, selectable=selectable)
        for config, selectable in grid_configs(calibration)
    ]
    chosen = choose_config(grid)
    chosen_config = MatchConfig(**chosen["config"])
    low, high = TOP_SCORE_RANGE

    top = {}
    for run in runs:
        ranked = rank_candidates(run["candidates"][chosen_config.query], chosen_config)
        top[run["id"]] = [
            {
                "job_id": job_id,
                "score": round(score, 4),
                "gold": job_id in run["gold"],
                "judged": job_id in run["judged"],
                "title": titles.get(job_id, "?"),
            }
            for job_id, score in ranked[:TOP_N]
        ]

    checked = [r for r in records if r.get("skills_check")]
    return {
        "cases": len(records),
        "cases_scored": len(runs),
        "calibration_step": CALIBRATION_STEP,
        "calibration": {
            name: {"floor": floor, "ceiling": ceiling}
            for name, (floor, ceiling) in calibration.items()
        },
        "chosen": chosen,
        "current_default": evaluate_config(DEFAULT, runs, selectable=False),
        "out_of_range": [
            cv_id
            for cv_id, score in chosen["top_score"].items()
            if score is None or not low <= score <= high
        ],
        "top": top,
        "unjudged": {
            cv_id: [item["job_id"] for item in items if not item["judged"]]
            for cv_id, items in top.items()
            if any(not item["judged"] for item in items)
        },
        "skill_recall_mean": round(statistics.mean(r["skills_check"]["recall"] for r in checked), 4)
        if checked
        else None,
        "skills_invented": sum(len(r["skills_check"]["invented"]) for r in checked),
        "achievements": sum(r.get("achievements_check", {}).get("count", 0) for r in checked),
        "achievements_not_verbatim": sum(
            len(r.get("achievements_check", {}).get("not_verbatim", ())) for r in checked
        ),
        "profile_fields_ok": sum(
            item["expected"] == item["actual"]
            for r in checked
            for item in r["profile_check"].values()
        ),
        "profile_fields": sum(len(r["profile_check"]) for r in checked),
        "grid": grid,
    }


def score_record(
    record: dict[str, Any], case: Mapping[str, Any], cv_text: str, titles: Mapping[int, str]
) -> dict[str, Any]:
    """Isi penilaian profil dan gold lowongan; dipakai saat run dan saat ``--rescore``."""
    profile = CVProfile.model_validate(record["profile"])
    record["skills_check"] = check_skills(profile.skills, case["profile"]["skills"], cv_text)
    record["profile_check"] = check_profile(profile, case["profile"])
    record["achievements_check"] = check_achievements(profile.achievements, cv_text)
    record["gold_ids"] = gold_job_ids(case, titles)
    record["not_relevant_ids"] = sorted(case.get("not_relevant_ids") or ())
    record["gold_count_expected"] = case.get("gold_count")
    return record


def run_case(
    case: Mapping[str, Any],
    *,
    llm: Any,
    embeddings: Any,
    vector_store: Any,
    db_path: str | Path,
) -> tuple[dict[str, Any], str | None]:
    """Jalankan satu CV dengan API asli. Tidak melempar exception; kegagalan di ``error``.

    Returns:
        ``(record, cv_text)``. ``cv_text`` hanya dipakai untuk penilaian, tidak disimpan.
    """
    record: dict[str, Any] = {"id": case["id"], "file": case["file"], "error": None}
    try:
        cv = extract_cv_text(Path(case["file"]).read_bytes())
    except (OSError, CVRejectedError) as exc:
        record["error"] = f"PDF tidak terbaca: {exc}"
        return record, None
    record.update(pages=cv.pages, chars=len(cv.text))

    started = time.perf_counter()
    result = extract_cv_profile(cv.text, llm=llm)
    record["profile_seconds"] = round(time.perf_counter() - started, 2)
    record["usage"] = result.usage.model_dump(
        include={"llm_calls", "input_tokens", "output_tokens"}
    )
    if result.profile is None:
        record["error"] = f"Profil gagal: {result.error}"
        return record, cv.text
    profile = result.profile
    # Nama dan kontak tidak dinilai eval, jadi tidak ditulis ke JSON hasil.
    record["profile"] = profile.model_dump(mode="json", exclude={"full_name", "contacts"})

    texts = {"profile": profile_query_text(profile), "cv_text": cv.text}
    record["query_chars"] = {name: len(text) for name, text in texts.items()}
    cv_skills = build_cv_skills(profile.skills)
    record["candidates"] = {}
    record["seconds"] = {}
    try:
        index = load_job_skill_index(str(Path(db_path).resolve()))
        for name, text in texts.items():
            started = time.perf_counter()
            vector = embeddings.embed_query(text)
            embedded = time.perf_counter()
            record["candidates"][name] = collect_candidates(
                vector, cv_skills, vector_store=vector_store, index=index
            )
            record["seconds"][name] = {
                "embed": round(embedded - started, 2),
                "search": round(time.perf_counter() - embedded, 2),
            }
    except Exception as exc:  # noqa: BLE001 - error provider embedding/Qdrant beragam
        record["error"] = f"Embedding/Qdrant error: {exc}"
    return record, cv.text


def run_metadata(label: str, started: datetime, cases_path: Path) -> dict[str, Any]:
    """Commit, status working tree, dan hash file yang menentukan perilaku Cocokkan CV."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "-"

    return {
        "label": label,
        "timestamp": started.isoformat(timespec="seconds"),
        "cases_version": yaml.safe_load(cases_path.read_text(encoding="utf-8")).get(
            "version", "1.0"
        ),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest()[:12],
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty_files": git("status", "--porcelain").splitlines(),
        "file_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
            for path in TRACKED_FILES
            if Path(path).exists()
        },
        "k": DEFAULT.k,
        "top_chunks": DEFAULT.top_chunks,
    }


def _config_label(config: Mapping[str, Any]) -> str:
    return (
        f"{config['query']:<7} {config['aggregation']:<4} prior {config['skill_prior']:g} "
        f"w {config['semantic_weight']:.1f} [{config['semantic_floor']:.2f}-"
        f"{config['semantic_ceiling']:.2f}]"
    )


def _row_line(row: Mapping[str, Any]) -> str:
    per_cv = " ".join(f"{cv_id} {value:.1f}" for cv_id, value in row["p5"].items())
    return (
        f"{_config_label(row['config'])}  p@5 {row['mean_p5']:.2f}  p@10 {row['mean_p10']:.2f}  "
        f"min p@5 {row['min_p5']:.1f} | {per_cv}"
    )


def print_record(record: Mapping[str, Any]) -> None:
    """Ringkasan satu CV: profil, penilaian, gold, kandidat."""
    if record.get("error") and not record.get("profile"):
        print(f"{record['id']:<5} GAGAL: {record['error']}")
        return
    skills = record["skills_check"]
    checks = record["profile_check"]
    edu = checks["education_level"]
    exp = checks["experiences"]
    usage = record.get("usage") or {}
    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    candidates = record.get("candidates") or {}
    sizes = " / ".join(f"{name} {len(items)}" for name, items in candidates.items())
    print(
        f"{record['id']:<5} profil {record.get('profile_seconds', 0):>4.1f}s {tokens:>5} tok | "
        f"skill {skills['found']}/{skills['gold']} dikarang {len(skills['invented'])} | "
        f"pendidikan {edu['actual']} {'ok' if edu['actual'] == edu['expected'] else 'SALAH'} | "
        f"pengalaman {exp['actual']}/{exp['expected']} "
        f"{'ok' if exp['actual'] == exp['expected'] else 'SALAH'} | "
        f"gold {len(record['gold_ids'])} | kandidat {sizes or '-'}"
    )
    achieved = record.get("achievements_check")
    if achieved:
        print(
            f"      pencapaian {achieved['count']} (berangka {achieved['numeric']}, "
            f"tidak persis {len(achieved['not_verbatim'])}, "
            f"konteks tidak ditemukan {achieved['context_unknown']})"
        )
    if record["gold_count_expected"] not in (None, len(record["gold_ids"])):
        print(
            f"      PERINGATAN gold berubah: {record['gold_count_expected']} -> "
            f"{len(record['gold_ids'])}"
        )
    for name in ("missed", "invented", "extra"):
        if skills[name]:
            print(f"      {name}: {', '.join(skills[name])}")
    if record.get("error"):
        print(f"      error: {record['error']}")


def print_summary(summary: Mapping[str, Any]) -> None:
    """Kalibrasi, baris grid teratas, pembanding, konfigurasi terpilih, dan top-N per CV."""
    print("\n" + "=" * 70)
    if summary.get("error"):
        print(summary["error"])
        return
    print(
        f"CV dinilai {summary['cases_scored']}/{summary['cases']} | recall skill rata-rata "
        f"{summary['skill_recall_mean']} | skill dikarang {summary['skills_invented']} | "
        f"field profil benar {summary['profile_fields_ok']}/{summary['profile_fields']} | "
        f"pencapaian {summary.get('achievements', 0)} "
        f"(tidak persis {summary.get('achievements_not_verbatim', 0)})"
    )
    print(
        "\nKalibrasi (floor = median kandidat terendah antar-CV, ceiling = cosine teratas "
        f"tertinggi; kelipatan {summary['calibration_step']:g} ke bawah):"
    )
    for name, bounds in summary["calibration"].items():
        print(f"  {name:<15} floor {bounds['floor']:.2f}  ceiling {bounds['ceiling']:.2f}")

    grid = summary["grid"]
    selectable = [row for row in grid if row["selectable"]]
    ordered = sorted(
        selectable, key=lambda row: (-row["mean_p5"], -row["mean_p10"], -row["min_p5"])
    )
    print(f"\nGrid: {SHOW_ROWS} teratas dari {len(selectable)}")
    for row in ordered[:SHOW_ROWS]:
        print("  " + _row_line(row))
    print("Pembanding semantik saja (w=1.0):")
    for row in grid:
        if not row["selectable"]:
            print("  " + _row_line(row))
    print("Default sekarang (MatchConfig()):")
    print("  " + _row_line(summary["current_default"]))

    chosen = summary["chosen"]
    print("\nTERPILIH: " + _row_line(chosen))
    print("  p@10 per CV: " + " ".join(f"{k} {v:.1f}" for k, v in chosen["p10"].items()))
    print("  skor teratas: " + " ".join(f"{k} {v:.0%}" for k, v in chosen["top_score"].items()))
    low, high = TOP_SCORE_RANGE
    if summary["out_of_range"]:
        print(f"  PERINGATAN skor teratas di luar {low:.0%}-{high:.0%}: {summary['out_of_range']}")
    if summary["unjudged"]:
        print(f"  PERINGATAN lowongan non-gold belum dinilai (? di bawah): {summary['unjudged']}")
    for cv_id, items in summary["top"].items():
        print(f"\n{cv_id} top-{TOP_N} (* = gold, ? = belum dinilai)")
        for i, item in enumerate(items, start=1):
            mark = "*" if item["gold"] else (" " if item["judged"] else "?")
            print(f"  {i:2}.{mark}#{item['job_id']:<3} {item['score']:4.0%}  {item['title'][:60]}")


def save_results(
    records: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    path: Path,
    summary: Mapping[str, Any],
) -> None:
    """Cetak ringkasan lalu simpan JSON berisi meta, ringkasan, dan semua record."""
    print_summary(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "summary": summary, "cases": records}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nHasil disimpan: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval Cocokkan CV.")
    parser.add_argument("--label", default="run", help="Nama hasil, mis. baseline.")
    parser.add_argument("--only", nargs="*", default=[], help="Prefix ID CV, mis. cv01.")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--rescore", type=Path, help="JSON hasil lama, dihitung ulang tanpa API.")
    args = parser.parse_args()

    started = datetime.now(UTC).astimezone()  # waktu lokal dengan zona waktu
    load_dotenv()
    db_path = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")
    titles = load_titles(db_path)
    cases = load_cases(args.cases, args.only)
    records: list[dict[str, Any]] = []

    if args.rescore:
        old = json.loads(args.rescore.read_text(encoding="utf-8"))
        old_records = {record["id"]: record for record in old["cases"]}
        for case in cases:
            if case["id"] not in old_records:
                print(f"{case['id']:<5} dilewati: tidak ada di {args.rescore.name}")
                continue
            record = dict(old_records[case["id"]])
            if record.get("profile"):
                cv_text = extract_cv_text(Path(case["file"]).read_bytes()).text
                score_record(record, case, cv_text, titles)
            print_record(record)
            records.append(record)
        label = f"{old['meta']['label']}-rescored" if args.label == "run" else args.label
        meta = {
            **run_metadata(label, started, args.cases),
            "rescored_from": args.rescore.name,
            "original_meta": old["meta"],
        }
    else:
        from src.config.database import VECTOR_DB  # import di sini: butuh .env
        from src.config.llm import EMBEDDING, LLM_MODEL

        for case in cases:
            record, cv_text = run_case(
                case, llm=LLM_MODEL, embeddings=EMBEDDING, vector_store=VECTOR_DB, db_path=db_path
            )
            if record.get("profile") and cv_text is not None:
                score_record(record, case, cv_text, titles)
            print_record(record)
            records.append(record)
        label = args.label
        meta = run_metadata(label, started, args.cases)

    summary = analyze(records, titles)
    path = args.out / f"cv-{label}-{started:%Y%m%d-%H%M}.json"
    save_results(records, meta, path, summary)


if __name__ == "__main__":
    main()
