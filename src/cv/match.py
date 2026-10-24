"""Cocokkan CV dengan lowongan: kemiripan Qdrant + porsi skill. 1 embedding, 0 LLM.

Alur ``match_jobs``:
1. Teks query: ringkasan profil (default) atau teks CV mentah; di-embed sekali.
2. Qdrant: ``k`` chunk terdekat dari seluruh koleksi, lalu skor diagregasi per ``job_id``.
3. Skor semantik dikalibrasi ke 0..1 dengan batas tetap (``semantic_floor``/``ceiling``),
   bukan min-max per query, supaya persentase tidak bergeser setiap "Cocokkan ulang".
4. Skor skill dari ``cv.skills`` (regex, kosakata yang di-cache per proses):
   ``cocok / (disebut + skill_prior)``. Tanpa ``skill_prior``, lowongan yang hanya menyebut
   satu skill umum (71 dari 473 lowongan) mendapat 100% jika skill itu ada di CV, sehingga
   lowongan sales masuk top-10 CV data hanya karena "English".
5. Skor akhir = ``w * semantik + (1 - w) * skor skill``. Lowongan yang tidak menyebut skill
   kosakata mendapat skor skill 0 (rumus yang sama, tanpa kasus khusus).

Ringkasan profil menjadi default karena koreksi pengguna di UI ikut di-embed saat "Cocokkan
ulang", dan ringkasan tidak memuat nama atau kontak. Teks CV mentah (``query="cv_text"``)
hanya untuk pengukuran.

Nilai default ``MatchConfig`` dipilih dengan set eval Cocokkan CV (tests/eval/cv_cases.yaml,
5 CV fiktif): precision@5 dan @10 = 1.00 untuk kelima CV. Default berada di tengah rentang
yang sama baiknya: query profil + ``max`` dengan ``skill_prior`` 1-3 dan ``semantic_weight``
0.4-0.7 (12 dari 80 konfigurasi grid). ``skill_prior`` 0 dan query teks CV tidak pernah
mencapai 1.00/1.00.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from agents.rag_agent import MIN_CHUNK_CHARS
from cv.profile import CVProfile
from cv.skills import CVSkills, build_cv_skills, match_skills, vocab_in_text
from repositories.job_repository import get_job_descriptions, get_jobs_by_ids


class MatchConfig(BaseModel):
    """Parameter pencocokan. Semua bisa diubah tanpa embed ulang, kecuali ``query``.

    Attributes:
        query: Teks yang di-embed: ringkasan profil atau teks CV mentah.
        aggregation: Skor lowongan dari chunk-nya: chunk terbaik (``max``) atau rata-rata
            ``top_chunks`` teratas (``mean``). Default ``max``: pada CV asli top-10 keduanya
            berisi lowongan yang sama, sedangkan ``max`` tidak butuh isian skor dan skor
            chunk terbaik tidak berubah saat ``k`` dinaikkan.
        top_chunks: Jumlah chunk yang dirata-rata (hanya untuk ``mean``).
        k: Jumlah chunk yang diambil dari Qdrant (dari 3.613).
        semantic_weight: Bobot komponen semantik; sisanya untuk skor skill.
        skill_prior: Penambah penyebut skor skill; makin besar, makin sedikit pengaruh
            lowongan yang hanya menyebut 1-2 skill. ``0`` = porsi mentah.
        semantic_floor: Skor cosine yang dianggap 0%: median cosine kandidat terendah di antara
            CV set eval (0,494), dibulatkan ke bawah ke kelipatan 0,05.
        semantic_ceiling: Skor cosine yang dianggap 100%: cosine teratas tertinggi di antara CV
            set eval (0,707), dibulatkan ke bawah. Skor teratas set eval menjadi 71-89%
            (batas lama 0,25-0,65: 84-92%, hampir semua lowongan tampak sangat cocok);
            urutan lowongan tidak berubah.
        limit: Jumlah lowongan yang dikembalikan.
    """

    query: Literal["profile", "cv_text"] = "profile"
    aggregation: Literal["mean", "max"] = "max"
    top_chunks: int = Field(default=3, ge=1)
    k: int = Field(default=400, ge=1)
    semantic_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    skill_prior: float = Field(default=2.0, ge=0.0)
    semantic_floor: float = 0.45
    semantic_ceiling: float = 0.70
    limit: int = Field(default=10, ge=1)


class JobMatch(BaseModel):
    """Satu lowongan hasil pencocokan, berisi data yang ditampilkan di kartu hasil."""

    job_id: int
    job_title: str
    company_name: str | None = None
    location: str | None = None
    work_type: str | None = None
    salary_min: int | None = None
    salary_max: int | None = None
    salary_period: str | None = None
    salary_currency: str | None = None
    score: float
    semantic: float
    semantic_scaled: float
    skill_portion: float | None = None
    skill_score: float = 0.0
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    """Hasil ``match_jobs``. ``error`` terisi jika gagal; ``matches`` kosong."""

    matches: list[JobMatch] = Field(default_factory=list)
    query_chars: int = 0
    embedding_calls: int = 0
    chunks_found: int = 0
    unique_jobs: int = 0
    error: str | None = None


@dataclass(frozen=True)
class JobSkillIndex:
    """Skill kosakata per lowongan; tidak bergantung pada CV, jadi dihitung sekali."""

    vocab: dict[int, frozenset[str]]
    descriptions: dict[int, str]
    frequency: dict[str, int]


@lru_cache(maxsize=4)
def load_job_skill_index(db_path: str) -> JobSkillIndex:
    """Bangun indeks skill semua lowongan (473 lowongan: sekitar 0,5 dtk), di-cache per proses.

    Database read-only dan hanya berubah saat seeding ulang (aplikasi di-restart), jadi
    cache seumur proses aman.
    """
    rows = get_job_descriptions(db_path)
    vocab = {job_id: vocab_in_text(text) for job_id, text in rows}
    frequency = Counter(name for names in vocab.values() for name in names)
    return JobSkillIndex(vocab=vocab, descriptions=dict(rows), frequency=dict(frequency))


def profile_query_text(profile: CVProfile) -> str:
    """Ringkasan profil yang di-embed: posisi tujuan, skill, jabatan, pendidikan.

    Nama, kontak, dan lokasi tidak dimasukkan: tidak relevan untuk kemiripan isi lowongan,
    dan chunk Qdrant hampir tidak pernah memuat nama kota (lokasi ada di kolom terpisah).
    """
    parts: list[str] = []
    if profile.target_role:
        parts.append(profile.target_role)
    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))
    titles = list(dict.fromkeys(exp.title for exp in profile.experiences if exp.title))
    if titles:
        parts.append("Experience: " + "; ".join(titles))
    education = " ".join(p for p in (profile.education_level, profile.education_major) if p)
    if education:
        parts.append("Education: " + education)
    return "\n".join(parts)


def aggregate_job_scores(
    hits: Iterable[tuple[int, float]],
    *,
    method: Literal["mean", "max"] = "mean",
    top_chunks: int = 3,
) -> dict[int, float]:
    """Gabungkan skor chunk menjadi satu skor per lowongan.

    Untuk ``mean``, lowongan yang chunk-nya terambil kurang dari ``top_chunks`` diisi dengan
    skor chunk terendah yang terambil dari Qdrant. Chunk lowongan itu yang tidak terambil
    pasti skornya tidak lebih tinggi dari nilai tersebut, jadi isian ini batas atas yang
    jujur. Tanpa isian, lowongan dengan satu chunk bagus akan mengalahkan lowongan dengan
    tiga chunk bagus.

    Args:
        hits: Pasangan ``(job_id, skor cosine)``.
        method: ``mean`` (rata-rata ``top_chunks`` teratas) atau ``max``.
        top_chunks: Jumlah chunk yang dirata-rata.

    Returns:
        ``{job_id: skor}``; kosong jika ``hits`` kosong.
    """
    by_job: dict[int, list[float]] = defaultdict(list)
    for job_id, score in hits:
        by_job[job_id].append(score)
    if not by_job:
        return {}
    if method == "max":
        return {job_id: max(scores) for job_id, scores in by_job.items()}
    pad = min(score for scores in by_job.values() for score in scores)
    result: dict[int, float] = {}
    for job_id, scores in by_job.items():
        top = sorted(scores, reverse=True)[:top_chunks]
        top += [pad] * (top_chunks - len(top))
        result[job_id] = sum(top) / top_chunks
    return result


def scale_semantic(score: float, floor: float, ceiling: float) -> float:
    """Petakan cosine ke 0..1 secara linear antara ``floor`` dan ``ceiling``, lalu dipotong."""
    if ceiling <= floor:
        raise ValueError("semantic_ceiling harus lebih besar dari semantic_floor")
    return min(1.0, max(0.0, (score - floor) / (ceiling - floor)))


def skill_score(matched: int, required: int, prior: float) -> float:
    """Skor skill 0..1: ``matched / (required + prior)``; 0 jika lowongan tidak menyebut skill.

    ``prior`` bekerja seperti menambah skill "tak diketahui" yang dianggap tidak dimiliki:
    1/1 dengan prior 2 menjadi 0,33, sedangkan 6/7 menjadi 0,67.
    """
    denominator = required + prior
    return matched / denominator if denominator > 0 else 0.0


def combine_score(semantic_scaled: float, skill: float, weight: float) -> float:
    """Skor akhir 0..1: rata-rata berbobot skor semantik terkalibrasi dan skor skill."""
    return weight * semantic_scaled + (1 - weight) * skill


def rank_jobs(
    query_vector: list[float],
    cv_skills: CVSkills,
    *,
    vector_store: Any,
    db_path: str | Path,
    config: MatchConfig,
) -> tuple[list[JobMatch], int, int]:
    """Urutkan lowongan untuk satu vektor query; tanpa embedding dan tanpa LLM.

    Dipisah dari ``match_jobs`` agar skrip pengukuran bisa mencoba banyak konfigurasi
    dengan satu vektor (satu kali bayar embedding).

    Args:
        query_vector: Hasil ``embed_query``.
        cv_skills: Hasil ``build_cv_skills(profile.skills)``.
        vector_store: Objek dengan ``similarity_search_with_score_by_vector``
            (``QdrantVectorStore`` atau stub saat test).
        db_path: Path SQLite.
        config: Parameter pencocokan.

    Returns:
        ``(matches, chunks_found, unique_jobs)``. ``matches`` urut skor menurun, maksimal
        ``config.limit``.

    Raises:
        Exception: Error Qdrant atau SQLite diteruskan; ``match_jobs`` yang menangkapnya.
    """
    chunks = vector_store.similarity_search_with_score_by_vector(query_vector, k=config.k)
    hits = [
        (int(doc.metadata["job_id"]), float(score))
        for doc, score in chunks
        if doc.metadata.get("job_id") is not None
        and len(doc.page_content.strip()) >= MIN_CHUNK_CHARS
    ]
    semantic = aggregate_job_scores(hits, method=config.aggregation, top_chunks=config.top_chunks)
    index = load_job_skill_index(str(Path(db_path).resolve()))

    scored: list[tuple[float, float, int, float, float, Any]] = []
    for job_id, raw in semantic.items():
        if job_id not in index.vocab:
            continue  # vektor basi: ID tidak ada lagi di SQLite
        skills = match_skills(
            cv_skills,
            index.vocab[job_id],
            index.descriptions[job_id],
            frequency=index.frequency,
        )
        scaled = scale_semantic(raw, config.semantic_floor, config.semantic_ceiling)
        skill = skill_score(len(skills.matched), skills.required, config.skill_prior)
        total = combine_score(scaled, skill, config.semantic_weight)
        scored.append((total, raw, job_id, scaled, skill, skills))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    top = scored[: config.limit]
    details = {job["id"]: job for job in get_jobs_by_ids(db_path, [item[2] for item in top])}

    matches = [
        JobMatch(
            job_id=job_id,
            job_title=details[job_id]["job_title"],
            company_name=details[job_id]["company_name"],
            location=details[job_id]["location"],
            work_type=details[job_id]["work_type"],
            salary_min=details[job_id]["salary_min"],
            salary_max=details[job_id]["salary_max"],
            salary_period=details[job_id]["salary_period"],
            salary_currency=details[job_id]["salary_currency"],
            score=total,
            semantic=raw,
            semantic_scaled=scaled,
            skill_portion=skills.portion,
            skill_score=skill,
            matched_skills=list(skills.matched),
            missing_skills=list(skills.missing),
        )
        for total, raw, job_id, scaled, skill, skills in top
        if job_id in details
    ]
    return matches, len(chunks), len(semantic)


def match_jobs(
    profile: CVProfile,
    *,
    embeddings: Embeddings,
    vector_store: Any,
    db_path: str | Path,
    cv_text: str | None = None,
    config: MatchConfig | None = None,
) -> MatchResult:
    """Cari lowongan paling cocok untuk profil CV. Tepat satu embedding, nol LLM.

    Args:
        profile: Profil (mungkin sudah dikoreksi pengguna).
        embeddings: Model embedding yang sama dengan koleksi Qdrant.
        vector_store: Vector store (lihat ``rank_jobs``).
        db_path: Path SQLite.
        cv_text: Wajib jika ``config.query == "cv_text"``.
        config: Parameter; default ``MatchConfig()``.

    Returns:
        ``MatchResult``. Tidak pernah melempar exception; kegagalan di ``error``.
        Hasil cukup di-cache per versi profil oleh pemanggil (``st.session_state``).
    """
    config = config or MatchConfig()
    text = (cv_text or "").strip() if config.query == "cv_text" else profile_query_text(profile)
    if not text:
        return MatchResult(error="Profil atau teks CV kosong")

    try:
        vector = embeddings.embed_query(text)
    except Exception as exc:  # noqa: BLE001 - tipe error provider embedding beragam
        return MatchResult(query_chars=len(text), error=f"Embedding error: {exc}")

    try:
        matches, chunks_found, unique_jobs = rank_jobs(
            vector,
            build_cv_skills(profile.skills),
            vector_store=vector_store,
            db_path=db_path,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001 - error Qdrant/SQLite dilaporkan, bukan dilempar
        return MatchResult(
            query_chars=len(text), embedding_calls=1, error=f"Pencarian error: {exc}"
        )

    return MatchResult(
        matches=matches,
        query_chars=len(text),
        embedding_calls=1,
        chunks_found=chunks_found,
        unique_jobs=unique_jobs,
    )
