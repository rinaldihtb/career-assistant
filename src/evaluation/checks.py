"""Pemeriksaan otomatis set eval (tests/eval/queries.yaml). Tidak ada LLM di file ini.

Semua fungsi di sini murni: menerima jawaban, ID, dan hasil gold SQL, lalu
mengembalikan lulus/gagal beserta alasannya. Runner (``run_eval.py``) yang
memanggil supervisor dan database; file ini bisa dites tanpa API key.

Tanpa LLM-as-judge: kunci jawaban dihitung dari gold SQL, jadi angka dan ID bisa
dicocokkan langsung. Hasilnya bisa diulang dan tidak menambah token.
LLM-as-judge baru layak jika muncul kriteria yang tidak bisa diukur seperti ini.
"""

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.models.agent import AgentResult, LLMUsage

# Sama dengan pola sitasi di rag_agent.py: "[ID:123]" atau "ID: 123".
_CITATION = re.compile(r"\bID:\s*(\d+)", re.IGNORECASE)
# Format angka di jawaban: "Rp6.812.500", "6.812.500", "15 juta", "6,8 jt", "473".
# Alternatif pertama (pemisah ribuan titik) dicoba lebih dulu agar "6.812.500"
# tidak terbaca sebagai 6.812.
_NUMBER = re.compile(
    r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+|\d+(?:[.,]\d+)?)\s*(juta|jt)?(?![\d])", re.IGNORECASE
)
_RUPIAH = re.compile(r"rp\s?\d|\d\s*(juta|jt)\b", re.IGNORECASE)
RELATIVE_TOLERANCE = 0.01

AGENT_LLM_BUDGET = {
    "sql": 3,
    "rag": 1,
    "cv_match": 0,
    "cv_gap": 0,
    "cv_role": 0,
    "skill_stats": 0,
    "data_source": 0,
    "job_detail": 0,
}


@dataclass
class CheckResult:
    """Hasil satu pemeriksaan. ``detail`` diisi alasan jika gagal."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class RunOutput:
    """Hal yang dinilai dari satu jalannya supervisor, apa pun bentuk graph-nya.

    Attributes:
        answer: Jawaban akhir yang dilihat pengguna.
        agents: Agent per sub-tugas, misalnya ``["sql", "rag"]``. ``[]`` = tanpa agent.
        standalone: Pertanyaan mandiri per sub-tugas.
        subresults: ``AgentResult`` per sub-tugas, sumber ``job_ids`` untuk sitasi.
        usage: Usage total (planner/router + semua agent + sintesis).
        error: Error yang dicatat supervisor.
    """

    answer: str
    agents: list[str] = field(default_factory=list)
    standalone: list[str] = field(default_factory=list)
    subresults: list[AgentResult] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)
    error: str | None = None


def extract_run(state: Mapping[str, Any]) -> RunOutput:
    """Ubah state akhir graph supervisor menjadi ``RunOutput``.

    Mendukung dua bentuk state:
        - Supervisor sekarang (satu rute): ``route``, ``standalone_question``, ``result``.
        - Supervisor multi-sub-tugas (Tugas B): ``subtasks`` berisi
          ``{"agent": ..., "question": ...}`` dan ``subresults`` berisi ``AgentResult``
          dengan urutan yang sama. Ini kontrak yang wajib dipenuhi graph baru.
    """
    result: AgentResult = state["result"]
    if "subtasks" in state:
        subtasks = list(state["subtasks"])
        return RunOutput(
            answer=result.answer,
            agents=[task["agent"] for task in subtasks],
            standalone=[task["question"] for task in subtasks],
            subresults=list(state.get("subresults", [])),
            usage=result.usage,
            error=result.error,
        )
    route = state.get("route")
    has_agent = route in AGENT_LLM_BUDGET
    return RunOutput(
        answer=result.answer,
        agents=[route] if has_agent else [],
        standalone=[state.get("standalone_question", "")],
        subresults=[result] if has_agent else [],
        usage=result.usage,
        error=result.error,
    )


# --- Helper ------------------------------------------------------------------


def cited_ids(answer: str) -> list[int]:
    """ID yang dikutip di jawaban, urut kemunculan pertama, tanpa duplikat."""
    return list(dict.fromkeys(int(match) for match in _CITATION.findall(answer)))


def split_citations(run: RunOutput) -> tuple[list[int], list[int], list[int]]:
    """Kelompokkan ID yang dikutip menurut agent yang menghasilkannya.

    Returns:
        ``(dari_sql, dari_rag, tidak_valid)``. ID tidak valid = dikutip di jawaban
        tetapi tidak ada di ``job_ids`` sub-hasil mana pun (halusinasi atau rusak
        saat sintesis). ID dari agent CV sah tetapi tidak masuk dua daftar pertama.
    """
    pools: dict[str, set[int]] = {"sql": set(), "rag": set(), "cv": set()}
    for sub in run.subresults:
        if sub.source in pools:
            pools[sub.source].update(sub.job_ids)
    cited = cited_ids(run.answer)
    from_sql = [job_id for job_id in cited if job_id in pools["sql"]]
    from_rag = [job_id for job_id in cited if job_id in pools["rag"]]
    valid = pools["sql"] | pools["rag"] | pools["cv"]
    invalid = [job_id for job_id in cited if job_id not in valid]
    return from_sql, from_rag, invalid


def extract_numbers(text: str) -> list[float]:
    """Ambil semua angka dari jawaban dalam satuan aslinya (Rupiah, jumlah).

    Teks sitasi ``[ID:x]`` dibuang dulu agar ID tidak terbaca sebagai angka
    jawaban. "6,8 juta" menjadi 6.800.000; "6.812.500" menjadi 6812500.
    """
    text = _CITATION.sub(" ", text)
    numbers = []
    for raw, unit in _NUMBER.findall(text):
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):
            value = float(raw.replace(".", ""))
        else:
            value = float(raw.replace(",", "."))
        numbers.append(value * 1_000_000 if unit else value)
    return numbers


def number_present(expected: float, numbers: Iterable[float]) -> bool:
    """True jika ``expected`` muncul di ``numbers`` dengan toleransi 1% (minimal 0,5).

    Toleransi relatif menerima pembulatan wajar ("Rp6,8 juta" untuk 6.812.500),
    sedangkan batas 0,5 membuat angka kecil (jumlah lowongan) harus persis.
    """
    tolerance = max(abs(expected) * RELATIVE_TOLERANCE, 0.5)
    return any(abs(value - expected) <= tolerance for value in numbers)


def route_ok(expected: Mapping[str, Any], actual_agents: Sequence[str]) -> bool:
    """Bandingkan multiset agent; urutan sub-tugas tidak dinilai."""
    options = [expected["agents"], *expected.get("agents_alt", [])]
    return any(Counter(option) == Counter(actual_agents) for option in options)


def llm_call_budget(agents: Sequence[str]) -> int:
    """Batas panggilan LLM: planner 1 + agent + sintesis 1 jika sub-tugas > 1."""
    return 1 + sum(AGENT_LLM_BUDGET[agent] for agent in agents) + (1 if len(agents) > 1 else 0)


def _contains_any(text: str, phrases: Iterable[str]) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in phrases if phrase.lower() in lowered]


def _job_matches(job: Mapping[str, Any], group: Mapping[str, Any]) -> bool:
    """Satu lowongan cocok dengan grup RAG (kata kunci judul/deskripsi dan lokasi).

    ``location_none`` dicek sebelum ``location_any``: lowongan di lokasi yang
    dikecualikan pengguna ("selain Jakarta") dihitung tidak relevan.
    """
    if "title_any" in group:
        text, keywords = job.get("job_title") or "", group["title_any"]
    else:
        text, keywords = job.get("job_description") or "", group["desc_any"]
    if not _contains_any(text, keywords):
        return False
    location = job.get("location") or ""
    if _contains_any(location, group.get("location_none", [])):
        return False
    locations = group.get("location_any")
    return not locations or bool(_contains_any(location, locations))


# --- Pemeriksaan per jenis ---------------------------------------------------


def check_sql_values(gold_rows: Sequence[Sequence[Any]], answer: str) -> CheckResult:
    """Setiap sel angka non-NULL dari hasil gold SQL wajib muncul di jawaban."""
    numbers = extract_numbers(answer)
    missing = [
        cell
        for row in gold_rows
        for cell in row
        if isinstance(cell, int | float) and not number_present(cell, numbers)
    ]
    return CheckResult(
        "sql_values", not missing, f"tidak ada di jawaban: {missing}" if missing else ""
    )


def check_sql_ids(gold_ids: Iterable[int], ids_from_sql: Iterable[int]) -> CheckResult:
    """ID yang dikutip dari sub-tugas SQL harus sama persis dengan gold."""
    gold, got = set(gold_ids), set(ids_from_sql)
    if gold == got:
        return CheckResult("sql_ids", True)
    return CheckResult("sql_ids", False, f"kurang {sorted(gold - got)}, lebih {sorted(got - gold)}")


def check_rag(
    groups: Sequence[Mapping[str, Any]],
    ids_from_rag: Sequence[int],
    jobs_by_id: Mapping[int, Mapping[str, Any]],
) -> CheckResult:
    """Tiap grup terpenuhi minimal ``min_ids`` lowongan; tiap ID RAG cocok dengan suatu grup."""
    problems = []
    for index, group in enumerate(groups):
        matched = [
            i for i in ids_from_rag if i in jobs_by_id and _job_matches(jobs_by_id[i], group)
        ]
        if len(matched) < group["min_ids"]:
            problems.append(f"grup {index + 1}: {len(matched)} cocok < {group['min_ids']}")
    off_topic = [
        i
        for i in ids_from_rag
        if i not in jobs_by_id or not any(_job_matches(jobs_by_id[i], g) for g in groups)
    ]
    if off_topic:
        problems.append(f"tidak relevan: {off_topic}")
    return CheckResult("rag", not problems, "; ".join(problems))


def run_checks(
    case: Mapping[str, Any],
    run: RunOutput,
    gold: Mapping[str, list[Any]],
    jobs_by_id: Mapping[int, Mapping[str, Any]],
) -> list[CheckResult]:
    """Jalankan semua pemeriksaan di ``case["checks"]``.

    Args:
        case: Satu kasus dari queries.yaml.
        run: Hasil supervisor untuk kasus ini.
        gold: ``{"sql_values": [rows, ...], "sql_ids": [[id, ...], ...]}`` hasil gold SQL.
        jobs_by_id: Detail lowongan yang dikutip (judul, deskripsi, lokasi) untuk cek RAG.
    """
    checks = case.get("checks", {})
    answer = run.answer
    from_sql, from_rag, _ = split_citations(run)
    results: list[CheckResult] = []

    for rows in gold.get("sql_values", []):
        results.append(check_sql_values(rows, answer))
    for ids in gold.get("sql_ids", []):
        results.append(check_sql_ids(ids, from_sql))
    if "rag" in checks:
        results.append(check_rag(checks["rag"], from_rag, jobs_by_id))
    if checks.get("no_ids"):
        ids = cited_ids(answer)
        results.append(CheckResult("no_ids", not ids, f"mengutip {ids}" if ids else ""))
    if "answer_any" in checks:
        ok = bool(_contains_any(answer, checks["answer_any"]))
        results.append(
            CheckResult("answer_any", ok, "" if ok else f"tidak ada {checks['answer_any']}")
        )
    if "answer_none" in checks:
        found = _contains_any(answer, checks["answer_none"])
        results.append(CheckResult("answer_none", not found, f"memuat {found}" if found else ""))
    if checks.get("no_rupiah"):
        match = _RUPIAH.search(answer)
        results.append(
            CheckResult("no_rupiah", match is None, f"memuat '{match.group(0)}'" if match else "")
        )
    standalone = " | ".join(run.standalone)
    if "standalone_any" in checks:
        ok = bool(_contains_any(standalone, checks["standalone_any"]))
        results.append(CheckResult("standalone_any", ok, "" if ok else f"mandiri: {standalone!r}"))
    if "standalone_none" in checks:
        found = _contains_any(standalone, checks["standalone_none"])
        results.append(
            CheckResult("standalone_none", not found, f"mandiri memuat {found}" if found else "")
        )
    if "standalone_all" in checks:
        missing = [
            group for group in checks["standalone_all"] if not _contains_any(standalone, group)
        ]
        results.append(
            CheckResult(
                "standalone_all",
                not missing,
                f"mandiri {standalone!r} tanpa {missing}" if missing else "",
            )
        )
    return results
