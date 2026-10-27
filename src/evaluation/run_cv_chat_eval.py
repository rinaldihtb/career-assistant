"""Set eval Asisten Karier dengan CV aktif (tests/eval/cv_chat_cases.yaml), LLM dan Qdrant asli.

    make eval-cv-chat ARGS="--label nama"          semua kasus, satu kali
    make eval-cv-chat ARGS="--only C3 --repeat 3"  sebagian kasus, diulang (planner tidak
                                                   deterministik untuk pertanyaan ambigu)
    make eval-cv-chat ARGS="--rescore tests/eval/results/<file>.json"
                                                   nilai ulang hasil lama tanpa API

Profil CV fiktif (cv01..cv05) dan kandidat lowongannya dibaca dari fixture yang disebut di field
``profiles`` file kasus (tests/eval/cv_chat_profiles.json, diambil dari satu run ``make eval-cv``
tanpa nama, kontak, lokasi, dan riwayat kerja). Hasil Cocokkan CV dibangun ulang dengan
``rank_jobs`` dari kandidat tersimpan (``rebuild_matches``), jadi tidak ada embedding: yang
memanggil API hanya planner dan worker SQL/RAG. Input graph sama dengan halaman Asisten:
``messages`` (riwayat lalu pertanyaan), ``profile``, dan ``matches``. Field ``profile`` kasus
(opsional) menimpa sebagian field profil fixture, misalnya ``target_role`` dengan dua posisi;
hasil Cocokkan CV tetap milik fixture.

Yang dinilai per kasus:
- ``response``: jenis respons planner.
- ``route``: agent per sub-tugas (multiset, seperti ``make eval``).
- ``cv_match``: ID yang dikutip sama dengan ``CV_MATCH_LIMIT`` lowongan teratas, urut.
- ``cv_gap``: ``CV_GAP_CHECK`` skill kurang teratas muncul di jawaban.
- ``standalone_all`` / ``standalone_none`` / ``answer_*``: aturan sama dengan queries.yaml,
  untuk memeriksa personalisasi (posisi tujuan dari profil hanya saat pengguna merujuk dirinya).
- Sitasi: setiap ID yang dikutip berasal dari sub-hasil agent.

Isi jawaban SQL/RAG tidak dinilai di sini; itu tugas set eval utama (``make eval``). Kasus
``report_only`` (pertanyaan tentang CV saat CV belum ada) hanya dicetak dan disimpan, tidak
dihitung lulus/gagal.
"""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document

from src.agents.cv_agent import CV_MATCH_LIMIT
from src.agents.rag_agent import MIN_CHUNK_CHARS
from src.cv.gap import compute_skill_gap
from src.cv.match import MatchConfig, MatchResult, rank_jobs
from src.cv.profile import CVProfile
from src.cv.skills import build_cv_skills
from src.evaluation.checks import (
    CheckResult,
    RunOutput,
    cited_ids,
    extract_run,
    llm_call_budget,
    route_ok,
    run_checks,
)
from src.evaluation.run_eval import graph_input, run_metadata
from src.models.agent import AgentResult

CASES_PATH = Path("tests/eval/cv_chat_cases.yaml")
RESULTS_DIR = Path("tests/eval/results")
TRACKED_FILES = (
    "src/agents/supervisor_agent_graph.py",
    "src/agents/prompts.py",
    "src/agents/cv_agent.py",
    "src/evaluation/run_cv_chat_eval.py",
)
CV_GAP_CHECK = 3


class CandidateStore:
    """Stub vector store: satu chunk per kandidat tersimpan, skornya cosine ``max``.

    ``rank_jobs`` dengan agregasi ``max`` (default ``MatchConfig``) hanya memakai chunk
    terbaik per lowongan, jadi satu chunk per kandidat memberi urutan yang sama dengan
    pencarian aslinya.
    """

    def __init__(self, candidates: Sequence[Mapping[str, Any]]):
        self.hits = sorted(
            (
                (
                    Document(page_content="x" * MIN_CHUNK_CHARS, metadata={"job_id": c["job_id"]}),
                    c["max"],
                )
                for c in candidates
            ),
            key=lambda hit: hit[1],
            reverse=True,
        )

    def similarity_search_with_score_by_vector(self, vector, k=4, **kwargs):
        return self.hits[:k]


def rebuild_matches(
    profile: CVProfile, candidates: Sequence[Mapping[str, Any]], db_path: str | Path
) -> MatchResult:
    """Hasil Cocokkan CV dari kandidat tersimpan, dengan kode ``rank_jobs`` yang dipakai app.

    Raises:
        ValueError: ``MatchConfig()`` bukan agregasi ``max`` (stub hanya menyimpan ``max``).
    """
    config = MatchConfig()
    if config.aggregation != "max":
        raise ValueError("rebuild_matches hanya mendukung aggregation='max'")
    matches, chunks, unique = rank_jobs(
        [],
        build_cv_skills(profile.skills),
        vector_store=CandidateStore(candidates),
        db_path=db_path,
        config=config,
    )
    return MatchResult(matches=matches, chunks_found=chunks, unique_jobs=unique)


def load_cases(path: Path = CASES_PATH, only: Sequence[str] = ()) -> tuple[Path, list[dict]]:
    """Baca file kasus; ``only`` menyaring prefix ID. Returns ``(path JSON profil, kasus)``."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    if only:
        cases = [case for case in cases if case["id"].startswith(tuple(only))]
    return Path(data["profiles"]), cases


def load_profiles(path: Path, db_path: str | Path) -> dict[str, tuple[CVProfile, MatchResult]]:
    """Profil dan hasil Cocokkan CV per ID CV dari fixture ``profiles`` file kasus."""
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = {}
    for case in data["cases"]:
        profile = CVProfile(**case["profile"])
        profiles[case["id"]] = (profile, rebuild_matches(profile, case["candidates"], db_path))
    return profiles


def expected_cv(matches: MatchResult | None) -> dict[str, list[Any]]:
    """Jawaban agent CV yang benar: ID ``cv_match`` dan skill gap teratas."""
    if matches is None:
        return {"ids": [], "gap": []}
    gap = compute_skill_gap(matches.matches)
    return {
        "ids": [match.job_id for match in matches.matches[:CV_MATCH_LIMIT]],
        "gap": [skill.name for skill in gap.missing[:CV_GAP_CHECK]],
    }


def cv_checks(
    case: Mapping[str, Any], run: Mapping[str, Any], expected: Mapping[str, list[Any]]
) -> list[CheckResult]:
    """Pemeriksaan khusus set ini: respons, isi jawaban agent CV, dan sitasi."""
    want = [case["expected"].get("response", "answer"), *case["expected"].get("response_alt", [])]
    same = run["response"] in want
    results = [CheckResult("response", same, "" if same else f"dapat {run['response']}")]
    if "cv_match" in run["agents"]:
        ok = run["cv_ids"] == expected["ids"]
        results.append(
            CheckResult("cv_match", ok, "" if ok else f"{run['cv_ids']} != {expected['ids']}")
        )
    if "cv_gap" in run["agents"]:
        missing = [name for name in expected["gap"] if name not in run["answer"]]
        results.append(
            CheckResult("cv_gap", not missing, f"tidak ada {missing}" if missing else "")
        )
    invalid = [i for i in cited_ids(run["answer"]) if i not in set(run["valid_ids"])]
    results.append(
        CheckResult("citations", not invalid, f"tidak valid {invalid}" if invalid else "")
    )
    return results


def score(
    case: Mapping[str, Any], run: Mapping[str, Any], expected: Mapping[str, list[Any]]
) -> dict[str, Any]:
    """Nilai satu run; dipakai bersama oleh run baru dan ``--rescore``."""
    output = RunOutput(answer=run["answer"], agents=run["agents"], standalone=run["standalone"])
    checks = [*cv_checks(case, run, expected), *run_checks(case, output, {}, {})]
    report_only = bool(case["expected"].get("report_only"))
    routed = route_ok(case["expected"], run["agents"]) if not report_only else None
    passed = None if report_only else bool(routed) and all(check.passed for check in checks)
    return {
        "id": case["id"],
        "category": case["category"],
        "cv": case.get("cv"),
        "question": case["question"],
        "passed": passed,
        "route_ok": routed,
        "expected_response": case["expected"].get("response", "answer"),
        "expected_agents": case["expected"].get("agents", []),
        **run,
        "checks": [vars(check) for check in checks],
        "llm_budget": llm_call_budget(run["agents"]),
    }


def run_case(
    case: Mapping[str, Any],
    graph: Any,
    profiles: Mapping[str, tuple[CVProfile, MatchResult]],
) -> dict[str, Any]:
    """Jalankan graph supervisor sekali untuk satu kasus; error dicatat, tidak dilempar.

    Args:
        case: Satu kasus dari cv_chat_cases.yaml.
        graph: Graph supervisor yang punya ``invoke`` (``SupervisorAgent.graph``).
        profiles: Hasil ``load_profiles``; kasus ``cv: null`` berjalan tanpa CV.
    """
    profile, matches = profiles[case["cv"]] if case.get("cv") else (None, None)
    if profile is not None and case.get("profile"):
        profile = profile.model_copy(update=case["profile"])
    start = time.perf_counter()
    try:
        state = graph.invoke({**graph_input(case), "profile": profile, "matches": matches})
        output, response, crashed = extract_run(state), state.get("response", "-"), None
    except Exception as exc:  # noqa: BLE001 - satu kasus rusak tidak boleh menghentikan eval
        output, response = RunOutput(answer=""), "-"
        crashed = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - start
    subresults: list[AgentResult] = output.subresults
    planner = [call for call in output.usage.calls if call.step == "planner"]
    run = {
        "response": response,
        "agents": output.agents,
        "standalone": output.standalone,
        "answer": output.answer,
        "cv_ids": [i for sub in subresults if sub.source == "cv" for i in sub.job_ids],
        "valid_ids": sorted({i for sub in subresults for i in sub.job_ids}),
        "llm_calls": output.usage.llm_calls,
        "planner_tokens": sum(call.total_tokens for call in planner),
        "total_tokens": output.usage.total_tokens,
        "seconds": round(seconds, 2),
        "error": crashed or output.error,
        "sql": [sub.sql for sub in subresults if sub.sql],
    }
    return score(case, run, expected_cv(matches))


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Lulus per run, kasus yang selalu lulus di semua ulangan, dan biaya rata-rata."""
    scored = [r for r in records if r["passed"] is not None]
    by_case: dict[str, list[bool]] = defaultdict(list)
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in scored:
        by_case[record["id"]].append(record["passed"])
        by_category[record["category"]].append(record)
    with_cv = [r for r in records if r["cv"]]
    return {
        "runs": len(scored),
        "runs_passed": sum(r["passed"] for r in scored),
        "cases": len(by_case),
        "cases_always_passed": sum(all(results) for results in by_case.values()),
        "route_ok": sum(bool(r["route_ok"]) for r in scored),
        "avg_llm_calls": round(sum(r["llm_calls"] for r in scored) / max(len(scored), 1), 2),
        "avg_planner_tokens_cv": round(
            sum(r["planner_tokens"] for r in with_cv) / max(len(with_cv), 1)
        ),
        "avg_seconds": round(sum(r["seconds"] for r in scored) / max(len(scored), 1), 2),
        "by_category": {
            category: {"runs": len(items), "passed": sum(r["passed"] for r in items)}
            for category, items in by_category.items()
        },
    }


def print_record(record: Mapping[str, Any]) -> None:
    """Satu baris per run; pemeriksaan yang gagal dan pertanyaan mandiri di bawahnya."""
    status = {True: "LULUS", False: "GAGAL", None: "LAPOR"}[record["passed"]]
    print(
        f"{record['id']:<7} {record.get('cv') or '-':<5} {status:<6} "
        f"{record['response']:<12} {record['agents']!s:<24} "
        f"LLM {record['llm_calls']}/{record['llm_budget']}  "
        f"{record['planner_tokens']:>5} tok planner"
    )
    if record["passed"] is False or record["passed"] is None:
        for question in record["standalone"]:
            print(f"          > {question}")
    if record["route_ok"] is False:
        print(f"          - rute: harus {record['expected_agents']}")
    for check in record["checks"]:
        if not check["passed"]:
            print(f"          - {check['name']}: {check['detail']}")
    if record["error"]:
        print(f"          - error: {record['error']}")


def print_summary(summary: Mapping[str, Any]) -> None:
    print("\n" + "=" * 60)
    for category, item in summary["by_category"].items():
        print(f"{category:<6} {item['passed']:>3}/{item['runs']}")
    print("-" * 60)
    print(f"Run lulus             {summary['runs_passed']}/{summary['runs']}")
    print(f"Kasus selalu lulus    {summary['cases_always_passed']}/{summary['cases']}")
    print(f"Rute benar            {summary['route_ok']}/{summary['runs']}")
    print(f"Rata-rata LLM call    {summary['avg_llm_calls']}")
    print(f"Token planner (CV)    {summary['avg_planner_tokens_cv']}")
    print(f"Latensi rata-rata     {summary['avg_seconds']}s")


def save_results(records: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], path: Path) -> None:
    summary = summarize(records)
    print_summary(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "summary": summary, "cases": records}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nHasil disimpan: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval Asisten Karier dengan CV aktif.")
    parser.add_argument("--label", default="cv-chat")
    parser.add_argument("--only", nargs="*", default=[], help="Prefix ID kasus, mis. C3 C7-01.")
    parser.add_argument("--repeat", type=int, default=1, help="Jumlah run per kasus.")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--rescore", type=Path, help="JSON hasil lama yang dinilai ulang.")
    args = parser.parse_args()

    started = datetime.now(UTC).astimezone()
    db_path = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")
    profiles_path, cases = load_cases(args.cases, args.only)
    profiles = load_profiles(profiles_path, db_path)
    meta = {
        **run_metadata(args.label, started, args.cases),
        "file_sha256": file_hashes(),
        "profiles": str(profiles_path),
        "repeat": args.repeat,
    }

    if args.rescore:
        rescore_file(args, cases, profiles, meta, started)
        return
    load_dotenv()
    from src.containers import get_supervisor_agent  # import di sini: butuh .env

    graph = get_supervisor_agent().graph
    records = []
    for repeat in range(args.repeat):
        for case in cases:
            record = {**run_case(case, graph, profiles), "repeat": repeat + 1}
            print_record(record)
            records.append(record)
    save_results(records, meta, args.out / f"{args.label}-{started:%Y%m%d-%H%M}.json")


def rescore_file(
    args: argparse.Namespace,
    cases: Sequence[Mapping[str, Any]],
    profiles: Mapping[str, tuple[CVProfile, MatchResult]],
    meta: Mapping[str, Any],
    started: datetime,
) -> None:
    """Nilai ulang run lama dengan file kasus sekarang (tanpa LLM dan tanpa .env)."""
    old = json.loads(args.rescore.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in cases}
    run_keys = (
        "response", "agents", "standalone", "answer", "cv_ids", "valid_ids", "llm_calls",
        "planner_tokens", "total_tokens", "seconds", "error", "sql",
    )  # fmt: skip
    records = []
    for old_record in old["cases"]:
        case = by_id.get(old_record["id"])
        if case is None:
            continue
        matches = profiles[case["cv"]][1] if case.get("cv") else None
        run = {key: old_record[key] for key in run_keys}
        record = {**score(case, run, expected_cv(matches)), "repeat": old_record["repeat"]}
        print_record(record)
        records.append(record)
    label = f"{old['meta']['label']}-rescored"
    meta = {**meta, "label": label, "rescored_from": args.rescore.name}
    save_results(records, meta, args.out / f"{label}-{started:%Y%m%d-%H%M}.json")


def file_hashes() -> dict[str, str]:
    """Hash file yang menentukan perilaku Asisten ber-CV, supaya run bisa dibandingkan."""
    return {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
        for path in TRACKED_FILES
        if Path(path).exists()
    }


if __name__ == "__main__":
    main()
