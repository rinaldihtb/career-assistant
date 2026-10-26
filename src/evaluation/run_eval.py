"""Jalankan set eval (tests/eval/queries.yaml) terhadap supervisor dengan LLM dan Qdrant asli.

    make eval ARGS="--label baseline"     semua kasus (butuh .env)
    make eval ARGS="--only T3 T6b-01"     sebagian kasus (prefix kategori atau ID)
    make eval ARGS="--rescore tests/eval/results/<file>.json"
                                          nilai ulang hasil lama dengan queries.yaml terbaru,
                                          tanpa LLM (dipakai saat set eval dikoreksi)

Alur per kasus: hitung gold SQL di database read-only -> jalankan graph
supervisor -> nilai dengan ``evaluation.checks`` -> cetak dan simpan JSON di
tests/eval/results/. File JSON menyimpan commit dan hash file supervisor/prompt,
supaya baseline dan hasil perbaikan bisa dibandingkan secara jujur.

Input graph sama dengan halaman Asisten: ``messages`` berisi riwayat kasus lalu
pertanyaannya. State akhir (``subtasks``, ``subresults``, ``result``) dibaca utuh
karena penilaian butuh rute dan pertanyaan mandiri per sub-tugas, bukan hanya jawaban.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.evaluation.checks import (
    RunOutput,
    cited_ids,
    extract_run,
    llm_call_budget,
    route_ok,
    run_checks,
    split_citations,
)
from src.models.agent import AgentResult
from src.repositories.job_repository import connect_readonly, get_jobs_by_ids

CASES_PATH = Path("tests/eval/queries.yaml")
RESULTS_DIR = Path("tests/eval/results")
TRACKED_FILES = (
    "src/agents/supervisor_agent_graph.py",
    "src/agents/sql_agent.py",
    "src/agents/rag_agent.py",
    "src/agents/prompts.py",
)


def load_cases(path: Path = CASES_PATH, only: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Baca kasus dari YAML; ``only`` menyaring berdasarkan prefix ID (mis. "T3", "T6b-01")."""
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]
    if only:
        cases = [case for case in cases if case["id"].startswith(tuple(only))]
    return cases


def compute_gold(case: Mapping[str, Any], db_path: str | Path) -> dict[str, list[Any]]:
    """Jalankan gold SQL kasus pada koneksi read-only.

    Returns:
        ``{"sql_values": [rows, ...], "sql_ids": [[id, ...], ...]}``.
    """
    checks = case.get("checks", {})
    with closing(connect_readonly(db_path)) as conn:
        return {
            "sql_values": [conn.execute(sql).fetchall() for sql in checks.get("sql_values", [])],
            "sql_ids": [[row[0] for row in conn.execute(sql)] for sql in checks.get("sql_ids", [])],
        }


def score(
    case: Mapping[str, Any], run: RunOutput, db_path: str | Path, *, crashed: bool = False
) -> dict[str, Any]:
    """Nilai satu jawaban terhadap kasus: gold SQL, sitasi, dan semua pemeriksaan.

    Dipakai bersama oleh ``run_case`` (jawaban baru) dan ``rescore_record``
    (jawaban lama), sehingga keduanya dinilai dengan aturan yang persis sama.
    """
    gold = compute_gold(case, db_path)
    from_sql, from_rag, invalid = split_citations(run)
    jobs = get_jobs_by_ids(db_path, from_rag) if from_rag else []
    checks = run_checks(case, run, gold, {job["id"]: job for job in jobs})
    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "passed": not crashed and all(check.passed for check in checks),
        "route_ok": route_ok(case["expected"], run.agents),
        "expected_agents": case["expected"]["agents"],
        "agents": run.agents,
        "standalone": run.standalone,
        "answer": run.answer,
        "cited_ids": cited_ids(run.answer),
        "cited_sql": from_sql,
        "cited_rag": from_rag,
        "invalid_citations": invalid,
        "checks": [vars(check) for check in checks],
        "llm_budget": llm_call_budget(case["expected"]["agents"]),
    }


def graph_input(case: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Riwayat kasus (format ``st.session_state``) diikuti pertanyaan kasus sebagai pesan."""
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in case.get("history", [])
    ]
    return {"messages": [*history, {"role": "user", "content": case["question"]}]}


def run_case(case: Mapping[str, Any], graph: Any, db_path: str | Path) -> dict[str, Any]:
    """Jalankan satu kasus dan kembalikan catatan hasil yang siap disimpan ke JSON.

    Args:
        case: Satu kasus dari queries.yaml.
        graph: Graph supervisor yang punya ``invoke`` (``SupervisorAgent.graph``).
        db_path: SQLite untuk gold SQL dan detail lowongan yang dikutip.

    Exception dari graph tidak menghentikan eval: kasus dicatat gagal dengan
    pesan error-nya, lalu kasus berikutnya tetap jalan.
    """
    compute_gold(case, db_path)  # gold SQL rusak harus gagal sebelum memanggil LLM
    start = time.perf_counter()
    try:
        state = graph.invoke(graph_input(case))
        run = extract_run(state)
        crashed = None
    except Exception as exc:  # noqa: BLE001 - satu kasus rusak tidak boleh menghentikan eval
        run = RunOutput(answer="")
        crashed = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - start

    record = score(case, run, db_path, crashed=crashed is not None)
    return {
        **record,
        "llm_calls": run.usage.llm_calls,
        "within_budget": run.usage.llm_calls <= record["llm_budget"],
        "total_tokens": run.usage.total_tokens,
        "models": sorted({call.model for call in run.usage.calls}),
        "seconds": round(seconds, 2),
        "error": crashed or run.error,
        "sql": [sub.sql for sub in run.subresults if sub.sql],
    }


def rescore_record(
    case: Mapping[str, Any], old: Mapping[str, Any], db_path: str | Path
) -> dict[str, Any]:
    """Nilai ulang catatan run lama dengan kasus versi sekarang, TANPA memanggil LLM.

    Jawaban, rute, dan ID yang dikutip diambil dari catatan lama; sub-hasil
    dibangun ulang dari ``cited_sql``/``cited_rag`` agar ``split_citations``
    memberi pembagian yang sama. Biaya (LLM call, token, waktu) tidak berubah.
    """
    run = RunOutput(
        answer=old["answer"],
        agents=list(old["agents"]),
        standalone=list(old["standalone"]),
        subresults=[
            AgentResult(answer="", source="sql", job_ids=old["cited_sql"]),
            AgentResult(answer="", source="rag", job_ids=old["cited_rag"]),
        ],
    )
    crashed = not old["answer"] and bool(old.get("error"))
    record = score(case, run, db_path, crashed=crashed)
    keep = ("llm_calls", "total_tokens", "models", "seconds", "error")
    return {
        **record,
        **{key: old[key] for key in keep},
        "within_budget": old["llm_calls"] <= record["llm_budget"],
        "sql": old.get("sql", []),
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Metrik total dan per kategori."""
    n = len(records)
    seconds = sorted(record["seconds"] for record in records)
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)
    return {
        "cases": n,
        "passed": sum(r["passed"] for r in records),
        "route_ok": sum(r["route_ok"] for r in records),
        "citations_valid": sum(not r["invalid_citations"] for r in records),
        "within_budget": sum(r["within_budget"] for r in records),
        "avg_llm_calls": round(sum(r["llm_calls"] for r in records) / n, 2),
        "total_tokens": sum(r["total_tokens"] for r in records),
        "avg_seconds": round(sum(seconds) / n, 2),
        "p95_seconds": seconds[min(n - 1, int(0.95 * n))],
        "by_category": {
            category: {
                "cases": len(items),
                "passed": sum(r["passed"] for r in items),
                "route_ok": sum(r["route_ok"] for r in items),
            }
            for category, items in by_category.items()  # urutan sesuai queries.yaml
        },
    }


def run_metadata(label: str, started: datetime, cases_path: Path = CASES_PATH) -> dict[str, Any]:
    """Commit, status working tree, dan hash file yang menentukan perilaku supervisor."""

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
    }


def print_record(record: Mapping[str, Any]) -> None:
    """Satu baris per kasus; pemeriksaan yang gagal dicetak di bawahnya."""
    status = "LULUS" if record["passed"] else "GAGAL"
    route = "ok" if record["route_ok"] else f"harus {record['expected_agents']}"
    print(
        f"{record['id']:<9} {status:<6} rute {record['agents']!s:<16} {route:<22} "
        f"LLM {record['llm_calls']}/{record['llm_budget']}  {record['total_tokens']:>5} tok  "
        f"{record['seconds']:>5.1f}s"
    )
    for check in record["checks"]:
        if not check["passed"]:
            print(f"          - {check['name']}: {check['detail']}")
    if record["invalid_citations"]:
        print(f"          - sitasi tidak valid: {record['invalid_citations']}")
    if record["error"]:
        print(f"          - error: {record['error']}")


def print_summary(summary: Mapping[str, Any]) -> None:
    """Ringkasan per kategori dan total."""
    print("\n" + "=" * 60)
    print(f"{'Kategori':<10} {'lulus':>7} {'rute':>7}")
    for category, item in summary["by_category"].items():
        print(
            f"{category:<10} {item['passed']:>3}/{item['cases']:<3} {item['route_ok']:>3}/{item['cases']:<3}"
        )
    n = summary["cases"]
    print("-" * 60)
    print(f"Lulus semua cek      {summary['passed']}/{n}")
    print(f"Rute benar           {summary['route_ok']}/{n}")
    print(f"Sitasi valid         {summary['citations_valid']}/{n}")
    print(f"Dalam batas LLM call {summary['within_budget']}/{n}")
    print(f"Rata-rata LLM call   {summary['avg_llm_calls']}")
    print(f"Total token          {summary['total_tokens']}")
    print(f"Latensi rata-rata    {summary['avg_seconds']}s | p95 {summary['p95_seconds']}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval Career Assistant supervisor.")
    parser.add_argument("--label", default="run", help="Nama hasil, mis. baseline atau planner-v1.")
    parser.add_argument("--only", nargs="*", default=[], help="Prefix ID kasus, mis. T3 T6b-01.")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--rescore", type=Path, help="JSON hasil lama yang dinilai ulang tanpa LLM."
    )
    args = parser.parse_args()

    started = datetime.now(UTC).astimezone()  # waktu lokal dengan zona waktu
    if args.rescore:
        rescore_file(args, started)
        return
    load_dotenv()
    from src.containers import get_supervisor_agent  # import di sini: butuh .env

    graph = get_supervisor_agent().graph
    db_path = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")
    cases = load_cases(args.cases, args.only)

    records = []
    for case in cases:
        record = run_case(case, graph, db_path)
        print_record(record)
        records.append(record)
    meta = run_metadata(args.label, started, args.cases)
    save_results(records, meta, args.out / f"{args.label}-{started:%Y%m%d-%H%M}.json")


def rescore_file(args: argparse.Namespace, started: datetime) -> None:
    """Nilai ulang file hasil lama dengan queries.yaml sekarang (tanpa LLM, tanpa .env)."""
    old = json.loads(args.rescore.read_text(encoding="utf-8"))
    old_records = {record["id"]: record for record in old["cases"]}
    db_path = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")
    cases = [case for case in load_cases(args.cases, args.only) if case["id"] in old_records]

    records = []
    for case in cases:
        record = rescore_record(case, old_records[case["id"]], db_path)
        print_record(record)
        records.append(record)
    label = f"{old['meta']['label']}-rescored" if args.label == "run" else args.label
    meta = {
        **run_metadata(label, started, args.cases),
        "rescored_from": args.rescore.name,
        "original_meta": old["meta"],
    }
    save_results(records, meta, args.out / f"{label}-{started:%Y%m%d-%H%M}.json")


def save_results(records: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], path: Path) -> None:
    """Cetak ringkasan lalu simpan JSON berisi meta, ringkasan, dan semua kasus."""
    summary = summarize(records)
    print_summary(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "summary": summary, "cases": records}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nHasil disimpan: {path}")


if __name__ == "__main__":
    main()
