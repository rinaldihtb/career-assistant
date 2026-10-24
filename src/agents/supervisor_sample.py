"""Uji manual supervisor dengan LLM, Qdrant, dan data/database.sqlite asli.

    make sample-supervisor   routing + jawaban lengkap untuk pertanyaan uji (butuh .env)
    make supervisor-graph    cetak diagram Mermaid graph (tanpa API key)

Rute yang diharapkan ditulis sebelum hasil agent dilihat. Pertanyaan yang
bisa masuk dua rute sengaja ikut diuji (gaji di atas angka tertentu -> sql).
Kasus lanjutan dinilai lulus jika rute benar, pertanyaan mandiri memuat semua
kata wajib, dan tidak memuat kata yang dilarang (konteks lama yang seharusnya
diganti, atau konteks yang tidak boleh ditempel ke pertanyaan baru).

Rute di sini = daftar agent dari sub-tugas planner; "general" berarti tanpa agent.
Evaluasi lengkap, termasuk pertanyaan majemuk, ada di ``make eval``.
"""

import os
import sys
import time

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from agents.supervisor_agent import SUPERVISOR_GRAPH, SupervisorContext

load_dotenv()
DB_PATH = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")

# (pertanyaan, rute yang diharapkan)
QUESTIONS = [
    ("Halo, selamat pagi!", "general"),
    ("Terima kasih!", "general"),
    ("Resep nasi goreng yang enak", "general"),
    ("Hapus semua data lowongan", "general"),
    ("Berapa jumlah lowongan di Jakarta?", "sql"),
    ("Berapa rata-rata gaji data analyst?", "sql"),
    ("Perusahaan mana yang paling banyak membuka lowongan?", "sql"),
    ("Lowongan data analyst di Jakarta yang gajinya di atas 10 juta", "sql"),
    ("Lowongan data analyst di Jakarta", "rag"),
    ("Saya lulusan akuntansi, lowongan apa yang cocok?", "rag"),
    ("Lowongan yang membutuhkan skill Python", "rag"),
]

# (riwayat, pertanyaan lanjutan, rute, kata wajib ada, kata dilarang ada) di pertanyaan mandiri.
# Kasus 3-5 tidak dipakai sebagai contoh di prompt planner, supaya menguji aturan, bukan
# hafalan contoh.
FOLLOW_UPS = [
    (
        [
            {"role": "user", "content": "Lowongan data analyst di Jakarta"},
            {"role": "assistant", "content": "Ada 3 lowongan data analyst di Jakarta: ..."},
        ],
        "Kalau yang di Surabaya?",
        "rag",
        ("data analyst", "surabaya"),
        ("jakarta",),
    ),
    (
        [
            {"role": "user", "content": "Berapa jumlah lowongan di Bandung?"},
            {"role": "assistant", "content": "Ada 12 lowongan di Bandung."},
        ],
        "Rata-rata gajinya berapa?",
        "sql",
        ("bandung", "gaji"),
        (),
    ),
    (
        [
            {"role": "user", "content": "Lowongan graphic designer di Bandung"},
            {"role": "assistant", "content": "Ada 2 lowongan graphic designer di Bandung: ..."},
        ],
        "Kalau di Jakarta ada juga?",
        "rag",
        ("design", "jakarta"),
        ("bandung",),
    ),
    (
        [
            {"role": "user", "content": "Berapa lowongan HR di Surabaya?"},
            {"role": "assistant", "content": "Ada 5 lowongan HR di Surabaya."},
        ],
        "Gaji tertingginya berapa?",
        "sql",
        ("hr", "surabaya", "gaji"),
        (),
    ),
    (
        [
            {"role": "user", "content": "Berapa jumlah lowongan di Bandung?"},
            {"role": "assistant", "content": "Ada 10 lowongan di Bandung."},
        ],
        "Berapa total semua lowongan yang ada di database?",
        "sql",
        ("total",),
        ("bandung",),
    ),
]


def _route(state) -> str:
    """Rute ringkas dari sub-tugas planner: "sql", "rag", "sql+rag", atau "general"."""
    return "+".join(task["agent"] for task in state["subtasks"]) or "general"


def _standalone(state) -> str:
    """Gabungan pertanyaan mandiri semua sub-tugas."""
    return " | ".join(task["question"] for task in state["subtasks"])


def _invoke(question, history, context):
    start = time.perf_counter()
    state = SUPERVISOR_GRAPH.invoke({"question": question, "history": history}, context=context)
    return state, time.perf_counter() - start


def _print_case(question, state, elapsed, check):
    result = state["result"]
    print("=" * 100)
    print(f"Q: {question}")
    print(f"Rute: {_route(state)} | mandiri: {_standalone(state)!r} | {check}")
    print(f"Jawaban:\n{result.answer}")
    print(f"job_ids {result.job_ids} | sql {result.sql!r} | error {result.error}")
    print(
        f"Waktu {elapsed:.2f}s | LLM calls {result.usage.llm_calls} | "
        f"token {result.usage.total_tokens}"
    )


def run_supervisor_check(llm: BaseChatModel, vector_store: VectorStore) -> None:
    """Jalankan graph untuk semua kasus uji dan cetak akurasi rute serta biaya."""
    context = SupervisorContext(llm=llm, vector_store=vector_store, db_path=DB_PATH)
    summary = []

    for question, expected in QUESTIONS:
        state, elapsed = _invoke(question, [], context)
        check = "OK" if _route(state) == expected else f"SALAH (harusnya {expected})"
        summary.append((question, state, elapsed, check))
        _print_case(question, state, elapsed, check)

    for history, question, expected, required, forbidden in FOLLOW_UPS:
        state, elapsed = _invoke(question, history, context)
        rewritten = _standalone(state).lower()
        route_ok = _route(state) == expected
        words_ok = all(word in rewritten for word in required) and not any(
            word in rewritten for word in forbidden
        )
        check = (
            "OK"
            if route_ok and words_ok
            else f"SALAH (rute {expected}, wajib {required}, dilarang {forbidden})"
        )
        summary.append((f"[lanjutan] {question}", state, elapsed, check))
        _print_case(question, state, elapsed, check)

    print("\n" + "=" * 100)
    print(f"{'Pertanyaan':<55} {'rute':<8} {'detik':>6} {'calls':>5} {'token':>6}  cek")
    for question, state, elapsed, check in summary:
        usage = state["result"].usage
        print(
            f"{question[:55]:<55} {_route(state):<8} {elapsed:>6.2f} {usage.llm_calls:>5} "
            f"{usage.total_tokens:>6}  {check}"
        )
    correct = sum(item[3] == "OK" for item in summary)
    avg_time = sum(item[2] for item in summary) / len(summary)
    total_tokens = sum(item[1]["result"].usage.total_tokens for item in summary)
    print(
        f"\nBenar {correct}/{len(summary)} | rata-rata waktu {avg_time:.2f}s | "
        f"total token {total_tokens}"
    )


if __name__ == "__main__":
    if "--graph" in sys.argv:
        print(SUPERVISOR_GRAPH.get_graph().draw_mermaid())
    else:
        from config.database import VECTOR_DB  # import di sini: --graph tidak butuh .env
        from config.llm import LLM_MODEL

        run_supervisor_check(LLM_MODEL, VECTOR_DB)
