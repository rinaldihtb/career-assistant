# Career Assistant

Asisten karier untuk pencari kerja Indonesia berbasis ±473 lowongan kerja 2025.
Multi-agent: supervisor + RAG agent + SQL agent, REST API, dan UI Streamlit.

## Menjalankan

```bash
source venv/bin/activate
pip install -r requirements-dev.txt
make run
```

Perintah lain: `make help`.

| Perintah | Fungsi | Butuh API key |
|---|---|---|
| `make run` | Jalankan Streamlit (wajib lewat make: butuh `PYTHONPATH` root) | ya |
| `make test` | Unit test dengan LLM palsu | tidak |
| `make lint` | Ruff untuk agent, repository, model, dan test | tidak |
| `make sample-sql` | Cek validasi dan eksekusi SQL ke database asli | tidak |
| `make sample-sql-llm` | Smoke test SQL agent dengan LLM sungguhan | ya |
| `make sample-rag` | Retrieval ke Qdrant asli: skor, threshold, filter lokasi | ya (embedding saja) |
| `make sample-rag-llm` | Smoke test RAG agent: jawaban, sitasi, relevansi ID dikutip | ya |
| `make sample-supervisor` | Smoke test routing supervisor dan pertanyaan lanjutan | ya |
| `make supervisor-graph` | Cetak diagram graph supervisor (Mermaid) | tidak |

## Struktur agent

| File | Isi |
|---|---|
| `src/models/agent.py` | `AgentResult` dan `LLMUsage`: bentuk output semua agent |
| `src/agents/prompts.py` | Semua prompt dan pesan tetap, tanpa logika |
| `src/repositories/job_repository.py` | Koneksi SQLite read-only |
| `src/agents/sql_agent.py` | SQL agent: pertanyaan statistik |
| `src/agents/rag_agent.py` | RAG agent: pencarian lowongan dengan sitasi `[ID:x]` |
| `src/agents/supervisor_agent.py` | Supervisor LangGraph: router + memanggil agent |
| `src/agents/*_sample.py` | Smoke test manual terhadap data dan LLM asli |

Nama file Python wajib memakai underscore (`job_repository.py`), bukan titik, agar bisa di-import.

## Supervisor (pintu masuk UI)

UI cukup memanggil satu fungsi. Riwayat chat dikirim apa adanya dari
`st.session_state`, **tanpa** pertanyaan saat ini.

```python
import os

from config.database import VECTOR_DB
from config.llm import LLM_MODEL
from agents.supervisor_agent import run_supervisor

DB_PATH = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")

result = run_supervisor(
    "Kalau yang di Surabaya?",
    llm=LLM_MODEL,
    vector_store=VECTOR_DB,
    db_path=DB_PATH,
    history=[
        {"role": "user", "content": "Lowongan data analyst di Jakarta"},
        {"role": "assistant", "content": "Ada 3 lowongan ..."},
    ],
)
```

Graph LangGraph (`make supervisor-graph`):

```
START -> router -> sql_agent | rag_agent | general -> finalize -> END
```

| Node | Isi | Panggilan LLM |
|---|---|---|
| `router` | Structured output: pilih rute + tulis ulang pertanyaan lanjutan jadi mandiri | 1 |
| `sql_agent` | `run_sql_agent` | 1-3 |
| `rag_agent` | `run_rag_agent` | 0-1 |
| `general` | Pesan tetap untuk sapaan / di luar topik | 0 |
| `finalize` | Jumlahkan token router + agent, gabungkan error | 0 |

Jawaban agent diteruskan apa adanya, tidak dirangkum ulang oleh LLM supervisor,
agar angka SQL dan sitasi `[ID:x]` tidak berubah. Jika router gagal, pertanyaan
diteruskan ke RAG (lihat `FALLBACK_ROUTE`).

## Kontrak agent

Setiap agent adalah satu fungsi biasa; agent tidak tahu apa-apa tentang LangGraph state.

```python
result = run_sql_agent(question, llm=LLM_MODEL, db_path=DB_PATH)
result = run_rag_agent(question, llm=LLM_MODEL, vector_store=VECTOR_DB, db_path=DB_PATH)
```

Aturan yang dijamin oleh setiap `run_*` (termasuk `run_supervisor`):

1. **Input**: `run_*_agent` menerima satu pertanyaan *mandiri* (`str`). Pertanyaan
   lanjutan ditulis ulang oleh router supervisor.
2. **Dependency disuntikkan**: `llm`, `vector_store`, dan `db_path` dikirim lewat
   parameter, sehingga test bisa memakai LLM palsu.
3. **Tidak pernah melempar exception.** Jika gagal, `answer` berisi pesan tetap
   yang aman ditampilkan ke user dan `error` berisi detail teknis untuk log.
4. **Output** selalu `AgentResult`:

| Field | Isi |
|---|---|
| `answer` | Teks untuk ditampilkan ke user |
| `source` | `"sql"`, `"rag"`, atau `"general"` (pesan tetap supervisor) |
| `job_ids` | ID lowongan terkait (untuk kartu lowongan di UI) |
| `rows` | Data mentah (hasil query / lowongan yang dikutip) |
| `sql` | Query yang dieksekusi (SQL agent saja) |
| `usage` | `llm_calls`, `input_tokens`, `output_tokens`, `total_tokens` |
| `error` | `None` jika sukses |

Jumlah panggilan LLM SQL agent: 2 normal, 3 jika retry, 1 jika hasil kosong
atau pertanyaan tidak relevan (misalnya permintaan menghapus data).

RAG agent:

- Retrieval tanpa LLM chat: filter lokasi dari kamus kolom `location`,
  threshold skor 0,45 (hanya untuk pertanyaan tanpa lokasi), top 5 lowongan.
- Tanpa kandidat: pesan tetap, 0 panggilan LLM. Ada kandidat: 1 panggilan LLM.
- **Guard sitasi**: `job_ids` dan `rows` hanya berisi ID yang dikutip jawaban
  (`[ID:x]`) DAN ada di kandidat. ID lain dibuang dan dicatat di `error`.
- `retrieve_jobs` dan `answer_from_jobs` bisa dipakai terpisah (misalnya untuk
  fitur Cocokkan CV).
