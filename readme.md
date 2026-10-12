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
| `make sample-sql` | Cek validasi dan eksekusi SQL ke database asli | tidak |
| `make sample-sql-llm` | Smoke test SQL agent dengan LLM sungguhan | ya |

## Struktur agent

| File | Isi |
|---|---|
| `src/models/agent.py` | `AgentResult` dan `LLMUsage`: bentuk output semua agent |
| `src/agents/prompts.py` | Semua prompt dan pesan tetap, tanpa logika |
| `src/repositories/job_repository.py` | Koneksi SQLite read-only |
| `src/agents/sql_agent.py` | SQL agent: pertanyaan statistik |
| `src/agents/rag_agent.py` | RAG agent: pencarian lowongan |
| `src/agents/supervisor_agent.py` | Router: memilih agent |

Nama file Python wajib memakai underscore (`job_repository.py`), bukan titik, agar bisa di-import.

## Kontrak untuk supervisor

Setiap agent adalah satu fungsi biasa. Supervisor cukup memanggilnya; agent tidak
tahu apa-apa tentang LangGraph state.

```python
import os

from config.llm import LLM_MODEL
from agents.sql_agent import run_sql_agent

DB_PATH = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")

result = run_sql_agent("Berapa lowongan di Jakarta?", llm=LLM_MODEL, db_path=DB_PATH)
```

Aturan yang dijamin oleh setiap `run_*_agent`:

1. **Input**: satu pertanyaan *mandiri* (`str`). Pertanyaan lanjutan seperti
   "yang di Surabaya saja" harus ditulis ulang oleh supervisor menjadi kalimat
   lengkap sebelum diteruskan, karena agent tidak membaca riwayat chat.
2. **Dependency disuntikkan**: `llm` (dan nanti `vector_store` untuk RAG) dikirim
   lewat parameter, sehingga test bisa memakai LLM palsu.
3. **Tidak pernah melempar exception.** Jika gagal, `answer` berisi pesan tetap
   yang aman ditampilkan ke user dan `error` berisi detail teknis untuk log.
4. **Output** selalu `AgentResult`:

| Field | Isi |
|---|---|
| `answer` | Teks untuk ditampilkan ke user |
| `source` | `"sql"` atau `"rag"` |
| `job_ids` | ID lowongan terkait (untuk kartu lowongan di UI) |
| `rows` | Data mentah (hasil query / lowongan ditemukan) |
| `sql` | Query yang dieksekusi (SQL agent saja) |
| `usage` | `llm_calls`, `input_tokens`, `output_tokens`, `total_tokens` |
| `error` | `None` jika sukses |

Jumlah panggilan LLM SQL agent: 2 normal, 3 jika retry, 1 jika hasil kosong
atau pertanyaan tidak relevan (misalnya permintaan menghapus data).
