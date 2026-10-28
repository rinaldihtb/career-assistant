# Career Assistant

Asisten karier untuk pencari kerja Indonesia berbasis 473 lowongan kerja 2025 (SQLite + Qdrant).
UI Streamlit, API FastAPI, dan supervisor LangGraph yang membagi pertanyaan ke agent SQL, RAG,
dan CV.

| Halaman | Isi | LLM |
|---|---|---|
| Cari Lowongan | Kata kunci dan filter lokasi, tipe kerja, gaji, mode kerja | tidak |
| Cocokkan CV | Upload CV PDF, profil bisa dikoreksi, 10 lowongan paling cocok | 1 panggilan per CV, 1 embedding per pencocokan |
| Skill Gap | Skill yang paling sering diminta lowongan teratas dan belum ada di CV, rencana belajar 4 minggu | 1 panggilan untuk rencana |
| Surat Lamaran | Surat untuk satu lowongan, bisa diedit, unduh PDF atau DOCX | 1 panggilan |
| Asisten Karier | Chat tentang lowongan, dengan atau tanpa CV aktif | planner 1 + per agent; jumlahnya tampil di bawah jawaban |

## Persiapan

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Isi `OPENAI_API_KEY`, `QDRANT_URL`, dan `QDRANT_API_KEY` di `.env`; variabel lain sudah berisi
nilai default. `data/database.sqlite` sudah ada di repo. Koleksi Qdrant cukup diisi sekali per
cluster dengan `make seed-vector` (butuh embedding OpenAI).

## Menjalankan

Lokal, dua terminal:

```bash
make api
```

```bash
make run
```

Streamlit di http://localhost:8501. Halaman Asisten memanggil API di `ASSISTANT_API_URL`
(default `http://localhost:8000`); dokumentasi API di http://localhost:8000/docs.

Docker:

```bash
docker compose up --build -d
docker compose ps
docker compose down
```

Layanan `streamlit` baru dijalankan setelah `fastapi` sehat (healthcheck ke `/docs`). Folder
`data/` di-mount read-only, container berjalan sebagai user `app`, dan secret dibaca dari
`.env`. Untuk memeriksa konfigurasi pakai `docker compose config --quiet`: tanpa `--quiet`,
isi `.env` ikut tercetak.

## Perintah

| Perintah | Fungsi | Butuh API key |
|---|---|---|
| `make run` | Streamlit | ya |
| `make api` | FastAPI untuk halaman Asisten | ya |
| `make test` | Unit test dengan LLM dan vector store palsu | tidak |
| `make lint` | Ruff untuk `src` dan `tests` | tidak |
| `make eval` | Set eval Asisten tanpa CV (`tests/eval/queries.yaml`) | ya |
| `make eval-cv-chat` | Set eval Asisten dengan CV aktif (`tests/eval/cv_chat_cases.yaml`) | ya |
| `make eval-cv` | Set eval Cocokkan CV pada 5 CV fiktif (`tests/eval/cv_cases.yaml`) | ya |
| `make sample-cv ARGS=path/cv.pdf` | Ekstraksi profil dari CV sendiri, hanya angka yang dicetak | ya |
| `make sample-plan` | Metrik rencana belajar Skill Gap | ya |
| `make sample-letter` | Metrik surat lamaran (skill dan angka yang dikarang) | ya |
| `make seed-all` | Bersihkan dataset, isi SQLite dan Qdrant | ya |

Argumen eval lewat `ARGS`, misalnya `make eval ARGS="--label coba --only T3"`. Hasil eval
disimpan di `tests/eval/results/` (tidak di-track). `make eval-cv ARGS="--rescore <file>.json"`
menghitung ulang metrik dari kandidat tersimpan tanpa API.

## Arsitektur

```
Streamlit (src/app.py, src/pages, src/components)
  └─ Asisten ─ POST /chat-assistant ─> FastAPI (src/api.py, src/routes)
                                         └─ SupervisorAgent (src/agents/supervisor_agent_graph.py)
                                              planner -> penjaga rute -> maks 3 worker paralel -> synthesize
                                              worker: rag | sql | cv_match | cv_gap | cv_role
```

| Folder | Isi |
|---|---|
| `src/agents` | `SupervisorAgent`, `RagAgent`, `SqlAgent`, `cv_agent` (tanpa LLM), dan semua prompt di `prompts.py` |
| `src/cv` | PDF ke teks, profil CV, kontak (regex), pencocokan, skill gap, rencana belajar, surat lamaran |
| `src/repositories` | `JobRepository` (SQLite read-only + Qdrant) dan pencarian halaman Cari Lowongan |
| `src/containers.py` | Composition root: `get_job_repository()` dan `get_supervisor_agent()` |
| `src/config` | Model LLM, embedding, dan koneksi database dari `.env` |
| `src/evaluation` | Runner tiga set eval dan pemeriksa jawaban |
| `src/models` | `AgentResult` dan `LLMUsage` (token per panggilan LLM) |
| `scripts` | Pembersihan dataset dan seeding SQLite/Qdrant |

Body `POST /chat-assistant`: `history` (daftar pesan), `user_message`, dan `cv` opsional
`{"profile": CVProfile, "matches": MatchResult}`. Respons: `{"result": AgentResult}`.

Aturan agent:

- SQL hanya satu `SELECT`/`WITH` lewat koneksi read-only; perintah tulis ditolak dan query tanpa
  `LIMIT` diberi `LIMIT`.
- RAG hanya boleh mengutip lowongan kandidat dengan format `[ID:x]`; ID lain dibuang.
- Worker CV tidak memanggil LLM; `cv_role` memakai 1 embedding nama posisi.
- Agent tidak melempar exception: kegagalan menjadi pesan aman di `answer` dan detail di `error`.

## Data CV

Teks CV hanya hidup selama ekstraksi. Profil disimpan di `st.session_state`, tidak ditulis ke
disk, log, atau hasil eval. Nama dan kontak tidak masuk prompt LLM mana pun (dijaga
`tests/test_cv_contacts.py`). Body API Asisten hanya berisi profil ringkas dan hasil
Cocokkan CV, dan tidak di-log.
