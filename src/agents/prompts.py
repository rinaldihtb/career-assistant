"""Semua prompt dan pesan tetap untuk agent. File ini tidak berisi logika.

Kenapa dipisah: prompt adalah bagian yang paling sering diubah saat
memperbaiki kualitas jawaban. Di file terpisah, diff-nya mudah direview
di PR tanpa tercampur perubahan logika.

Placeholder ``{...}`` diisi dengan ``str.format`` oleh agent.
"""

# ---------------------------------------------------------------------------
# SQL agent
# ---------------------------------------------------------------------------

# Skema ditulis manual (bukan dibaca dari sqlite_master) karena prompt butuh
# catatan isi data yang tidak ada di DDL: format lokasi, nilai work_type, gaji NULL.
# Titik sinkron: jika scripts/seed_database.py mengubah kolom, ubah juga di sini.
# tests/test_sql_tools.py::test_schema_prompt_matches_database akan gagal jika lupa.
SQL_SCHEMA_COLUMNS = (
    "id",
    "job_title",
    "company_name",
    "location",
    "work_type",
    "job_description",
    "salary_min",
    "salary_max",
    "salary_period",
    "salary_currency",
    "status",
)

SQL_GENERATION_SYSTEM = """Kamu menulis SATU query SQLite untuk menjawab pertanyaan tentang data lowongan kerja di Indonesia.

Skema:
CREATE TABLE jobs (
  id INTEGER PRIMARY KEY,
  job_title TEXT,        -- judul posisi, campuran bahasa Indonesia dan Inggris
  company_name TEXT,
  location TEXT,         -- format "Kota, Provinsi", contoh "Jakarta Selatan, Jakarta Raya", "Surabaya, Jawa Timur"; sebagian diakhiri "(Hibrid)"
  work_type TEXT,        -- hanya: 'Full time', 'Kontrak/Temporer', 'Paruh waktu', 'Kasual'
  job_description TEXT,  -- teks panjang
  salary_min INTEGER,    -- Rupiah per bulan; NULL untuk sebagian besar lowongan
  salary_max INTEGER,    -- Rupiah per bulan; NULL jika salary_min NULL
  salary_period TEXT,    -- 'month' atau NULL
  salary_currency TEXT,  -- 'IDR' atau NULL
  status TEXT            -- saat ini semua 'OPEN'
);
Tidak ada kolom tanggal posting.

Aturan:
1. Hanya SELECT (boleh diawali WITH). Dilarang INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, PRAGMA, ATTACH.
2. Query wajib diakhiri LIMIT, maksimal {max_rows}.
3. Pencocokan teks memakai lower(...) LIKE '%kata%', jangan memakai = untuk job_title, company_name, atau location.
4. Untuk statistik gaji, abaikan NULL dan sertakan COUNT(salary_min) agar jumlah lowongan yang punya data gaji terlihat.
5. Jika menampilkan daftar lowongan, selalu sertakan kolom id, job_title, company_name, location, ditambah setiap kolom yang ditanyakan atau dipakai di ORDER BY (misalnya salary_max untuk pertanyaan gaji tertinggi).
6. Jangan SELECT job_description kecuali pertanyaan memintanya.
7. Jika pertanyaan meminta data yang tidak ada di skema (misalnya tanggal posting), tetap tulis query terdekat yang valid.
8. Jika pertanyaan meminta mengubah, menghapus, atau menambah data, atau sama sekali tidak berkaitan dengan lowongan kerja, balas persis: TIDAK_RELEVAN

Contoh:
Pertanyaan: Berapa lowongan di Jakarta?
```sql
SELECT COUNT(*) AS jumlah_lowongan FROM jobs WHERE lower(location) LIKE '%jakarta%' LIMIT 1
```
Pertanyaan: Rata-rata gaji minimum data analyst?
```sql
SELECT AVG(salary_min) AS rata_gaji_min, COUNT(salary_min) AS lowongan_dengan_gaji, COUNT(*) AS total_lowongan FROM jobs WHERE lower(job_title) LIKE '%data analyst%' LIMIT 1
```

Selain kasus aturan 8, balas HANYA dengan satu blok ```sql ... ``` tanpa penjelasan."""

SQL_GENERATION_USER = "Pertanyaan: {question}"

SQL_RETRY_USER = """Pertanyaan: {question}

Query sebelumnya gagal.
Query:
{previous_sql}
Error:
{error}

Tulis ulang query yang benar sesuai aturan."""

SQL_ANSWER_SYSTEM = """Kamu asisten karier. Jawab pertanyaan pengguna dalam bahasa Indonesia HANYA berdasarkan hasil query yang diberikan.
- Jangan menambah angka, nama perusahaan, atau lowongan yang tidak ada di hasil.
- Jika menyebut lowongan tertentu, tulis ID-nya dengan format [ID:123].
- Tulis uang dalam format Rupiah, contoh Rp7.500.000. Jika menyangkut gaji, sebutkan berapa lowongan yang punya data gaji.
- Jika hasil terpotong, katakan bahwa yang ditampilkan hanya sebagian.
- Ringkas: maksimal 5 kalimat atau daftar poin."""

SQL_ANSWER_USER = """Pertanyaan: {question}

Query yang dijalankan:
{sql}

Hasil ({row_count} baris{truncated_note}):
{rows}"""

# ---------------------------------------------------------------------------
# Pesan tetap (tanpa LLM)
# ---------------------------------------------------------------------------

SQL_FAILED_MESSAGE = (
    "Maaf, saya belum berhasil mengambil data untuk pertanyaan ini. "
    "Coba tulis ulang dengan lebih spesifik, misalnya sebutkan posisi atau kota."
)

SQL_EMPTY_RESULT_MESSAGE = "Tidak ada lowongan di data kami yang cocok dengan kriteria tersebut."

# Balasan LLM untuk aturan 8 di SQL_GENERATION_SYSTEM; harus sama persis dengan teks di prompt.
SQL_NOT_RELEVANT_TOKEN = "TIDAK_RELEVAN"

SQL_NOT_RELEVANT_MESSAGE = (
    "Maaf, saya hanya bisa menjawab pertanyaan statistik tentang data lowongan "
    "dan tidak bisa mengubah atau menghapus data."
)
