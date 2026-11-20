# Brand bisakerja

**bisakerja** (huruf kecil, satu kata) artinya mampu dan siap bekerja, sekaligus janji "kamu bisa dapat kerja".
Tagline: **Tahu jaraknya, kejar kerjanya.**

Positioning: untuk pencari kerja pertama di Indonesia, bisakerja mencocokkan CV dengan lowongan nyata, menunjukkan seberapa jauh jaraknya lewat skor yang bisa dicek, lalu memberi langkah belajar berikutnya.

Sifat: **jujur pada data** (skor selalu disertai rumusnya), **menyemangati tanpa berlebihan** (tanpa tanda seru beruntun dan emoji), **praktis** (setiap pesan berakhir dengan satu langkah).

## Motif: Rute
Jalur titik-titik dari titik awal (CV) ke target kunyit (kerja). Skill gap adalah sisa jarak di jalur itu. Motif ini dipakai di logomark, banner Beranda, dan ilustrasi empty state.

## Warna

| Token | Light | Dark | Dipakai untuk |
|---|---|---|---|
| primary | `#0D6B6B` | `#127B77` | tombol utama, fokus, elemen aktif |
| link | `#0D6B6B` | `#5CC8C2` | link dan teks info |
| logo-teal | `#127B77` | `#127B77` | logomark, wordmark mono, garis ilustrasi |
| kunyit (aksen) | `#F2A93B` | `#F5B95C` | titik tujuan saja, **tidak untuk teks** |
| text | `#1B2424` | `#EDE8DF` | teks utama |
| muted | `#55605F` | `#B0AA9F` | caption |
| bg / bg2 | `#FBF7EF` / `#F1EADD` | `#141A1A` / `#1E2626` | latar / sidebar dan panel |
| border | `#DDD2C0` | `#34403F` | garis |
| success / warning / error | `#2E7D32` / `#9A5B00` / `#B3261E` | `#7BC67E` / `#F2B45C` / `#FF8A80` | status |

Semua pasangan teks lulus WCAG AA (4.5:1) dan elemen UI lulus 3:1. Tombol primary Streamlit selalu memakai teks putih, karena itu primary dark bernada sedang. Nilai di `.streamlit/config.toml` adalah sumber kebenarannya.

## Font
- **Fraunces** (judul, `wght` 500–800, sumbu SOFT=100, WONK=0, opsz=72), 35 KB.
- **Plus Jakarta Sans** (isi, `wght` 200–800), 33 KB.

Keduanya berlisensi SIL OFL 1.1 dan di-host sendiri di `src/static/fonts/` (berkas lisensi ada di sebelahnya). Tidak ada font yang diambil dari CDN.

## Logo
- `logo-wordmark.svg` untuk latar terang, `logo-wordmark-dark.svg` untuk latar gelap, `logo-wordmark-mono.svg/.png` untuk `st.logo`. Versi mono dipakai karena `st.logo` hanya menerima satu gambar untuk kedua mode, dan warnanya terbaca di keduanya.
- `logo-mark.svg` (+ PNG 32/64/128) untuk favicon, sidebar tertutup, dan avatar asisten.
- **Ruang aman:** di setiap sisi, minimal setinggi titik huruf i. **Ukuran minimum:** wordmark 20 px tinggi, logomark 16 px.
- **Dilarang:** mengubah warna di luar token, memutar, memberi bayangan atau gradien, menulis "Bisa Kerja"/"BisaKerja", atau mengganti titik i dengan bentuk lain.

## Tone of voice
- Sapa dengan "kamu"/"-mu". Asisten tidak menyebut dirinya "aku".
- Satu pesan = keadaan sekarang + satu langkah berikutnya.
- Akui batasan (hasil LLM bisa salah, skor tidak dihitung) tanpa minta maaf berulang.
- Pesan error untuk pengguna ditulis lebih dulu; instruksi teknis di akhir dengan awalan "Developer:".

## Aset (`assets/brand/`)
`logo-wordmark.svg`, `logo-wordmark-dark.svg`, `logo-wordmark-mono.svg`, `logo-wordmark-mono.png`, `logo-mark.svg`, `logo-mark-{32,64,128}.png`, `banner-beranda.svg`, `empty-cv.svg`, `empty-results.svg`, `empty-skillgap.svg`, `empty-chat.svg`. Setiap ilustrasi di bawah 15 KB, hanya memakai warna token, dan memakai latar transparan supaya terbaca di light maupun dark.
