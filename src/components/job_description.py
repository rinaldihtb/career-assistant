"""Merapikan deskripsi lowongan mentah menjadi bagian baku untuk ditampilkan.

Deskripsi di database adalah teks hasil scraping dari ratusan perusahaan:
indentasi ``\\xa0``, karakter tak terlihat, bullet yang berdiri sendiri di satu
baris, kalimat terputus di tengah, dan lebih dari 60 variasi judul untuk makna
yang sama ("Kualifikasi", "Requirements", "What we're looking for", "Soft Skills").

Modul ini mengubahnya menjadi lima bagian baku dengan urutan tetap::

    Ringkasan -> Tanggung Jawab -> Kualifikasi -> Benefit -> Info Lainnya

Deskripsi tanpa satu pun judul menjadi satu bagian "Deskripsi". Semua aturan
deterministik (tanpa LLM) dan tidak pernah membuang kata dari teks asli, kecuali
penomoran lama yang diganti penomoran baru. Data di SQLite dan Qdrant tidak diubah.

Tidak bergantung pada Streamlit, sehingga bisa dites langsung dan dipakai halaman lain.
"""

import re
from dataclasses import dataclass, field

SECTION_TITLES = {
    "deskripsi": "Deskripsi",
    "ringkasan": "Ringkasan",
    "tanggung_jawab": "Tanggung Jawab",
    "kualifikasi": "Kualifikasi",
    "benefit": "Benefit",
    "info": "Info Lainnya",
}
SECTION_ORDER = ("ringkasan", "tanggung_jawab", "kualifikasi", "benefit", "info")
# Bagian yang isinya hampir selalu daftar: selalu ditampilkan bernomor.
LIST_SECTIONS = frozenset({"tanggung_jawab", "kualifikasi", "benefit"})

# Aturan judul, dicek berurutan: "What you will get" harus menjadi benefit sebelum
# dicek "what you". PHRASES dicocokkan di bagian mana pun dalam judul ("Main Job
# Description"); PREFIXES hanya sebagai kata utuh di awal karena kata-katanya umum
# ("About", "Tugas", dan "note" yang tidak boleh cocok dengan "Notebook").
_HEADER_RULES = (
    ("benefit",
     ("what we offer", "we offer", "what you will get", "what you'll get", "why join",
      "why you'll love", "we have your back", "yang kami tawarkan", "penawaran kami",
      "mengapa bergabung", "kenapa bergabung", "why work", "bergabung dengan kami",
      "bekerja bersama kami", "bekerja dengan kami"),
     ("benefit", "benefits", "perks", "fasilitas", "keuntungan", "kompensasi", "compensation",
      "tunjangan", "penawaran")),
    ("kualifikasi",
     ("kualifikasi", "qualification", "requirement", "persyaratan", "looking for", "who you are",
      "what you'll need", "what you need", "what you bring", "soft skill", "hard skill",
      "technical skill", "must-have", "must have", "nice to have", "nice-to-have",
      "keterampilan", "kriteria"),
     ("syarat", "skills", "competenc", "kompetensi")),
    ("tanggung_jawab",
     ("job description", "job desc", "job deskripsi", "deskripsi pekerjaan", "deskripsi kerja",
      "responsibilit", "tanggung jawab", "what you'll do", "what you will do", "you'll be doing",
      "you will be doing", "deliverables", "uraian pekerjaan", "job scope", "scope of work",
      "rincian pekerjaan"),
     ("tugas", "tasks", "your role", "the role", "duties")),
    ("ringkasan",
     ("about the", "about us", "job summary", "role overview", "job overview", "gambaran umum",
      "company description", "company overview", "get to know", "who we are",
      "profil perusahaan", "deskripsi perusahaan", "tentang perusahaan"),
     ("about", "tentang", "ringkasan", "summary", "overview", "life at", "description")),
    ("info",
     ("employment type", "work arrangement", "working hours", "jam kerja", "sistem kerja",
      "additional information", "informasi tambahan", "recruitment detail", "how to apply",
      "cara melamar"),
     ("location", "lokasi", "penempatan", "salary", "catatan", "note")),
)  # fmt: skip
# Judul satu kata yang hanya diakui jika berdiri sendiri: "Experience in SQL" adalah item.
_HEADER_EXACT = {
    "experience": "kualifikasi",
    "education": "kualifikasi",
    "pendidikan": "kualifikasi",
    "pengalaman": "kualifikasi",
    "skill": "kualifikasi",
    "kemampuan": "kualifikasi",
}
MAX_HEADER_WORDS = 5
MAX_PREFIX_HEADER_WORDS = 3
MAX_HEADER_CHARS = 60
# Kalimat pengantar daftar yang diakhiri ":" boleh lebih panjang:
# "We are looking for a candidate who has:".
MAX_INTRO_HEADER_CHARS = 100
# Judul sepanjang ini biasanya memuat informasi ("Tentang PT X", "As a Finance Analyst,
# your main responsibilities will include"), jadi tetap ditampilkan sebagai pengantar.
INTRO_MIN_WORDS = 5
PARAGRAPH_MIN_CHARS = 160
MAX_LABEL_CHARS = 80

_INVISIBLE = dict.fromkeys(map(ord, "\u2060\u200b\u200e\ufeff"), None)
# Penanda daftar: simbol bullet umum atau penomoran "1." "2)" "a." "B)".
_LIST_PREFIX = re.compile(r"^(?:[•·●○▪▫◦■□➢➤►▶✓✔✗✘*>»\-–—]+|(?:\d{1,2}|[a-zA-Z])[.)](?=\s))\s*")
# Simbol dekoratif lain di awal baris (emoji) dibuang tanpa menjadikannya bullet.
# Tanda baca penutup tidak termasuk: baris ", ensuring ..." adalah lanjutan kalimat.
_DECOR_PREFIX = re.compile(r"^[^\w\s(\[\"'“‘$€£@#,.;:!?%)\]}]+\s*")
_KEY_VALUE_INFO = re.compile(
    r"^(location|lokasi|penempatan|employment type|tipe pekerjaan|status kerja|salary|gaji|"
    r"jam kerja|working hours|work arrangement|sistem kerja)\s*:\s*\S",
    re.IGNORECASE,
)
# Baris yang berakhir dengan ini hampir pasti kalimat yang terputus ke baris berikutnya:
# "Proficient in" + "Python", "Familiarity with" + "SQL".
_CONNECTORS = (
    " in", " with", " of", " and", " or", " for", " to", " the", " a", " an", " at", " on",
    " as", " dan", " atau", " di", " untuk", " dengan", " yang", " seperti", " pada", " dari",
    " ke", " serta", " sebagai", " bagi", " minimal", " minimum", " min.", "&", ",", "(", "[",
    "/", "-",
)  # fmt: skip
_CLOSING = ")]},.;:"
_SENTENCE_BREAK = re.compile(r"[.!?]\s+[A-Z]")
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]#|~$<>])")


@dataclass
class Section:
    """Satu bagian baku. ``items`` berisi ``(jenis, teks)``.

    Jenis: ``"bullet"`` (sumber memakai penanda daftar), ``"text"``, atau
    ``"intro"`` (judul asli yang panjang, ditampilkan sebagai pengantar).
    """

    key: str
    items: list[tuple[str, str]] = field(default_factory=list)

    @property
    def title(self) -> str:
        return SECTION_TITLES[self.key]


# --- API utama ---------------------------------------------------------------


def structure_description(text: str | None) -> list[Section]:
    """Kelompokkan deskripsi ke bagian baku, urut ``SECTION_ORDER``.

    Teks sebelum judul pertama masuk Ringkasan. Bagian yang muncul berkali-kali
    (misalnya "Technical Skills" dan "Soft Skills") digabung ke satu bagian.
    """
    lines = clean_lines(text or "")
    if not any(kind.startswith("header:") for kind, _ in lines):
        items = [("bullet" if kind == "info" else kind, line) for kind, line in lines]
        return [Section("deskripsi", items)] if items else []

    sections = {key: Section(key) for key in SECTION_ORDER}
    current = "ringkasan"
    for kind, line in lines:
        if kind.startswith("header:"):
            current = kind.split(":", 1)[1]
            if len(line.split()) >= INTRO_MIN_WORDS:
                sections[current].items.append(("intro", line))
        elif kind == "info":
            sections["info"].items.append(("bullet", line))
        else:
            sections[current].items.append((kind, line))
    return [section for section in sections.values() if section.items]


def description_to_markdown(text: str | None) -> str:
    """Deskripsi siap tampil sebagai Markdown.

    - Judul bagian: tebal, nama baku.
    - Tanggung Jawab, Kualifikasi, Benefit: daftar bernomor.
    - Ringkasan, Info Lainnya, Deskripsi: paragraf untuk kalimat panjang,
      bullet untuk deretan pernyataan pendek.
    - Sub-judul di dalam daftar (baris diakhiri ":" atau baris tanpa bullet yang
      diikuti item ber-bullet) dicetak miring dan penomoran dimulai ulang.
    """
    blocks: list[str] = []
    for section in structure_description(text):
        blocks.append(f"**{section.title}**")
        blocks.extend(_section_blocks(section))
    return "\n\n".join(blocks)


def escape_markdown(text: str) -> str:
    """Escape karakter Markdown agar teks lowongan tampil apa adanya.

    ``$`` ikut di-escape karena Streamlit membaca ``$...$`` sebagai rumus LaTeX.
    """
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


# --- Pembersihan baris -------------------------------------------------------


def clean_lines(text: str) -> list[tuple[str, str]]:
    """Ubah teks mentah menjadi baris bersih ``(jenis, teks)``.

    Jenis: ``"header:<kunci>"``, ``"bullet"``, ``"text"``, atau ``"info"`` (baris
    kunci-nilai seperti "Location: Jakarta"). Aturan:
        1. Spasi aneh diseragamkan, karakter tak terlihat dan emoji di awal dibuang.
        2. Bullet/penomoran lama dibuang dan dicatat sebagai ``"bullet"``; bullet
           yang berdiri sendiri digabung dengan baris berikutnya.
        3. Potongan kalimat disambung: baris berhuruf kecil, baris diawali tanda
           baca penutup, atau baris setelah kata sambung ("Proficient in").
        4. Judul dikenali dari kata kuncinya, kecuali baris itu potongan kalimat.
    """
    raw_lines = [" ".join(raw.split()) for raw in text.translate(_INVISIBLE).splitlines()]
    raw_lines = [line for line in raw_lines if line]
    items: list[tuple[str, str]] = []
    pending_bullet = False
    for index, line in enumerate(raw_lines):
        next_line = raw_lines[index + 1] if index + 1 < len(raw_lines) else None
        if not any(char.isalnum() for char in line):
            # Tanda baca yang terpisah (")." atau "&") adalah bagian kalimat sebelumnya;
            # simbol lain ("•" atau emoji yang berdiri sendiri) menandai item berikutnya.
            joinable = line[0] in _CLOSING or line in ("&", "/", "+")
            if joinable and items and not items[-1][0].startswith("header:"):
                items[-1] = _merge(items[-1], line)
            else:
                pending_bullet = True
            continue

        body, is_bullet = _strip_prefix(line)
        if not body:
            pending_bullet = True
            continue
        # "Hybrid" lalu "– kombinasi kerja remote": tanda pisah sebagai penjelas, bukan bullet.
        if line[0] in "–—" and body[0].islower() and items and items[-1][0] in ("text", "bullet"):
            items[-1] = (items[-1][0], f"{items[-1][1]} {line}")
            pending_bullet = False
            continue

        key = None if _is_fragment(body, next_line) else header_key(body)
        if key is not None:
            items.append((f"header:{key}", body.rstrip(" :")))
        elif _KEY_VALUE_INFO.match(body):
            items.append(("info", body))
        elif is_bullet or pending_bullet:
            items.append(("bullet", body))
        elif items and not items[-1][0].startswith("header:") and _continues(items[-1][1], body):
            items[-1] = _merge(items[-1], body)
        else:
            items.append(("text", body))
        pending_bullet = False
    return items


def header_key(line: str) -> str | None:
    """Kunci bagian baku jika ``line`` adalah judul bagian, selain itu ``None``."""
    limit = MAX_INTRO_HEADER_CHARS if line.rstrip().endswith(":") else MAX_HEADER_CHARS
    if len(line) > limit:
        return None
    norm = line.lower().replace("’", "'").replace("(s)", "s").strip().rstrip(" :?.").strip()
    if not norm:
        return None
    if norm in _HEADER_EXACT:
        return _HEADER_EXACT[norm]
    if len(norm.split()) > MAX_HEADER_WORDS and not line.rstrip().endswith((":", "?")):
        return None
    # Kata kunci umum hanya judul jika barisnya pendek atau diakhiri ":":
    # "Fasilitas:" adalah judul, "Tunjangan Kesehatan & BPJS" adalah item.
    short = len(norm.split()) <= MAX_PREFIX_HEADER_WORDS or line.rstrip().endswith(":")
    for key, phrases, prefixes in _HEADER_RULES:
        if any(phrase in norm for phrase in phrases):
            return key
        if short and any(norm == prefix or norm.startswith(f"{prefix} ") for prefix in prefixes):
            return key
    return None


def _strip_prefix(line: str) -> tuple[str, bool]:
    """Buang penanda daftar (dicatat sebagai bullet) atau simbol dekoratif (tidak)."""
    prefix = _LIST_PREFIX.match(line)
    if prefix:
        return line[prefix.end() :].strip(), True
    decor = _DECOR_PREFIX.match(line)
    return (line[decor.end() :].strip() if decor else line), False


def _is_fragment(line: str, next_line: str | None) -> bool:
    """True jika baris adalah potongan kalimat, sehingga tidak boleh dianggap judul."""
    if line[0] in _CLOSING or line.lower().endswith(_CONNECTORS):
        return True
    # "We offer very" + "competitive salary" adalah kalimat yang berlanjut.
    # Judul pendek ("Kualifikasi") tetap boleh diikuti item berhuruf kecil.
    return (
        next_line is not None
        and next_line[0].islower()
        and not line.endswith(":")
        and len(line.split()) >= 3
    )


def _continues(previous: str, line: str) -> bool:
    """True jika ``line`` adalah lanjutan kalimat ``previous`` yang terputus."""
    return line[0].islower() or line[0] in _CLOSING or previous.lower().endswith(_CONNECTORS)


def _merge(previous: tuple[str, str], line: str) -> tuple[str, str]:
    """Sambung potongan kalimat; kalimat pengantar yang jadi utuh bisa menjadi judul.

    Contoh: "We are looking for a" + "candidate who has:" menjadi judul Kualifikasi.
    """
    kind, text = previous
    if text.endswith(("(", "[", "/")) or line[0] in _CLOSING:
        merged = f"{text}{line}"
    else:
        merged = f"{text} {line}"
    if kind == "text" and merged.endswith(":") and (key := header_key(merged)):
        return f"header:{key}", merged.rstrip(" :")
    return kind, merged


# --- Render Markdown ---------------------------------------------------------


def _section_blocks(section: Section) -> list[str]:
    """Blok Markdown untuk isi satu bagian (tanpa judulnya)."""
    blocks: list[str] = []
    run: list[str] = []
    number = 0
    numbered = section.key in LIST_SECTIONS
    has_bullets = any(kind == "bullet" for kind, _ in section.items)

    def flush() -> None:
        if run:
            blocks.append("\n".join(run))
            run.clear()

    for index, (kind, line) in enumerate(section.items):
        next_kind = section.items[index + 1][0] if index + 1 < len(section.items) else None
        escaped = escape_markdown(line)
        if kind == "intro" or _is_label(kind, line, next_kind, has_bullets):
            flush()
            blocks.append(f"*{escape_markdown(line.rstrip(' :'))}*")
            number = 0
        elif _is_paragraph(kind, line, numbered) or (
            not numbered and _is_lone_line(section, index)
        ):
            flush()
            blocks.append(escaped)
            number = 0
        elif numbered:
            number += 1
            run.append(f"{number}. {escaped}")
        else:
            run.append(f"- {escaped}")
    flush()
    return blocks


def _is_label(kind: str, line: str, next_kind: str | None, has_bullets: bool) -> bool:
    """Sub-judul: diakhiri ":" atau baris tanpa bullet yang langsung diikuti item ber-bullet.

    Contoh sumber: "🔹 Recruitment & Talent Acquisition" lalu "• Menentukan kebutuhan SDM".
    """
    if kind != "text" or len(line) > MAX_LABEL_CHARS:
        return False
    return line.endswith(":") or (has_bullets and next_kind == "bullet")


def _is_paragraph(kind: str, line: str, in_list: bool) -> bool:
    """True jika teks lebih rapi ditampilkan sebagai paragraf daripada item daftar.

    Di bagian daftar hanya teks panjang berisi lebih dari satu kalimat (pengantar
    seperti "We are seeking ... This role involves ..."), agar penomoran tidak
    terpecah. Di bagian lain, setiap teks berisi kalimat lengkap atau teks panjang.
    """
    if kind != "text":
        return False
    multi_sentence = bool(_SENTENCE_BREAK.search(line))
    if in_list:
        return multi_sentence and len(line) >= PARAGRAPH_MIN_CHARS
    return multi_sentence or len(line) >= PARAGRAPH_MIN_CHARS


def _is_lone_line(section: Section, index: int) -> bool:
    """Baris pendek tanpa tetangga pendek (di bagian non-daftar) lebih rapi sebagai paragraf.

    Satu bullet tunggal di antara paragraf terlihat janggal; bullet baru dipakai
    untuk deretan minimal dua pernyataan pendek.
    """

    def short(position: int) -> bool:
        if not 0 <= position < len(section.items):
            return False
        kind, line = section.items[position]
        return kind in ("text", "bullet") and not _is_paragraph(kind, line, in_list=False)

    return not short(index - 1) and not short(index + 1)
