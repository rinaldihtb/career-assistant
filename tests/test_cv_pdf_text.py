"""Ekstraksi teks CV dari PDF: jalur normal dan setiap alasan penolakan.

PDF dibuat langsung di test dengan pypdf (font standar Helvetica), jadi tidak perlu
file contoh di repo maupun library tambahan. Halaman tanpa teks mewakili PDF hasil
scan: keduanya sama-sama tidak punya lapisan teks.
"""

import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from cv import pdf_text
from cv.pdf_text import CVRejectedError, extract_cv_text

CV_PAGE = (
    "Dimas Prakoso - Data Analyst\n"
    "Surabaya, Jawa Timur\n"
    "Pengalaman: Data Analyst Intern di PT Ritel Nusantara, 2025.\n"
    "Membersihkan data transaksi dengan Python dan Pandas.\n"
    "Keahlian: Python, SQL, Pandas, Microsoft Excel, Power BI"
)


def make_pdf(
    pages: list[str], user_password: str | None = None, owner_password: str | None = None
) -> bytes:
    """Buat PDF dengan satu halaman per elemen ``pages``; string kosong = halaman tanpa teks."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in pages:
        page = writer.add_blank_page(width=595, height=842)
        if not text:
            continue
        escaped = [
            line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            for line in text.split("\n")
        ]
        ops = [
            "BT",
            "/F1 11 Tf",
            "14 TL",
            "50 800 Td",
            *(f"({line}) Tj T*" for line in escaped),
            "ET",
        ]
        stream = DecodedStreamObject()
        stream.set_data("\n".join(ops).encode("latin-1"))
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        page.replace_contents(stream)
    if user_password is not None or owner_password is not None:
        writer.encrypt(
            user_password=user_password or "", owner_password=owner_password, algorithm="RC4-128"
        )
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def rejected_reason(data: bytes) -> str:
    with pytest.raises(CVRejectedError) as error:
        extract_cv_text(data)
    assert str(error.value)  # selalu ada pesan untuk pengguna
    return error.value.reason


def test_extracts_text_from_every_page():
    result = extract_cv_text(
        make_pdf([CV_PAGE, "Pendidikan\nS1 Statistika, Universitas Contoh, 2020-2024. " * 3])
    )

    assert result.pages == 2
    assert result.truncated is False
    assert "Power BI" in result.text
    assert "S1 Statistika" in result.text
    assert "\n\n" in result.text  # pemisah antarhalaman


def test_collapses_extra_spaces_but_keeps_lines():
    result = extract_cv_text(make_pdf([CV_PAGE.replace("Python, SQL", "Python,    SQL")]))

    assert "Python, SQL" in result.text
    assert "Surabaya, Jawa Timur\n" in result.text


def test_owner_only_password_is_still_readable():
    result = extract_cv_text(make_pdf([CV_PAGE], owner_password="kunci-edit"))

    assert "Microsoft Excel" in result.text


def test_truncates_unusually_long_text(monkeypatch):
    monkeypatch.setattr(pdf_text, "MAX_TEXT_CHARS", 50)

    result = extract_cv_text(make_pdf([CV_PAGE]))

    assert result.truncated is True
    assert len(result.text) == 50


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (b"", "empty"),
        (b"PK\x03\x04 bukan pdf, misalnya .docx", "not_pdf"),
        (b"%PDF-1.7\n" + b"0" * (5 * 1024 * 1024), "too_large"),
        (b"%PDF-1.7\nini bukan isi pdf yang valid", "unreadable"),
    ],
    ids=["empty", "not-pdf", "too-large", "corrupt"],
)
def test_rejects_invalid_files(data, reason):
    assert rejected_reason(data) == reason


def test_rejects_scanned_pdf_without_text_layer():
    assert rejected_reason(make_pdf([""])) == "no_text"


def test_rejects_pdf_with_too_little_text_per_page():
    sparse = make_pdf(["Dimas Prakoso", "", ""])  # 12 karakter untuk 3 halaman

    assert rejected_reason(sparse) == "no_text"


def test_rejects_user_password():
    assert rejected_reason(make_pdf([CV_PAGE], user_password="rahasia")) == "encrypted"


def test_rejects_more_than_max_pages():
    assert rejected_reason(make_pdf([CV_PAGE] * (pdf_text.MAX_PAGES + 1))) == "too_many_pages"


def test_accepts_exactly_max_pages():
    assert extract_cv_text(make_pdf([CV_PAGE] * pdf_text.MAX_PAGES)).pages == pdf_text.MAX_PAGES
