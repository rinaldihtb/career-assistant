"""Token warna, path aset, dan komponen tampilan bersama untuk brand bisakerja.

Nilai warna di sini sama dengan ``.streamlit/config.toml``; test memeriksa keduanya tetap sama.
Panduan pemakaian ada di ``docs/brand.md``.
"""

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRAND_DIR = PROJECT_ROOT / "assets" / "brand"
FONT_DIR = PROJECT_ROOT / "src" / "static" / "fonts"

# st.logo memakai satu gambar untuk mode light dan dark, jadi wordmark-nya versi satu warna.
LOGO_WORDMARK = BRAND_DIR / "logo-wordmark-mono.png"
LOGO_MARK = BRAND_DIR / "logo-mark-128.png"
FAVICON = BRAND_DIR / "logo-mark-64.png"
BANNER = BRAND_DIR / "banner-beranda.svg"
ILLUSTRATION_NO_CV = BRAND_DIR / "empty-cv.svg"
ILLUSTRATION_NO_RESULTS = BRAND_DIR / "empty-results.svg"
ILLUSTRATION_SKILL_GAP = BRAND_DIR / "empty-skillgap.svg"
ILLUSTRATION_CHAT = BRAND_DIR / "empty-chat.svg"

# Kunci sama dengan [theme.light] dan [theme.dark] di config.toml.
LIGHT = {
    "primaryColor": "#0D6B6B",
    "backgroundColor": "#FBF7EF",
    "secondaryBackgroundColor": "#F1EADD",
    "textColor": "#1B2424",
    "borderColor": "#DDD2C0",
    "linkColor": "#0D6B6B",
    "blueColor": "#0D6B6B",
    "blueTextColor": "#0D6B6B",
    "greenColor": "#2E7D32",
    "orangeColor": "#9A5B00",
    "redColor": "#B3261E",
}
DARK = {
    "primaryColor": "#127B77",
    "backgroundColor": "#141A1A",
    "secondaryBackgroundColor": "#1E2626",
    "textColor": "#EDE8DF",
    "borderColor": "#34403F",
    "linkColor": "#5CC8C2",
    "blueColor": "#127B77",
    "blueTextColor": "#5CC8C2",
    "greenColor": "#7BC67E",
    "orangeColor": "#F2B45C",
    "redColor": "#FF8A80",
}
# Warna di luar tema Streamlit: teal logo (terbaca di light dan dark) dan aksen kunyit.
# Kunyit kontrasnya rendah di latar terang, jadi hanya untuk dekorasi, tidak untuk teks.
LOGO_TEAL = "#127B77"
KUNYIT = "#F2A93B"

# Ambang WCAG 2.x: teks normal 4.5:1, elemen UI dan teks besar 3:1.
TEXT_CONTRAST = 4.5
UI_CONTRAST = 3.0


def _channel(value: int) -> float:
    """Ubah satu kanal sRGB 0-255 menjadi nilai linear untuk rumus luminans WCAG."""
    c = value / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """Luminans relatif WCAG dari warna ``#RRGGBB`` (0 = hitam, 1 = putih)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(foreground: str, background: str) -> float:
    """Rasio kontras WCAG antara dua warna ``#RRGGBB``, dari 1.0 sampai 21.0."""
    light, dark = sorted(
        (relative_luminance(foreground), relative_luminance(background)), reverse=True
    )
    return (light + 0.05) / (dark + 0.05)


@dataclass(frozen=True)
class EmptyStateAction:
    """Link satu langkah berikutnya di empty state: halaman tujuan dan labelnya."""

    label: str
    page: str
    icon: str | None = None


def empty_state(
    illustration: Path,
    title: str,
    body: str,
    action: EmptyStateAction | None = None,
) -> None:
    """Tampilkan empty state: ilustrasi, judul, satu paragraf, dan link opsional.

    Ilustrasi berlatar transparan sehingga terbaca di mode light dan dark tanpa CSS tambahan.
    """
    with st.container(border=True):
        st.image(str(illustration), width=200)
        st.markdown(f"#### {title}")
        st.markdown(body)
        if action is not None:
            st.page_link(action.page, label=action.label, icon=action.icon)
