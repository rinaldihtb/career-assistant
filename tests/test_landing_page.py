"""Beranda dengan Streamlit AppTest: banner brand di atas hero, tanpa judul teks ganda."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.components.landing import BANNER_ALT

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"


def test_home_starts_with_brand_banner():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    images = app.get("image")
    assert len(images) == 1
    assert images[0].proto.imgs[0].alt == BANNER_ALT
    headings = [m.value for m in app.markdown if m.value.startswith("#")]
    assert "# Career Assistant" not in headings
    assert not any("💼" in m.value for m in app.markdown)
