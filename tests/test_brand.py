"""Aset dan token brand bisakerja: file ada, tema sama dengan config.toml, kontras WCAG."""

import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.components import brand

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".streamlit" / "config.toml"
APP_PATH = ROOT / "src" / "app.py"
ASSETS = [
    brand.LOGO_WORDMARK,
    brand.LOGO_MARK,
    brand.FAVICON,
    brand.BANNER,
    brand.ILLUSTRATION_NO_CV,
    brand.ILLUSTRATION_NO_RESULTS,
    brand.ILLUSTRATION_SKILL_GAP,
    brand.ILLUSTRATION_CHAT,
]
ILLUSTRATIONS = [path for path in ASSETS if path.suffix == ".svg" and path != brand.BANNER]


def _config() -> dict:
    with CONFIG.open("rb") as file:
        return tomllib.load(file)


@pytest.mark.parametrize("path", ASSETS, ids=lambda path: path.name)
def test_brand_asset_exists(path):
    assert path.is_file()


@pytest.mark.parametrize("path", ILLUSTRATIONS, ids=lambda path: path.name)
def test_illustration_is_valid_small_svg(path):
    ET.parse(path)
    assert path.stat().st_size < 15_000


def test_banner_is_valid_svg_within_budget():
    # Banner memuat wordmark dan teks sebagai path; batasnya lebih longgar agar huruf tetap tajam.
    ET.parse(brand.BANNER)
    assert brand.BANNER.stat().st_size < 25_000


@pytest.mark.parametrize(("mode", "tokens"), [("light", brand.LIGHT), ("dark", brand.DARK)])
def test_theme_tokens_match_config(mode, tokens):
    assert _config()["theme"][mode] == tokens


def test_font_faces_point_to_bundled_fonts():
    config = _config()
    assert config["server"]["enableStaticServing"] is True
    for face in config["theme"]["fontFaces"]:
        assert face["url"].startswith("app/static/fonts/")
        assert (brand.FONT_DIR / Path(face["url"]).name).is_file()


def test_contrast_ratio_known_values():
    assert brand.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert brand.contrast_ratio("#FFFFFF", "#FFFFFF") == pytest.approx(1.0)
    assert brand.contrast_ratio("#0D6B6B", "#FBF7EF") == pytest.approx(
        brand.contrast_ratio("#FBF7EF", "#0D6B6B")
    )


@pytest.mark.parametrize(("mode", "tokens"), [("light", brand.LIGHT), ("dark", brand.DARK)])
def test_theme_meets_wcag(mode, tokens):
    bg, panel = tokens["backgroundColor"], tokens["secondaryBackgroundColor"]
    for background in (bg, panel):
        assert brand.contrast_ratio(tokens["textColor"], background) >= brand.TEXT_CONTRAST
    assert brand.contrast_ratio(tokens["linkColor"], bg) >= brand.TEXT_CONTRAST
    # Tombol primary Streamlit selalu bertuliskan putih.
    assert brand.contrast_ratio("#FFFFFF", tokens["primaryColor"]) >= brand.TEXT_CONTRAST
    assert brand.contrast_ratio(tokens["primaryColor"], bg) >= brand.UI_CONTRAST
    for status in ("greenColor", "orangeColor", "redColor"):
        assert brand.contrast_ratio(tokens[status], bg) >= brand.TEXT_CONTRAST


def test_logo_teal_readable_in_both_sidebars():
    for tokens in (brand.LIGHT, brand.DARK):
        sidebar = tokens["secondaryBackgroundColor"]
        assert brand.contrast_ratio(brand.LOGO_TEAL, sidebar) >= brand.UI_CONTRAST


def _empty_state_app(with_action: bool) -> AppTest:
    def script(with_action):
        from src.components.brand import ILLUSTRATION_NO_CV, EmptyStateAction, empty_state

        link = EmptyStateAction("Buka dokumentasi", "https://docs.streamlit.io")
        empty_state(
            ILLUSTRATION_NO_CV, "Belum ada CV", "Upload CV dulu.", link if with_action else None
        )

    app = AppTest.from_function(script, kwargs={"with_action": with_action})
    return app.run()


def test_empty_state_renders_illustration_title_and_body():
    app = _empty_state_app(with_action=False)

    assert not app.exception
    assert len(app.get("image")) == 1
    assert [m.value for m in app.markdown] == ["#### Belum ada CV", "Upload CV dulu."]
    assert app.get("page_link") == []


def test_empty_state_action_renders_one_link():
    app = _empty_state_app(with_action=True)

    assert not app.exception
    assert [link.proto.label for link in app.get("page_link")] == ["Buka dokumentasi"]


def test_sidebar_links_use_material_icons():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    icons = [link.proto.icon for link in app.sidebar.get("page_link")]
    assert len(icons) == 5
    assert all(icon.startswith(":material/") for icon in icons)
