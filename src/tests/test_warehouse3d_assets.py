"""Static checks on the 3D warehouse view (static/js/warehouse3d.js).

Two things must never break silently during collection:
  * everything the view loads is served by this app (no third-party CDN), and
  * the view reads only the numbers the participant's panel already shows --
    never the condition, the AI's advice or accuracy, Q*, or delivery timing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STATIC = Path(__file__).resolve().parents[1] / "interface" / "static"
JS_PATH = STATIC / "js" / "warehouse3d.js"
VENDOR = STATIC / "vendor" / "three"

# the panel values the picture may depict (state payload + the week's outcome)
ALLOWED_FIELDS = {
    "inventory", "on_order", "backorders", "transport_ok", "supplier_reliability",   # state
    "arrivals", "inventory_close", "backorders_close",                               # outcome
}
FORBIDDEN_TOKENS = ("condition", "q_ai", "ai_correct", "q_star", "qstar", "risk", "lead_time",
                    "pipeline", "placement", "reliability_level", "disruption")


@pytest.fixture
def client(tmp_path, monkeypatch):
    import interface.db as db
    from interface import app as app_module

    monkeypatch.setattr(db, "DEFAULT_DB", tmp_path / "t.sqlite3")
    app_module._SESSIONS.clear()
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _js() -> str:
    return JS_PATH.read_text(encoding="utf-8")


def _code_only(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _module_imports(src: str) -> list[str]:
    return re.findall(r"""\bfrom\s+['"]([^'"]+)['"]""", src)


def test_every_module_the_view_imports_is_served_locally(client):
    imports = _module_imports(_js())
    assert "three" in imports                          # bare specifier, resolved by task.html's importmap
    local = [p for p in imports if p != "three"]
    assert local, "expected vendored three.js addons"
    for path in local + ["/static/vendor/three/three.module.min.js", "/static/js/warehouse3d.js"]:
        assert path.startswith("/static/"), path
        assert client.get(path).status_code == 200, path


def test_vendored_addons_only_import_three_or_local_files():
    """An addon that imported from a URL would reintroduce a CDN at runtime."""
    for f in VENDOR.glob("*.js"):
        if f.name == "three.module.min.js":
            continue
        for spec in _module_imports(f.read_text(encoding="utf-8")):
            assert spec == "three" or spec.startswith(("./", "/static/")), (f.name, spec)
            if spec.startswith("./"):
                assert (f.parent / spec).exists(), (f.name, spec)
    assert (VENDOR / "LICENSE").exists()                # MIT notice travels with the files


def test_view_has_no_remote_urls():
    js = _js()
    assert "http://" not in js and "https://" not in js
    assert "//cdn" not in js and "unpkg" not in js and "jsdelivr" not in js


def test_view_reads_only_panel_numbers():
    """Hard rule 1: the picture depicts only what the number panel shows."""
    code = _code_only(_js())
    fields = set(re.findall(r"\b(?:s|before|after|outcome|last)\.(\w+)", code))
    assert fields, "state fields not found -- has the API changed?"
    assert fields <= ALLOWED_FIELDS, fields - ALLOWED_FIELDS
    lowered = code.lower()
    for tok in FORBIDDEN_TOKENS:
        assert tok not in lowered, tok


def test_public_api_and_fallback_are_kept():
    js = _js()
    assert "setState: snap" in js and "playWeek" in js
    assert "window.W3D = start(el)" in js
    assert "scene3dCard" in js and "classList.add('hide')" in js      # no-WebGL fallback
    assert "prefers-reduced-motion" in js
    assert "g === gen" in js                                          # a newer state cancels an animation
