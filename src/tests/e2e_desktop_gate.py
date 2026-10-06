"""Browser check of the desktop-only gate on the start page.

Standalone Playwright script (deliberately NOT named test_*.py, so pytest does
not collect it: it needs a running server and a Chromium install).

Usage:
    python src/interface/app.py --port 8795 --db "$TEMP/gate_test.sqlite3" &
    python src/tests/e2e_desktop_gate.py [--base http://127.0.0.1:8795]

Cases:
  1. desktop 1280x900: consent visible, full start reaches /task
  2. phone 400x800 (is_mobile, has_touch): desktop-only notice, consent hidden
  3. narrow desktop 900x800: notice shown; widened to 1280x900 -> consent shown
Exit code 0 only if every case passes.
"""

from __future__ import annotations

import argparse
import sys
import time

from playwright.sync_api import expect, sync_playwright

PHONE_UA = ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0 Mobile Safari/537.36")


def case_desktop_full_start(browser, base: str) -> None:
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    page = ctx.new_page()
    page.goto(f"{base}/?PROLIFIC_PID=GATE{int(time.time())}")
    expect(page.locator("#step1")).to_be_visible()
    expect(page.locator("#desktopOnly")).to_be_hidden()
    page.check("#consent")
    page.click("#next1")
    expect(page.locator("#step2")).to_be_visible()
    for item_id, val in (("num_1", "500"), ("num_2", "10"), ("num_3", "0.1")):
        page.fill(f"#{item_id}", val)
    page.click("#go")
    page.wait_for_url("**/task", timeout=15000)
    ctx.close()


def case_phone(browser, base: str) -> None:
    ctx = browser.new_context(viewport={"width": 400, "height": 800}, is_mobile=True,
                              has_touch=True, user_agent=PHONE_UA)
    page = ctx.new_page()
    page.goto(base + "/")
    expect(page.locator("#desktopOnly")).to_be_visible()
    expect(page.locator("#desktopOnly [role=status]")).to_contain_text("desktop or laptop")
    expect(page.locator("#step1")).to_be_hidden()
    ctx.close()


def case_narrow_then_widen(browser, base: str) -> None:
    ctx = browser.new_context(viewport={"width": 900, "height": 800})
    page = ctx.new_page()
    page.goto(base + "/")
    expect(page.locator("#desktopOnly")).to_be_visible()
    expect(page.locator("#step1")).to_be_hidden()
    page.set_viewport_size({"width": 1280, "height": 900})
    expect(page.locator("#step1")).to_be_visible()
    expect(page.locator("#desktopOnly")).to_be_hidden()
    ctx.close()


CASES = (
    ("desktop 1280x900: consent visible, full start reaches /task", case_desktop_full_start),
    ("phone 400x800 mobile+touch: notice shown, consent hidden", case_phone),
    ("narrow 900x800: notice, then widen to 1280 -> consent shown", case_narrow_then_widen),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8795")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    base = args.base.rstrip("/")
    failures = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        for name, fn in CASES:
            try:
                fn(browser, base)
                print(f"PASS  {name}")
            except Exception as exc:  # report every case, not just the first failure
                failures += 1
                print(f"FAIL  {name}\n      {type(exc).__name__}: {str(exc).splitlines()[0]}")
        browser.close()
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
