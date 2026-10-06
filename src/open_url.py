"""
Open a URL in a real Chromium window, reusing the saved Kaggle login.

Usage:  python src/open_url.py <url>
"""
from __future__ import annotations

import os
import sys
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(HERE, "..", "data", ".browser")

DEFAULT_URL = "https://www.kaggle.com/c/m5-forecasting-accuracy/rules"


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    os.makedirs(PROFILE, exist_ok=True)

    print("=" * 66)
    print("  Opening: %s" % url)
    print("=" * 66)
    if "rules" in url:
        print("""
  Scroll to the bottom of the rules and click:

      [ I Understand and Accept ]

  You must be signed in for the button to appear. The profile from
  the earlier window is reused, so you are probably already in.

  Leave this running; close the window when done.
""")

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=PROFILE,
            headless=False,
            viewport={"width": 1280, "height": 900},
            args=["--start-maximized"],
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            print("  window is open", flush=True)
        except Exception as exc:
            print("  could not load: %s" % exc, flush=True)

        try:
            while ctx.pages:
                time.sleep(1)
            print("\n  window closed", flush=True)
        except KeyboardInterrupt:
            print("\n  stopped", flush=True)
        finally:
            try:
                ctx.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
