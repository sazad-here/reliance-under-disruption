"""
Open a real browser window on the Kaggle API settings page and wait for the token.

You log in yourself -- your password never passes through me. This script only
opens the window, puts you on the right page, and catches the kaggle.json file
when you click "Create New Token", installing it to ~/.kaggle/ automatically.

The browser profile is kept in data/.browser so you stay logged in next time.

Run:  python src/open_kaggle.py
Close the window (or press Ctrl+C) when done.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(HERE, "..", "data", ".browser")
DOWNLOADS = os.path.join(HERE, "..", "data", ".downloads")
KAGGLE_DIR = os.path.join(os.path.expanduser("~"), ".kaggle")
KAGGLE_JSON = os.path.join(KAGGLE_DIR, "kaggle.json")

SETTINGS_URL = "https://www.kaggle.com/settings"

BANNER = """
======================================================================
  KAGGLE API TOKEN
======================================================================
  A Chromium window is opening on the Kaggle settings page.

  1. Sign in if you are not already (your password stays with you --
     this script never reads it).
  2. Scroll down to the section called  API
  3. Click  "Create New Token"
  4. kaggle.json downloads -- this script catches it and installs it.

  Leave this running. It will tell you the moment the token lands.
  Press Ctrl+C or close the window when you are done.
======================================================================
"""


def install_token(src_path: str) -> bool:
    os.makedirs(KAGGLE_DIR, exist_ok=True)
    shutil.copyfile(src_path, KAGGLE_JSON)
    try:
        os.chmod(KAGGLE_JSON, 0o600)
    except OSError:
        pass  # Windows ignores POSIX modes
    try:
        with open(KAGGLE_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        print("  [!] downloaded file is not valid JSON: %s" % exc)
        return False
    if "username" not in data or "key" not in data:
        print("  [!] file is missing 'username' or 'key' -- not a Kaggle token?")
        return False
    print("\n  [OK] token installed for user: %s" % data["username"])
    print("       -> %s" % KAGGLE_JSON)
    return True


def main():
    if os.path.exists(KAGGLE_JSON):
        print("  A token already exists at %s" % KAGGLE_JSON)
        print("  Opening anyway so you can replace it if you want.\n")

    os.makedirs(PROFILE, exist_ok=True)
    os.makedirs(DOWNLOADS, exist_ok=True)
    print(BANNER)

    got = {"done": False}

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=PROFILE,
            headless=False,                 # a real window, on your screen
            accept_downloads=True,
            downloads_path=DOWNLOADS,
            viewport={"width": 1280, "height": 900},
            args=["--start-maximized"],
        )

        def on_download(download):
            name = download.suggested_filename
            dest = os.path.join(DOWNLOADS, name)
            download.save_as(dest)
            print("\n  downloaded: %s" % name)
            if name.endswith(".json"):
                if install_token(dest):
                    got["done"] = True
                    print("\n  You can close the browser window now.")
                    print("  Then tell me, and I will pull Olist, M5 and DataCo.\n")
            else:
                print("  (not a .json -- ignoring)")

        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.on("download", on_download)
        ctx.on("page", lambda p: p.on("download", on_download))

        try:
            page.goto(SETTINGS_URL, wait_until="domcontentloaded", timeout=60_000)
            print("  window open at: %s" % SETTINGS_URL)
        except Exception as exc:
            print("  [!] could not load the page: %s" % exc)
            print("      Navigate to %s manually in the window." % SETTINGS_URL)

        # Keep the script (and therefore the window) alive.
        try:
            while True:
                if not ctx.pages:
                    print("\n  Window closed.")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Stopped.")
        finally:
            try:
                ctx.close()
            except Exception:
                pass

    if got["done"]:
        print("  RESULT: token installed. Ready to download datasets.")
        return 0
    if os.path.exists(KAGGLE_JSON):
        print("  RESULT: a token is present at %s" % KAGGLE_JSON)
        return 0
    print("  RESULT: no token installed yet.")
    print("  Re-run:  python src/open_kaggle.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
