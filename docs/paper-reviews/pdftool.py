"""Read a PDF without poppler: page text, and page images for figures/tables.

The Read tool's `pages` option needs pdftoppm, which is not installed here.
PyMuPDF is, so reviewers use this instead.

    python docs/paper-reviews/pdftool.py text  <pdf> [first] [last]   # text to stdout
    python docs/paper-reviews/pdftool.py pages <pdf> [first] [last]   # PNGs, prints paths

Pages are 1-based and inclusive. PNGs land in the scratchpad (or $OUTDIR),
one per page at 150 dpi, named <stem>_pNN.png -- read them with the Read tool
to inspect figures and tables.
"""
import os
import sys
from pathlib import Path

import pymupdf

OUTDIR = Path(os.environ.get(
    "OUTDIR",
    r"C:\Users\ZhuanZ\AppData\Local\Temp\claude\e--trae-project-New-folder--2-"
    r"\65ee7ad3-8639-43ef-af55-848f8c581b19\scratchpad\pdfpages"))


def main() -> None:
    # The Windows console is cp936 here; papers contain ligatures and dashes.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    mode, pdf = sys.argv[1], Path(sys.argv[2])
    doc = pymupdf.open(pdf)
    first = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    last = min(int(sys.argv[4]) if len(sys.argv) > 4 else doc.page_count, doc.page_count)

    if mode == "text":
        print("### %s -- %d pages total" % (pdf.name, doc.page_count))
        for i in range(first - 1, last):
            print("\n===== PAGE %d =====" % (i + 1))
            print(doc[i].get_text())
        return

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for i in range(first - 1, last):
        out = OUTDIR / ("%s_p%02d.png" % (pdf.stem[:28], i + 1))
        doc[i].get_pixmap(dpi=150).save(out)
        print(out)


if __name__ == "__main__":
    main()
