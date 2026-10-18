"""Figure 1, authored as SVG and rasterised through the local Chromium.

Why SVG rather than the AI image skills: every image-generation skill in the
vault needs a credential (OpenRouter, Gemini, fal, Stability) and none is
present. They would also have sent an unpublished manuscript to a third party
and, by their own documentation, routinely misspell labels -- unacceptable for
a figure carrying "194 trees" and "AUROC 0.897".

Why SVG rather than matplotlib: precise typographic control, real vector
output, and room for a third panel without the layout fighting back.

What the figure now explains that the earlier one did not:
  (a) what each development stage PRODUCES, not just its name;
  (b) the week, with the lock that makes weight of advice measurable and the
      return arrow that carries state forward;
  (c) NEW -- the session, so a reader sees that the week in (b) is one of 30,
      that five practice weeks precede them, and where the disruptions fall.

Run:  python figures/make_figure1_svg.py
Out:  figures/paper/main_text/Figure1_merged_workflow.png
"""
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "paper" / "main_text" / "Figure1_merged_workflow.png"
SVG = HERE / "paper" / "main_text" / "Figure1_merged_workflow.svg"

W, H = 1220, 356          # SVG user units; 1220 maps to the 122 mm print width
INK = "#1a1a19"
GREY_A = "#eceae3"        # development stages
GREY_M = "#c7c5bd"        # the frozen model
WHITE = "#ffffff"
DISRUPT = "#9a9890"

FONT = ("-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif")


# ---------------------------------------------------------------- icons
# Simple stroked glyphs, 24x24 in their own coordinate space, translated and
# scaled into place. Stroke-only so they read in greyscale and at small size.
ICON = {
    # stacked discs: open data
    "data": '<ellipse cx="12" cy="6" rx="8" ry="3.1"/>'
            '<path d="M4 6 V12 a8 3.1 0 0 0 16 0 V6"/>'
            '<path d="M4 12 V17 a8 3.1 0 0 0 16 0 V12"/>',
    # warehouse: the digital twin
    "twin": '<path d="M3 20 V10 L12 4 L21 10 V20 Z"/>'
            '<path d="M9 20 V14 H15 V20"/>',
    # grid of cells: engineered features
    "grid": '<rect x="3" y="3" width="18" height="18" rx="1.5"/>'
            '<path d="M9 3 V21 M15 3 V21 M3 9 H21 M3 15 H21"/>',
    # branching: a boosted tree ensemble
    "tree": '<path d="M12 21 V13"/><path d="M12 13 L6 8 M12 13 L18 8"/>'
            '<path d="M6 8 L3 4 M6 8 L9 4 M18 8 L15 4 M18 8 L21 4"/>',
    # padlock: the model is frozen
    "lock": '<rect x="4.5" y="10" width="15" height="10" rx="2"/>'
            '<path d="M8 10 V7 a4 4 0 0 1 8 0 V10"/>',
    # eye: inspect the state
    "eye": '<path d="M2 12 C6 5.5, 18 5.5, 22 12 C18 18.5, 6 18.5, 2 12 Z"/>'
           '<circle cx="12" cy="12" r="3.1"/>',
    # pencil: the participant enters their own order
    "pen": '<path d="M4 20 L5.4 15.2 L16.6 4 L20 7.4 L8.8 18.6 Z"/>'
           '<path d="M14.6 6 L18 9.4"/>',
    # chip: the AI recommendation
    "chip": '<rect x="6" y="6" width="12" height="12" rx="1.6"/>'
            '<path d="M10 2 V6 M14 2 V6 M10 18 V22 M14 18 V22'
            ' M2 10 H6 M2 14 H6 M18 10 H22 M18 14 H22"/>',
    # tick: the final committed order
    "tick": '<path d="M3.5 12.8 L9 18.2 L20.5 6.4"/>',
}


def icon(name, x, y, size=17):
    """Place a stroked icon with its top-left at (x, y)."""
    sc = size / 24.0
    return ('<g transform="translate(%g,%g) scale(%g)" fill="none" stroke="%s" '
            'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
            '%s</g>' % (x, y, sc, INK, ICON[name]))

def box(x, y, w, h, title, sub, fill, ts=15, ss=13, ic=None):
    s = ['<rect x="%d" y="%d" width="%d" height="%d" rx="7" ry="7" '
         'fill="%s" stroke="%s" stroke-width="1.4"/>' % (x, y, w, h, fill, INK)]
    if ic:
        tw = len(title) * ts * 0.54          # rough width of the title text
        s.append(icon(ic, x + w / 2 - tw / 2 - 23, y + 7))
        s.append('<text x="%d" y="%d" text-anchor="start" font-family="%s" '
                 'font-size="%d" font-weight="600" fill="%s">%s</text>'
                 % (x + w / 2 - tw / 2, y + 21, FONT, ts, INK, title))
    else:
        s.append('<text x="%d" y="%d" text-anchor="middle" font-family="%s" '
                 'font-size="%d" font-weight="600" fill="%s">%s</text>'
                 % (x + w / 2, y + 21, FONT, ts, INK, title))
    for i, ln in enumerate(sub):
        s.append('<text x="%d" y="%d" text-anchor="middle" font-family="%s" '
                 'font-size="%d" fill="%s">%s</text>'
                 % (x + w / 2, y + 41 + i * 16, FONT, ss, INK, ln))
    return "".join(s)


def arrow(x1, y1, x2, y2, dashed=False, w=1.5):
    d = ' stroke-dasharray="5,3.5"' if dashed else ""
    return ('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" '
            'stroke-width="%g"%s marker-end="url(#ah)"/>'
            % (x1, y1, x2, y2, INK, w, d))


def label(x, y, t, size=12, anchor="middle", weight="400", fill=INK):
    return ('<text x="%g" y="%g" text-anchor="%s" font-family="%s" '
            'font-size="%d" font-weight="%s" fill="%s">%s</text>'
            % (x, y, anchor, FONT, size, weight, fill, t))


def build():
    p = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
         'viewBox="0 0 %d %d">' % (W, H, W, H),
         '<defs><marker id="ah" markerWidth="7" markerHeight="7" refX="6.2" '
         'refY="2.6" orient="auto"><path d="M0,0 L0,5.2 L6.4,2.6 z" fill="%s"/>'
         '</marker></defs>' % INK,
         '<rect width="%d" height="%d" fill="white"/>' % (W, H)]

    # ------------------------------------------------------------- panel (a)
    p.append(label(4, 20, "a", 17, "start", "700"))
    p.append(label(26, 20, "Model development — once, before any participant",
                   15, "start", "600"))
    stages = [
        ("Open data", ["M5, Olist,", "DataCo"], "data"),
        ("Digital twin", ["520 episodes", "x 100 weeks"], "twin"),
        ("Features", ["30 inputs: state", "+ 4 wk history"], "grid"),
        ("LightGBM", ["194 trees,", "Platt scaled"], "tree"),
    ]
    bw, bh, gap, y0 = 196, 76, 42, 32
    for i, (t, sub, ic) in enumerate(stages):
        x = i * (bw + gap)
        p.append(box(x, y0, bw, bh, t, sub, GREY_A, ic=ic))
        if i:
            p.append(arrow(x - gap + 3, y0 + bh / 2, x - 3, y0 + bh / 2))
    mx = 4 * (bw + gap)
    p.append(box(mx, y0, W - mx, bh, "Frozen model",
                 ["AUROC 0.897", "ECE 0.006"], GREY_M, ic="lock"))
    p.append(arrow(mx - gap + 3, y0 + bh / 2, mx - 3, y0 + bh / 2))

    # ------------------------------------------------------------- panel (b)
    yb = 152
    p.append(label(4, yb, "b", 17, "start", "700"))
    p.append(label(26, yb, "One week — what the participant does", 15, "start", "600"))
    steps = [
        ("1  Inspect", ["stock, demand,", "arrivals, backorders"], "eye"),
        ("2  Own order", ["entered and", "locked"], "pen"),
        ("3  AI advice", ["recommended order", "+ risk %"], "chip"),
        ("4  Final order", ["accept, modify", "or override"], "tick"),
    ]
    sw, sh, sg, ys = 272, 80, 44, yb + 16
    xs = []
    for i, (t, sub, ic) in enumerate(steps):
        x = i * (sw + sg)
        xs.append(x)
        p.append(box(x, ys, sw, sh, t, sub, WHITE, ic=ic))
        if i:
            p.append(arrow(x - sg + 3, ys + sh / 2, x - 3, ys + sh / 2))

    # the lock between step 2 and step 3
    lx = xs[2] - sg / 2
    p.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="2" '
             'stroke-dasharray="3,3"/>' % (lx, ys + 4, lx, ys + sh - 4, INK))
    p.append(label(lx, ys - 7, "locked", 11.5, "middle", "600"))

    # frozen model feeds step 3
    p.append('<path d="M %g %g C %g %g, %g %g, %g %g" fill="none" stroke="%s" '
             'stroke-width="1.5" stroke-dasharray="5,3.5" marker-end="url(#ah)"/>'
             % (mx + (W - mx) / 2, y0 + bh + 2, mx + (W - mx) / 2, y0 + bh + 46,
                xs[2] + sw * 0.72, ys - 46, xs[2] + sw * 0.72, ys - 3, INK))
    p.append(label(W - 8, y0 + bh + 34, "risk shown on screen", 12, "end"))

    # return loop, below the row
    yl = ys + sh + 30
    p.append('<path d="M %g %g L %g %g L %g %g L %g %g" fill="none" stroke="%s" '
             'stroke-width="1.6" marker-end="url(#ah)"/>'
             % (xs[3] + sw / 2, ys + sh + 2, xs[3] + sw / 2, yl,
                xs[0] + sw / 2, yl, xs[0] + sw / 2, ys + sh + 4, INK))
    p.append(label(W / 2, yl + 17,
                   "inventory, pipeline orders and backorders carry into week t+1", 12.5))

    # ------------------------------------------------------------- panel (c)
    yc = 268
    p.append(label(4, yc, "c", 17, "start", "700"))
    p.append(label(26, yc, "One session — the week above, repeated", 15, "start", "600"))
    tx, tw, ty, th = 26, W - 52, yc + 22, 26
    p.append('<rect x="%d" y="%d" width="%d" height="%d" fill="white" stroke="%s" '
             'stroke-width="1.3"/>' % (tx, ty, tw, th, INK))
    # practice block
    pw = tw * 5.0 / 35.0
    p.append('<rect x="%d" y="%d" width="%g" height="%d" fill="%s" stroke="%s" '
             'stroke-width="1.3"/>' % (tx, ty, pw, th, GREY_A, INK))
    p.append(label(tx + pw / 2, ty + 17, "practice 5", 11.5))
    p.append(label(tx + pw + (tw - pw) / 2, ty + 17, "30 main weeks", 12.5))
    # disruption weeks, from the paper: 6, 9, 13, 17, 21, 26
    for wk in (6, 9, 13, 17, 21, 26):
        x = tx + pw + (tw - pw) * (wk - 0.5) / 30.0
        p.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="%s" '
                 'stroke-width="2.4"/>' % (x, ty + th, x, ty + th + 9, DISRUPT))
    p.append(label(tx + pw + (tw - pw) * 0.5, ty + th + 26,
                   "vertical ticks mark the six disruption weeks", 12))

    p.append("</svg>")
    return "".join(p)


def main():
    SVG.parent.mkdir(parents=True, exist_ok=True)
    SVG.write_text(build(), encoding="utf-8")
    print("wrote", SVG.name)

    from playwright.sync_api import sync_playwright
    scale = 3  # 1220 px * 3 = 3660 px across 122 mm -> about 760 dpi
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H},
                        device_scale_factor=scale)
        pg.goto(SVG.as_uri())
        pg.wait_for_timeout(350)
        pg.screenshot(path=str(OUT), omit_background=False)
        b.close()
    print("wrote", OUT.name)
    from PIL import Image
    im = Image.open(OUT)
    print("%dx%d px -> at 122 mm = %.1f mm tall, %.0f effective dpi"
          % (im.size[0], im.size[1], 122 * im.size[1] / im.size[0],
             im.size[0] / (122 / 25.4)))


if __name__ == "__main__":
    main()
