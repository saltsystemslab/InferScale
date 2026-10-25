# /// script
# requires-python = ">=3.10"
# dependencies = ["pymupdf==1.28.2"]
# ///
"""Export the paper's diagrams from its arXiv source as SVGs for the docs site.

The site shows each diagram as it appears in the paper, without the "NOVEL"
tags and dark orange side bars that mark the paper's contributions, and with
the overview's query box drawn plain. Run it again after a new arXiv version
by updating ARXIV_SOURCE:

    uv run docs/scripts/paper_figures.py
"""

from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path

import pymupdf

ARXIV_SOURCE = "https://arxiv.org/e-print/2607.27090v2"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "src" / "assets" / "paper"

# Figure PDF in the source archive -> SVG written for the site.
FIGURES = {
    "inferscale_pipeline_overview.pdf": "overview.svg",
    "saltmem_arch.pdf": "architecture.svg",
    "rope_encoding_example.pdf": "rope-encoding.svg",
}

TAG = "NOVEL"

# Fill of the dark orange bar along the left edge of each contribution box (#e0a53a).
BAR = (224, 165, 58)

# Color swaps per SVG written for the site. The overview's query box drops its
# gold highlight and is drawn like the diagram's plain white boxes.
RECOLOR = {
    "overview.svg": {
        'fill="#fbf4e3"': 'fill="#ffffff"',
        'stroke="#c9ad6a"': 'stroke="#c7ccd3"',
    },
}


def main() -> None:
    archive = download(ARXIV_SOURCE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for source, target in FIGURES.items():
            member = tar.extractfile(source)
            if member is None:
                raise SystemExit(f"{source} is not a file in {ARXIV_SOURCE}")
            svg = recolor(figure_svg(member.read()), RECOLOR.get(target, {}))
            (OUTPUT_DIR / target).write_text(svg, encoding="utf-8")
            print(f"wrote {OUTPUT_DIR / target}")


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "inferscale-docs"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def figure_svg(pdf: bytes) -> str:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        page = doc[0]
        remove_tags(page)
        remove_bars(page)
        fit_to_content(page)
        return page.get_svg_image(text_as_path=True)


def fit_to_content(page: pymupdf.Page, margin: float = 1.0) -> None:
    """Grow the page box over anything drawn past it.

    LaTeX draws a figure's overflow, so the paper shows titles that poke above
    the page box; an SVG viewBox would clip them.
    """
    content = page.rect
    for drawing in page.get_drawings():
        content |= drawing["rect"]
    for x0, y0, x1, y1, *_ in page.get_text("blocks"):
        content |= pymupdf.Rect(x0, y0, x1, y1)
    if content != page.rect:
        box = content + (-margin, -margin, margin, margin)
        page.set_mediabox(box * ~page.transformation_matrix)


def remove_tags(page: pymupdf.Page) -> None:
    """Delete each tag's word and the pill drawn behind it, keeping the box it sits on."""
    words = page.search_for(TAG)
    if not words:
        return
    filled = [drawing["rect"] for drawing in page.get_drawings() if drawing["fill"]]
    for word in words:
        # The pill is the smallest filled shape around the word.
        around = [rect for rect in filled if rect.contains(word)]
        pill = min(around, key=lambda rect: rect.get_area())
        page.add_redact_annot(pill + (-0.5, -0.5, 0.5, 0.5), fill=False)
    # Only shapes that lie entirely inside a pill are removed, so the box borders stay.
    page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_NONE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
    )
    if page.search_for(TAG):
        raise SystemExit(f"{TAG} tags remain after redaction")


def remove_bars(page: pymupdf.Page) -> None:
    """Delete the bar along each contribution box's left edge, keeping the box and its label."""
    bars = find_bars(page)
    if not bars:
        return
    for bar in bars:
        page.add_redact_annot(bar + (-0.5, -0.5, 0.5, 0.5), fill=False)
    # Each bar is the only shape inside its rect, and text is never removed.
    page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_NONE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
        text=pymupdf.PDF_REDACT_TEXT_NONE,
    )
    if find_bars(page):
        raise SystemExit("bars remain after redaction")


def find_bars(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Return the rect of each shape filled with the bar color."""
    return [
        drawing["rect"]
        for drawing in page.get_drawings()
        if drawing["fill"] and tuple(round(c * 255) for c in drawing["fill"]) == BAR
    ]


def recolor(svg: str, swaps: dict[str, str]) -> str:
    """Replace each color attribute, which must occur exactly once in the figure."""
    for old, new in swaps.items():
        count = svg.count(old)
        if count != 1:
            raise SystemExit(f"expected one {old} in the figure, found {count}")
        svg = svg.replace(old, new)
    return svg


if __name__ == "__main__":
    main()
