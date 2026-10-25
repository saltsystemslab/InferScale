"""Replay one question of several timelines as a GIF: every answer on one shared clock.

The clock starts when the engine receives the request. Each panel shows its
side's first-token time and then its answer, word by word. The InferScale
timeline comes first; every other timeline is a baseline it is compared with.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Timeline = dict[str, Any]
Row = dict[str, Any]
Event = tuple[float, str]
Color = tuple[int, int, int]

FRAMES_PER_SECOND = 20
MAX_FRAMES = 240
FINAL_HOLD_MS = 3000
BODY_LINES = 11
PANEL_COLUMN_WIDTH = 480
FAST_MODE = "inferscale"

GROUND: Color = (11, 18, 32)
CARD: Color = (19, 28, 46)
EDGE: Color = (36, 50, 77)
TEXT: Color = (226, 232, 240)
BRIGHT: Color = (248, 250, 252)
DIM: Color = (124, 138, 165)
BLUE: Color = (56, 189, 248)
ROSE: Color = (248, 113, 113)
AMBER: Color = (251, 191, 36)

# The label and accent color of each side a timeline can come from.
STYLES: dict[str, tuple[str, Color]] = {
    FAST_MODE: ("InferScale", BLUE),
    "mem0": ("Mem0", ROSE),
    "lmcache": ("LMCache CacheBlend", AMBER),
}


@dataclass(frozen=True)
class Layout:
    """Pixel geometry of the frame: one panel per column under a title row, above a footer."""

    columns: int = 2
    height: int = 480
    margin: int = 28
    gap: int = 16
    top: int = 72
    bottom: int = 56

    @property
    def width(self) -> int:
        return PANEL_COLUMN_WIDTH * self.columns

    @property
    def panel_width(self) -> int:
        return (self.width - 2 * self.margin - self.gap * (self.columns - 1)) // self.columns

    @property
    def floor(self) -> int:
        return self.height - self.bottom

    def panel_left(self, column: int) -> int:
        return self.margin + column * (self.panel_width + self.gap)


@dataclass(frozen=True)
class Fonts:
    title: Any
    heading: Any
    strong: Any
    note: Any
    clock: Any
    body: Any

    @classmethod
    def load(cls) -> Fonts:
        return cls(
            title=load_font(24, bold=True, mono=False),
            heading=load_font(19, bold=True, mono=False),
            strong=load_font(16, bold=True, mono=False),
            note=load_font(13, mono=False),
            clock=load_font(22, bold=True),
            body=load_font(15),
        )


@dataclass(frozen=True)
class Panel:
    """One side of the race: its label, accent color, measured row, and timed words."""

    name: str
    accent: Color
    row: Row
    events: list[Event]


@dataclass(frozen=True)
class Race:
    """Everything a frame needs that does not change between frames."""

    panels: tuple[Panel, ...]
    banner: str
    asked: str
    verdict: str
    footer_note: str
    end_ms: float
    slowdown: float
    layout: Layout
    fonts: Fonts

    @property
    def all_started_ms(self) -> float:
        return max(panel.row["ttft_ms"] for panel in self.panels)


def load_timelines(paths: Sequence[Path]) -> list[Timeline]:
    """The InferScale timeline followed by its baselines, all over the same questions."""
    if len(paths) < 2:
        sys.exit("Pass the InferScale timeline and at least one baseline timeline.")
    timelines = [json.loads(Path(path).read_text()) for path in paths]
    modes = [timeline["mode"] for timeline in timelines]
    if modes[0] != FAST_MODE or FAST_MODE in modes[1:]:
        sys.exit("Pass the InferScale timeline first and the baseline timelines after it.")
    unknown = [mode for mode in modes if mode not in STYLES]
    if unknown:
        sys.exit(f"Unknown timeline mode {unknown[0]!r}; expected one of {', '.join(STYLES)}.")
    if len(set(modes)) != len(modes):
        sys.exit("Each baseline timeline must come from a different side.")
    questions = [[row["question"] for row in timeline["rows"]] for timeline in timelines]
    if any(other != questions[0] for other in questions[1:]):
        sys.exit("The sides answered different questions; use the same config and --questions.")
    return timelines


def stream(row: Row) -> list[Event]:
    """(ms since the request, word) pairs for one answer.

    The engine generates offline, so only the first token and the finished
    answer are timed; the words in between are spaced evenly.
    """
    words = re.findall(r"\S+\s*", row["answer"])
    gap = (row["total_ms"] - row["ttft_ms"]) / max(1, len(words) - 1)
    return [(row["ttft_ms"] + gap * index, word) for index, word in enumerate(words)]


def render_gif(
    timelines: Sequence[Timeline],
    *,
    out: Path,
    question: int | None = None,
    slowdown: float = 20.0,
) -> int:
    """Write the GIF for one question and return its index.

    The default question is the median speedup over the first baseline, so the
    replay is representative rather than the best case.
    """
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Rendering the GIF needs Pillow: pip install pillow")
    fast, baselines = timelines[0], timelines[1:]
    speedups = [
        [slow["ttft_ms"] / quick["ttft_ms"] for quick, slow in zip(fast["rows"], base["rows"])]
        for base in baselines
    ]
    chosen = _choose_question(speedups[0], question)
    _print_speedups(timelines, speedups, chosen)
    race = _build_race(timelines, chosen, [column[chosen] for column in speedups], slowdown)

    # Twenty frames per second of playback, fewer when that would exceed MAX_FRAMES.
    step_ms = max(1000 / FRAMES_PER_SECOND, race.end_ms * slowdown / MAX_FRAMES)
    count = max(1, math.ceil(race.end_ms * slowdown / step_ms))
    frames = [_draw_frame(Image, race, race.end_ms * index / count) for index in range(count + 1)]
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=[round(step_ms)] * count + [FINAL_HOLD_MS],
        loop=0,
    )
    readings = " vs ".join(
        f"{timeline['rows'][chosen]['ttft_ms']:.0f} ms ({label(timeline)})"
        for timeline in timelines
    )
    print(f"wrote {out}: question {chosen}, first token {readings}")
    return chosen


def label(timeline: Timeline) -> str:
    return STYLES[timeline["mode"]][0]


def load_font(size: int, bold: bool = False, mono: bool = True) -> Any:
    """DejaVu Sans or Sans Mono (installed, or bundled with matplotlib), then a macOS face."""
    from PIL import ImageFont

    dejavu = f"DejaVuSans{'Mono' if mono else ''}{'-Bold' if bold else ''}.ttf"
    # (file, face index in the file)
    candidates = [(dejavu, 0), ("Menlo.ttc" if mono else "Helvetica.ttc", int(bold))]
    try:
        from matplotlib import get_data_path

        candidates.insert(1, (str(Path(get_data_path(), "fonts", "ttf", dejavu)), 0))
    except ImportError:
        pass
    for name, index in candidates:
        try:
            return ImageFont.truetype(name, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default(size)


def wrap_text(draw: Any, text: str, font: Any, width: int) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            longer = f"{line} {word}" if line else word
            if line and draw.textlength(longer, font=font) > width:
                lines.append(line)
                line = word
            else:
                line = longer
        lines.append(line)
    return lines


def _choose_question(speedups: list[float], requested: int | None) -> int:
    if requested is None:
        # A representative question, not the best one: the median speedup.
        return sorted(range(len(speedups)), key=speedups.__getitem__)[len(speedups) // 2]
    if not 0 <= requested < len(speedups):
        sys.exit(f"--question must be between 0 and {len(speedups) - 1}")
    return requested


def _print_speedups(
    timelines: Sequence[Timeline], speedups: list[list[float]], chosen: int
) -> None:
    """One first-token column per side and one speedup column per baseline."""
    widths = [max(11, len(label(timeline))) for timeline in timelines]
    header = "  ".join(f"{label(timeline):>{width}}" for timeline, width in zip(timelines, widths))
    columns = "  ".join(f"{'speedup':>7}" for _ in speedups)
    print(f"     {header}  {columns}  question")
    for index, question in enumerate(row["question"] for row in timelines[0]["rows"]):
        marker = "*" if index == chosen else " "
        readings = "  ".join(
            f"{timeline['rows'][index]['ttft_ms']:>{width - 3}.1f} ms"
            for timeline, width in zip(timelines, widths)
        )
        ratios = "  ".join(f"{column[index]:>6.2f}x" for column in speedups)
        print(f"{marker}{index:>2}  {readings}  {ratios}  {question}")


def _build_race(
    timelines: Sequence[Timeline], chosen: int, speedups: Sequence[float], slowdown: float
) -> Race:
    rows = [timeline["rows"][chosen] for timeline in timelines]
    layout = Layout(columns=len(timelines))
    fonts = Fonts.load()
    panels = tuple(
        Panel(_panel_name(timeline), STYLES[timeline["mode"]][1], row, stream(row))
        for timeline, row in zip(timelines, rows)
    )
    labels = [label(timeline) for timeline in timelines]
    banner = " vs ".join(labels)
    footer_note = f"   playback slowed {slowdown:g}x" if slowdown != 1 else ""
    return Race(
        panels=panels,
        banner=banner,
        asked=_shortened_question(rows[0]["question"], layout, fonts, banner),
        verdict=_verdict(list(zip(labels[1:], speedups))),
        footer_note=footer_note,
        end_ms=max(row["total_ms"] for row in rows),
        slowdown=slowdown,
        layout=layout,
        fonts=fonts,
    )


def _panel_name(timeline: Timeline) -> str:
    if timeline["mode"] == FAST_MODE:
        return f"{label(timeline)} (k={timeline['top_k']}, w={timeline['context_window']})"
    return f"{label(timeline)} (k={timeline['top_k']})"


def _verdict(baselines: Sequence[tuple[str, float]]) -> str:
    """How much sooner InferScale's first token came than each baseline's."""
    if len(baselines) == 1:
        return "InferScale: first token " + _sooner_or_later(baselines[0][1])
    return "InferScale: first token " + ", ".join(
        f"{_sooner_or_later(speedup)} than {name}" for name, speedup in baselines
    )


def _sooner_or_later(speedup: float) -> str:
    return f"{speedup:.1f}x sooner" if speedup >= 1 else f"{1 / speedup:.1f}x later"


def _shortened_question(question: str, layout: Layout, fonts: Fonts, banner: str) -> str:
    """The question shares the title row, so it is cut to stay clear of the banner."""
    asked = "LoCoMo question: " + question
    room = layout.width - 2 * layout.margin - 32 - fonts.title.getlength(banner)
    while fonts.note.getlength(asked) > room:
        asked = asked[:-4].rstrip() + "..."
    return asked


def _draw_frame(image_module: Any, race: Race, now: float) -> Any:
    from PIL import ImageDraw

    layout, fonts = race.layout, race.fonts
    image = image_module.new("RGB", (layout.width, layout.height), GROUND)
    draw = ImageDraw.Draw(image)
    draw.text((layout.margin, 22), race.banner, font=fonts.title, fill=BRIGHT)
    draw.text(
        (layout.width - layout.margin, 32), race.asked, font=fonts.note, fill=DIM, anchor="ra"
    )
    for column, panel in enumerate(race.panels):
        _draw_panel(draw, race, column, panel, now)
    foot = layout.floor + 20
    draw.text(
        (layout.margin, foot + 2), f"t = {now:,.0f} ms{race.footer_note}", font=fonts.note, fill=DIM
    )
    if now >= race.all_started_ms:
        draw.text(
            (layout.width - layout.margin, foot),
            race.verdict,
            font=fonts.strong,
            fill=TEXT,
            anchor="ra",
        )
    return image


def _draw_panel(draw: Any, race: Race, column: int, panel: Panel, now: float) -> None:
    layout, fonts = race.layout, race.fonts
    left = layout.panel_left(column)
    right, top, floor = left + layout.panel_width, layout.top, layout.floor
    # An accent-colored card under the real one leaves a stripe down its left edge.
    draw.rounded_rectangle((left, top, right, floor), radius=8, fill=panel.accent)
    draw.rounded_rectangle((left + 4, top, right, floor), radius=8, fill=CARD, outline=EDGE)
    draw.text((left + 22, top + 22), panel.name, font=fonts.heading, fill=TEXT)
    draw.line((left + 22, top + 70, right - 18, top + 70), fill=EDGE)
    if now < panel.row["ttft_ms"]:
        # Every side starts together; the dots keep the slower ones visibly working.
        dots = "." * (1 + int(now * race.slowdown / 400) % 3)
        start = right - 18 - draw.textlength("prefilling...", font=fonts.strong)
        draw.text((start, top + 24), "prefilling" + dots, font=fonts.strong, fill=DIM)
        return
    reading = f"{panel.row['ttft_ms']:,.0f} ms"
    draw.text((right - 18, top + 14), reading, font=fonts.clock, fill=panel.accent, anchor="ra")
    draw.text((right - 18, top + 44), "first token", font=fonts.note, fill=DIM, anchor="ra")
    shown = "".join(word for at, word in panel.events if at <= now).strip()
    lines = wrap_text(draw, shown, fonts.body, layout.panel_width - 40)[:BODY_LINES]
    for index, line in enumerate(lines):
        draw.text((left + 22, top + 86 + 22 * index), line, font=fonts.body, fill=TEXT)
