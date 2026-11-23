from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from loguru import logger

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
MS_PER_SECOND = 1000
LOOP_FOREVER = 0
DOT_STEP_MS = 400
MAX_DOTS = 3
MIN_READING_WIDTH = 11
ELLIPSIS = "..."
RIGHT_ALIGNED = "ra"

GROUND: Color = (11, 18, 32)
CARD: Color = (19, 28, 46)
EDGE: Color = (36, 50, 77)
TEXT: Color = (226, 232, 240)
BRIGHT: Color = (248, 250, 252)
DIM: Color = (124, 138, 165)
BLUE: Color = (56, 189, 248)
ROSE: Color = (248, 113, 113)
AMBER: Color = (251, 191, 36)

BANNER_TOP = 22
QUESTION_TOP = 32
BANNER_GAP = 32
FOOTER_GAP = 20
CLOCK_DROP = 2

CARD_RADIUS = 8
STRIPE_WIDTH = 4
PADDING_LEFT = 22
PADDING_RIGHT = 18
NAME_TOP = 22
DIVIDER_TOP = 70
PREFILLING_TOP = 24
READING_TOP = 14
CAPTION_TOP = 44
BODY_TOP = 86
LINE_HEIGHT = 22


class SideStyle(NamedTuple):
    label: str
    accent: Color


STYLES: dict[str, SideStyle] = {
    FAST_MODE: SideStyle("InferScale", BLUE),
    "mem0": SideStyle("Mem0", ROSE),
    "lmcache": SideStyle("LMCache CacheBlend", AMBER),
}


@dataclass(frozen=True)
class Layout:
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
    name: str
    accent: Color
    row: Row
    events: list[Event]


@dataclass(frozen=True)
class Race:
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
    words = re.findall(r"\S+\s*", row["answer"])
    gap_ms = (row["total_ms"] - row["ttft_ms"]) / max(1, len(words) - 1)
    return [(row["ttft_ms"] + gap_ms * index, word) for index, word in enumerate(words)]


def render_gif(
    timelines: Sequence[Timeline],
    *,
    out: Path,
    question: int | None = None,
    slowdown: float = 20.0,
) -> int:
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Rendering the GIF needs Pillow: pip install pillow")
    speedup_columns = _speedup_columns(timelines)
    chosen = _choose_question(speedup_columns[0], question)
    _log_speedups(timelines, speedup_columns, chosen)
    chosen_speedups = [column[chosen] for column in speedup_columns]
    race = _build_race(timelines, chosen, chosen_speedups, slowdown)

    playback_ms = race.end_ms * slowdown
    frame_ms = max(MS_PER_SECOND / FRAMES_PER_SECOND, playback_ms / MAX_FRAMES)
    frame_count = max(1, math.ceil(playback_ms / frame_ms))
    frames = [
        _draw_frame(Image, race, race.end_ms * index / frame_count)
        for index in range(frame_count + 1)
    ]
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=[round(frame_ms)] * frame_count + [FINAL_HOLD_MS],
        loop=LOOP_FOREVER,
    )
    readings = " vs ".join(
        f"{timeline['rows'][chosen]['ttft_ms']:.0f} ms ({label(timeline)})"
        for timeline in timelines
    )
    logger.info("wrote {}: question {}, first token {}", out, chosen, readings)
    return chosen


def label(timeline: Timeline) -> str:
    return STYLES[timeline["mode"]].label


def load_font(size: int, bold: bool = False, mono: bool = True) -> Any:
    from PIL import ImageFont

    dejavu = f"DejaVuSans{'Mono' if mono else ''}{'-Bold' if bold else ''}.ttf"
    macos_face = ("Menlo.ttc" if mono else "Helvetica.ttc", int(bold))
    faces = [(dejavu, 0)]
    try:
        from matplotlib import get_data_path

        faces.append((str(Path(get_data_path(), "fonts", "ttf", dejavu)), 0))
    except ImportError:
        pass
    faces.append(macos_face)
    for font_file, face_index in faces:
        try:
            return ImageFont.truetype(font_file, size, index=face_index)
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


def _speedup_columns(timelines: Sequence[Timeline]) -> list[list[float]]:
    fast, baselines = timelines[0], timelines[1:]
    return [
        [
            baseline_row["ttft_ms"] / fast_row["ttft_ms"]
            for fast_row, baseline_row in zip(fast["rows"], baseline["rows"])
        ]
        for baseline in baselines
    ]


def _choose_question(speedups: list[float], requested: int | None) -> int:
    if requested is None:
        return _median_speedup_question(speedups)
    if not 0 <= requested < len(speedups):
        sys.exit(f"--question must be between 0 and {len(speedups) - 1}")
    return requested


def _median_speedup_question(speedups: list[float]) -> int:
    by_speedup = sorted(range(len(speedups)), key=speedups.__getitem__)
    return by_speedup[len(speedups) // 2]


def _log_speedups(
    timelines: Sequence[Timeline], speedup_columns: list[list[float]], chosen: int
) -> None:
    widths = [max(MIN_READING_WIDTH, len(label(timeline))) for timeline in timelines]
    header = "  ".join(f"{label(timeline):>{width}}" for timeline, width in zip(timelines, widths))
    columns = "  ".join(f"{'speedup':>7}" for _ in speedup_columns)
    logger.info("     {}  {}  question", header, columns)
    for index, question in enumerate(row["question"] for row in timelines[0]["rows"]):
        marker = "*" if index == chosen else " "
        readings = "  ".join(
            f"{timeline['rows'][index]['ttft_ms']:>{width - 3}.1f} ms"
            for timeline, width in zip(timelines, widths)
        )
        ratios = "  ".join(f"{column[index]:>6.2f}x" for column in speedup_columns)
        logger.info("{}{:>2}  {}  {}  {}", marker, index, readings, ratios, question)


def _build_race(
    timelines: Sequence[Timeline], chosen: int, speedups: Sequence[float], slowdown: float
) -> Race:
    rows = [timeline["rows"][chosen] for timeline in timelines]
    layout = Layout(columns=len(timelines))
    fonts = Fonts.load()
    panels = tuple(
        Panel(
            name=_panel_name(timeline),
            accent=STYLES[timeline["mode"]].accent,
            row=row,
            events=stream(row),
        )
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
    if len(baselines) == 1:
        _, speedup = baselines[0]
        return "InferScale: first token " + _sooner_or_later(speedup)
    return "InferScale: first token " + ", ".join(
        f"{_sooner_or_later(speedup)} than {name}" for name, speedup in baselines
    )


def _sooner_or_later(speedup: float) -> str:
    return f"{speedup:.1f}x sooner" if speedup >= 1 else f"{1 / speedup:.1f}x later"


def _shortened_question(question: str, layout: Layout, fonts: Fonts, banner: str) -> str:
    asked = "LoCoMo question: " + question
    room = layout.width - 2 * layout.margin - BANNER_GAP - fonts.title.getlength(banner)
    while fonts.note.getlength(asked) > room:
        asked = asked[: -len(ELLIPSIS) - 1].rstrip() + ELLIPSIS
    return asked


def _draw_frame(image_module: Any, race: Race, now_ms: float) -> Any:
    from PIL import ImageDraw

    layout, fonts = race.layout, race.fonts
    image = image_module.new("RGB", (layout.width, layout.height), GROUND)
    draw = ImageDraw.Draw(image)
    right = layout.width - layout.margin
    draw.text((layout.margin, BANNER_TOP), race.banner, font=fonts.title, fill=BRIGHT)
    draw.text((right, QUESTION_TOP), race.asked, font=fonts.note, fill=DIM, anchor=RIGHT_ALIGNED)
    for column, panel in enumerate(race.panels):
        _draw_panel(draw, race, column, panel, now_ms)
    footer_top = layout.floor + FOOTER_GAP
    clock = f"t = {now_ms:,.0f} ms{race.footer_note}"
    draw.text((layout.margin, footer_top + CLOCK_DROP), clock, font=fonts.note, fill=DIM)
    if now_ms >= race.all_started_ms:
        draw.text(
            (right, footer_top), race.verdict, font=fonts.strong, fill=TEXT, anchor=RIGHT_ALIGNED
        )
    return image


def _draw_panel(draw: Any, race: Race, column: int, panel: Panel, now_ms: float) -> None:
    layout, fonts = race.layout, race.fonts
    left = layout.panel_left(column)
    right, top, floor = left + layout.panel_width, layout.top, layout.floor
    text_left, text_right = left + PADDING_LEFT, right - PADDING_RIGHT
    _draw_striped_card(draw, left, top, right, floor, panel.accent)
    draw.text((text_left, top + NAME_TOP), panel.name, font=fonts.heading, fill=TEXT)
    draw.line((text_left, top + DIVIDER_TOP, text_right, top + DIVIDER_TOP), fill=EDGE)
    if now_ms < panel.row["ttft_ms"]:
        _draw_prefilling(draw, race, text_right, top, now_ms)
        return
    reading = f"{panel.row['ttft_ms']:,.0f} ms"
    draw.text(
        (text_right, top + READING_TOP),
        reading,
        font=fonts.clock,
        fill=panel.accent,
        anchor=RIGHT_ALIGNED,
    )
    draw.text(
        (text_right, top + CAPTION_TOP),
        "first token",
        font=fonts.note,
        fill=DIM,
        anchor=RIGHT_ALIGNED,
    )
    shown = "".join(word for shown_at_ms, word in panel.events if shown_at_ms <= now_ms).strip()
    lines = wrap_text(draw, shown, fonts.body, text_right - text_left)[:BODY_LINES]
    for index, line in enumerate(lines):
        line_top = top + BODY_TOP + LINE_HEIGHT * index
        draw.text((text_left, line_top), line, font=fonts.body, fill=TEXT)


def _draw_striped_card(
    draw: Any, left: int, top: int, right: int, floor: int, accent: Color
) -> None:
    accent_underlay = (left, top, right, floor)
    card = (left + STRIPE_WIDTH, top, right, floor)
    draw.rounded_rectangle(accent_underlay, radius=CARD_RADIUS, fill=accent)
    draw.rounded_rectangle(card, radius=CARD_RADIUS, fill=CARD, outline=EDGE)


def _draw_prefilling(draw: Any, race: Race, text_right: int, top: int, now_ms: float) -> None:
    font = race.fonts.strong
    working_dots = "." * (1 + int(now_ms * race.slowdown / DOT_STEP_MS) % MAX_DOTS)
    widest = "prefilling..."
    left = text_right - draw.textlength(widest, font=font)
    draw.text((left, top + PREFILLING_TOP), "prefilling" + working_dots, font=font, fill=DIM)
