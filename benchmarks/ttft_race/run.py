"""Side-by-side TTFT on LoCoMo questions: Mem0 prompt text vs InferScale KV injection.

Measure each side on its own, so nothing shares a GPU, then replay one question:

  python -m benchmarks.ttft_race.run measure --mode mem0       --out mem0.json
  python -m benchmarks.ttft_race.run measure --mode inferscale --out inferscale.json
  python -m benchmarks.ttft_race.run gif inferscale.json mem0.json --out ttft_race.gif

A measurement is the memory benchmark itself, run on the first questions of the first
LoCoMo conversation. Both sides retrieve TOP_K facts from the same Mem0 fact catalog.
Mem0 reads them as prompt text; InferScale injects the KV it precomputed for each
fact with CONTEXT_WINDOW preceding turns as the encoding prefix.
The first WARMUP questions absorb the engine's one-off startup costs and are left out.
They differ from the timed ones, so no timed prompt is already in vLLM's prefix cache.

The GIF replays one question as two panels on a shared clock that starts when the
engine receives the request, each showing its first-token time and its answer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

CONFIG = Path("configs/memory/accuracy-latency/llama.json")
TOP_K = 100  # facts retrieved per question, on both sides
CONTEXT_WINDOW = 50  # preceding turns in each fact's InferScale encoding prefix
WARMUP = 3  # leading questions answered before the timed ones


def measure(args: argparse.Namespace) -> None:
    from benchmarks.common.config import (
        expand_path,
        load_json_object,
        load_runtime_config,
        stamp_now,
    )
    from benchmarks.memory.config import MemoryCell, MemoryRunConfig

    # The sweep's Mem0 baseline and InferScale cells: variant, answer and vector backends, k, w.
    cell = {
        "mem0": MemoryCell("prompt-injection", "prompt-injection", "qdrant", TOP_K, 0),
        "inferscale": MemoryCell("kv", "kv-injection", "jasper", TOP_K, CONTEXT_WINDOW),
    }[args.mode]
    runtime = load_runtime_config()
    data = cell.materialize(load_json_object(expand_path(args.config, root=runtime.root)))
    data.update(
        run_id=f"ttft-race-{args.mode}-{stamp_now()}",
        max_samples=1,
        max_questions=WARMUP + args.questions,
        skip_judge=True,
    )
    config = MemoryRunConfig.from_dict(data, runtime=runtime)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    runtime.apply_environment()  # Cache locations, before the GPU libraries load.

    from benchmarks.common.files import read_jsonl
    from benchmarks.memory.runner import run_benchmark

    run_benchmark(config)
    rows = [
        {
            "question": record["question"],
            "gold": record["gold_answer"],
            "answer": record["predicted_answer"],
            "ttft_ms": record["metrics"]["time_to_first_token_ms"],  # vLLM's own measurement
            "total_ms": record["metrics"]["answer_generate_time_ms"],
        }
        for record in read_jsonl(config.run_dir / "predictions.jsonl")[WARMUP:]
    ]
    for index, row in enumerate(rows):
        print(f"{index:>2}  TTFT {row['ttft_ms']:>7.1f} ms  {row['question']}")
    side = {
        "mode": args.mode,
        "model": config.model,
        "top_k": config.top_k,
        "context_window": config.context_window,
        "rows": rows,
    }
    args.out.write_text(json.dumps(side, indent=1))
    print(f"wrote {args.out}; the full benchmark run is in {config.run_dir}")


def load_pair(args: argparse.Namespace) -> tuple[dict, dict]:
    fast, base = json.loads(args.inferscale.read_text()), json.loads(args.mem0.read_text())
    if (fast["mode"], base["mode"]) != ("inferscale", "mem0"):
        sys.exit("Pass the InferScale timeline first and the Mem0 timeline second.")
    if [r["question"] for r in fast["rows"]] != [r["question"] for r in base["rows"]]:
        sys.exit("The two sides answered different questions; use the same config and --questions.")
    return fast, base


def stream(row: dict) -> list[tuple[float, str]]:
    """(ms since the request, word) pairs for one answer.

    The engine generates offline, so only the first token and the finished answer are
    timed; the words in between are spaced evenly.
    """
    words = re.findall(r"\S+\s*", row["answer"])
    gap = (row["total_ms"] - row["ttft_ms"]) / max(1, len(words) - 1)
    return [(row["ttft_ms"] + gap * index, word) for index, word in enumerate(words)]


def load_font(size: int, bold: bool = False, mono: bool = True):
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


def wrap(draw, text: str, font, width: int) -> list[str]:
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


def render(args: argparse.Namespace) -> None:
    """Replay one measured question as a GIF: both answers on one clock."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        sys.exit("Rendering the GIF needs Pillow: pip install pillow")
    fast, base = load_pair(args)
    pairs = list(zip(fast["rows"], base["rows"]))
    speedups = [b["ttft_ms"] / f["ttft_ms"] for f, b in pairs]
    question = args.question
    if question is None:  # A representative question, not the best one: the median speedup.
        question = sorted(range(len(pairs)), key=speedups.__getitem__)[len(pairs) // 2]
    if not 0 <= question < len(pairs):
        sys.exit(f"--question must be between 0 and {len(pairs) - 1}")
    print(f"     {'InferScale':>11}  {'Mem0':>11}  {'speedup':>7}  question")
    for index, (f, b) in enumerate(pairs):
        print(
            f"{'*' if index == question else ' '}{index:>2}  {f['ttft_ms']:>8.1f} ms  "
            f"{b['ttft_ms']:>8.1f} ms  {speedups[index]:>6.2f}x  {f['question']}"
        )
    f, b = pairs[question]
    speedup = speedups[question]

    width, height, margin, gap, top, bottom = 960, 480, 28, 16, 72, 56
    panel = (width - 2 * margin - gap) // 2
    title = load_font(24, bold=True, mono=False)
    head, strong = load_font(19, bold=True, mono=False), load_font(16, bold=True, mono=False)
    note, clock, body = load_font(13, mono=False), load_font(22, bold=True), load_font(15)
    ground, card, edge = (11, 18, 32), (19, 28, 46), (36, 50, 77)
    text, dim, blue, rose = (226, 232, 240), (124, 138, 165), (56, 189, 248), (248, 113, 113)
    sides = (
        (f"InferScale (k={fast['top_k']}, w={fast['context_window']})", blue, f, stream(f)),
        (f"Mem0 (k={base['top_k']})", rose, b, stream(b)),
    )
    verdict = "InferScale: first token " + (
        f"{speedup:.1f}x sooner" if speedup >= 1 else f"{1 / speedup:.1f}x later"
    )
    slowed = f"   playback slowed {args.slowdown:g}x" if args.slowdown != 1 else ""
    end = max(f["total_ms"], b["total_ms"])
    banner = "InferScale vs Mem0"
    # The question shares the title row, shortened to stay clear of the title.
    asked = "LoCoMo question: " + f["question"]
    while note.getlength(asked) > width - 2 * margin - 32 - title.getlength(banner):
        asked = asked[:-2].rstrip() + "…"

    def frame(now: float):
        image = Image.new("RGB", (width, height), ground)
        draw = ImageDraw.Draw(image)
        draw.text((margin, 22), banner, font=title, fill=(248, 250, 252))
        draw.text((width - margin, 32), asked, font=note, fill=dim, anchor="ra")
        for column, (name, accent, row, events) in enumerate(sides):
            left = margin + column * (panel + gap)
            right, floor = left + panel, height - bottom
            # An accent-colored card under the real one leaves a stripe down its left edge.
            draw.rounded_rectangle((left, top, right, floor), radius=8, fill=accent)
            draw.rounded_rectangle((left + 4, top, right, floor), radius=8, fill=card, outline=edge)
            draw.text((left + 22, top + 22), name, font=head, fill=text)
            draw.line((left + 22, top + 70, right - 18, top + 70), fill=edge)
            if now < row["ttft_ms"]:
                # Both sides start together; the dots keep the slower one visibly working.
                dots = "." * (1 + int(now * args.slowdown / 400) % 3)
                start = right - 18 - draw.textlength("prefilling...", font=strong)
                draw.text((start, top + 24), "prefilling" + dots, font=strong, fill=dim)
                continue
            reading = f"{row['ttft_ms']:,.0f} ms"
            draw.text((right - 18, top + 14), reading, font=clock, fill=accent, anchor="ra")
            draw.text((right - 18, top + 44), "first token", font=note, fill=dim, anchor="ra")
            shown = "".join(word for at, word in events if at <= now).strip()
            for index, line in enumerate(wrap(draw, shown, body, panel - 40)[:11]):
                draw.text((left + 22, top + 86 + 22 * index), line, font=body, fill=text)
        foot = height - bottom + 20
        draw.text((margin, foot + 2), f"t = {now:,.0f} ms{slowed}", font=note, fill=dim)
        if now >= max(f["ttft_ms"], b["ttft_ms"]):
            draw.text((width - margin, foot), verdict, font=strong, fill=text, anchor="ra")
        return image

    # Twenty frames per second of playback, fewer when that would exceed 240 frames.
    step = max(50.0, end * args.slowdown / 240)
    count = max(1, math.ceil(end * args.slowdown / step))
    frames = [frame(end * index / count) for index in range(count + 1)]
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=[round(step)] * count + [3000],  # hold the finished race
        loop=0,
    )
    print(
        f"wrote {args.out}: question {question}, first token {f['ttft_ms']:.0f} ms (InferScale) "
        f"vs {b['ttft_ms']:.0f} ms (Mem0)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("measure", help="time one side and write its timeline")
    run.add_argument("--mode", choices=("mem0", "inferscale"), required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--config", type=Path, default=CONFIG, help="memory benchmark JSON")
    run.add_argument("--questions", type=int, default=8, help="timed questions")
    run.set_defaults(func=measure)
    draw = commands.add_parser("gif", help="replay one question of two timelines as a GIF")
    draw.add_argument("inferscale", type=Path)
    draw.add_argument("mem0", type=Path)
    draw.add_argument("--out", type=Path, default=Path("ttft_race.gif"))
    draw.add_argument(
        "--question", type=int, help="question to replay; default: the median speedup"
    )
    draw.add_argument("--slowdown", type=float, default=20.0, help="playback slowdown factor")
    draw.set_defaults(func=render)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
