"""Command line of the TTFT race: measure one side, or replay several timelines as a GIF.

Measure each side on its own, so nothing shares a GPU, then replay one question:

  python -m benchmarks.ttft_race.run measure --mode mem0       --out mem0.json
  python -m benchmarks.ttft_race.run measure --mode lmcache    --out lmcache.json
  python -m benchmarks.ttft_race.run measure --mode inferscale --out inferscale.json
  python -m benchmarks.ttft_race.run gif inferscale.json mem0.json lmcache.json --out ttft_race.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.ttft_race.gif import load_timelines, render_gif
from benchmarks.ttft_race.measure import DEFAULT_CONFIG, SIDES, measure_side


def main() -> None:
    args = _parse_args()
    if args.command == "measure":
        measure_side(args.mode, config_path=args.config, questions=args.questions, out=args.out)
        return
    timelines = load_timelines(args.timelines)
    render_gif(timelines, out=args.out, question=args.question, slowdown=args.slowdown)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("measure", help="time one side and write its timeline")
    measure.add_argument("--mode", choices=tuple(SIDES), required=True)
    measure.add_argument("--out", type=Path, required=True)
    measure.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "memory benchmark JSON whose model and settings every side runs with "
            f"(default: {DEFAULT_CONFIG})"
        ),
    )
    measure.add_argument("--questions", type=int, default=8, help="timed questions")
    gif = commands.add_parser("gif", help="replay one question of several timelines as a GIF")
    gif.add_argument(
        "timelines",
        type=Path,
        nargs="+",
        metavar="TIMELINE",
        help="the InferScale timeline, then one or more baseline timelines, Mem0 first",
    )
    gif.add_argument("--out", type=Path, default=Path("ttft_race.gif"))
    gif.add_argument(
        "--question",
        type=int,
        help="question to replay; default: the median speedup over the first baseline",
    )
    gif.add_argument("--slowdown", type=float, default=20.0, help="playback slowdown factor")
    return parser.parse_args()


if __name__ == "__main__":
    main()
