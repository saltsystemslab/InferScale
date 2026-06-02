from __future__ import annotations

from .config import parse_args
from .runner import run_benchmark


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    summary = run_benchmark(config)
    print(f"wrote results to {config.run_dir}")
    if summary.get("accuracy") is not None:
        print(f"accuracy={summary['accuracy']:.4f} judged={summary['judged_count']}/{summary['question_count']}")
    else:
        print(f"questions={summary['question_count']} judged={summary['judged_count']}")


if __name__ == "__main__":
    main()
