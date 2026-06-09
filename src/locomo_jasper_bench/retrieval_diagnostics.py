from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import ConversationSample, QuestionAnswer, Turn, load_locomo
from .results import write_json
from .vector_types import SearchHit


DIA_ID_RE = re.compile(r"\bD(\d+):(\d+)\b", re.IGNORECASE)


@dataclass(slots=True)
class SampleLookup:
    sample: ConversationSample
    qa_by_id: dict[str, QuestionAnswer]
    dia_by_turn_id: dict[str, str]
    dia_by_session_turn: dict[tuple[str, int], str]


def build_exact_vector_diagnostics(
    *,
    retrieved_hits: list[SearchHit],
    candidate_hits: list[SearchHit],
    exact_hits: list[SearchHit],
    requested_top_k: int,
    diagnostic_k: int,
) -> dict[str, Any]:
    requested_hits = retrieved_hits[:requested_top_k]
    exact_requested = exact_hits[:requested_top_k]
    exact_diagnostic = exact_hits[:diagnostic_k]
    candidates = candidate_hits[:diagnostic_k]

    retrieved_ids = [hit.id for hit in requested_hits]
    candidate_ids = [hit.id for hit in candidates]
    exact_requested_ids = [hit.id for hit in exact_requested]
    exact_diagnostic_ids = [hit.id for hit in exact_diagnostic]
    exact_rank_by_id = {hit.id: hit.rank for hit in exact_diagnostic}

    missing_from_retrieved = [hit for hit in exact_requested if hit.id not in set(retrieved_ids)]
    found_below_requested = [
        hit for hit in missing_from_retrieved if hit.id in set(candidate_ids[requested_top_k:])
    ]
    missing_from_candidates = [hit for hit in exact_requested if hit.id not in set(candidate_ids)]

    retrieved_rank_rows = []
    for hit in requested_hits:
        row = _compact_search_hit(hit)
        row["exact_rank"] = exact_rank_by_id.get(hit.id)
        retrieved_rank_rows.append(row)

    return {
        "enabled": True,
        "requested_top_k": requested_top_k,
        "diagnostic_k": diagnostic_k,
        "exact_recall_at_requested_top_k": _overlap_ratio(retrieved_ids, exact_requested_ids),
        "jasper_candidate_recall_at_diagnostic_k": _overlap_ratio(candidate_ids, exact_diagnostic_ids),
        "exact_top_k_missing_from_retrieved_top_k": [_compact_search_hit(hit) for hit in missing_from_retrieved],
        "exact_top_k_found_below_retrieved_top_k": [_compact_search_hit(hit) for hit in found_below_requested],
        "exact_top_k_missing_from_jasper_candidates": [_compact_search_hit(hit) for hit in missing_from_candidates],
        "retrieved_top_k_exact_ranks": retrieved_rank_rows,
        "jasper_candidates": [_compact_search_hit(hit) for hit in candidates],
        "exact_neighbors": [_compact_search_hit(hit) for hit in exact_diagnostic],
    }


def diagnose_runs(
    *,
    dataset_path: str | Path,
    jasper_run: str | Path,
    qdrant_run: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_k: int | None = None,
    examples: int = 20,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    jasper_run = Path(jasper_run)
    qdrant_run = Path(qdrant_run) if qdrant_run is not None else None
    output_path = Path(output_dir) if output_dir is not None else jasper_run / "retrieval_diagnostics"
    output_path.mkdir(parents=True, exist_ok=True)

    lookups = _build_sample_lookups(load_locomo(dataset_path))
    jasper_rows = _diagnose_prediction_rows(
        run_label="jasper",
        rows=_read_jsonl(_predictions_path(jasper_run)),
        lookups=lookups,
        top_k=top_k,
    )
    _write_jsonl(output_path / "jasper_retrieval.jsonl", jasper_rows)

    run_summaries: dict[str, Any] = {
        "jasper": _summarize_retrieval_rows(jasper_rows),
    }
    comparison = None

    if qdrant_run is not None:
        qdrant_rows = _diagnose_prediction_rows(
            run_label="qdrant",
            rows=_read_jsonl(_predictions_path(qdrant_run)),
            lookups=lookups,
            top_k=top_k,
        )
        _write_jsonl(output_path / "qdrant_retrieval.jsonl", qdrant_rows)
        run_summaries["qdrant"] = _summarize_retrieval_rows(qdrant_rows)
        comparison = _compare_retrieval_rows(jasper_rows, qdrant_rows, examples=examples)

    summary: dict[str, Any] = {
        "dataset": str(dataset_path),
        "top_k": top_k,
        "runs": run_summaries,
        "comparison": comparison,
        "outputs": {
            "jasper_jsonl": str(output_path / "jasper_retrieval.jsonl"),
            "qdrant_jsonl": str(output_path / "qdrant_retrieval.jsonl") if qdrant_run is not None else None,
            "summary_json": str(output_path / "summary.json"),
        },
    }
    write_json(output_path / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="locomo-retrieval-diagnostics",
        description="Compare benchmark retrieval outputs against LoCoMo evidence dia_ids.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--jasper-run", required=True, type=Path, help="Jasper run directory or predictions.jsonl path.")
    parser.add_argument("--qdrant-run", type=Path, help="Optional Qdrant run directory or predictions.jsonl path.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, help="Only score the first K retrieved memories from each record.")
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args(argv)

    summary = diagnose_runs(
        dataset_path=args.dataset,
        jasper_run=args.jasper_run,
        qdrant_run=args.qdrant_run,
        output_dir=args.output_dir,
        top_k=args.top_k,
        examples=args.examples,
    )
    print(f"wrote retrieval diagnostics to {summary['outputs']['summary_json']}")
    jasper = summary["runs"]["jasper"]
    print(
        "jasper "
        f"any_evidence_hit_rate={_format_optional_float(jasper['any_evidence_hit_rate'])} "
        f"evidence_item_recall={_format_optional_float(jasper['evidence_item_recall'])} "
        f"mrr={_format_optional_float(jasper['mrr'])}"
    )
    qdrant = summary["runs"].get("qdrant")
    if qdrant is not None:
        print(
            "qdrant "
            f"any_evidence_hit_rate={_format_optional_float(qdrant['any_evidence_hit_rate'])} "
            f"evidence_item_recall={_format_optional_float(qdrant['evidence_item_recall'])} "
            f"mrr={_format_optional_float(qdrant['mrr'])}"
        )


def _build_sample_lookups(samples: Iterable[ConversationSample]) -> dict[str, SampleLookup]:
    lookups: dict[str, SampleLookup] = {}
    for sample in samples:
        dia_by_turn_id: dict[str, str] = {}
        dia_by_session_turn: dict[tuple[str, int], str] = {}
        for turn in sample.turns:
            dia_id = _turn_dia_id(turn)
            if dia_id is None:
                continue
            dia_by_turn_id[turn.id] = dia_id
            dia_by_session_turn[(turn.session_id, turn.turn_index)] = dia_id
        lookups[sample.sample_id] = SampleLookup(
            sample=sample,
            qa_by_id={qa.question_id: qa for qa in sample.qa},
            dia_by_turn_id=dia_by_turn_id,
            dia_by_session_turn=dia_by_session_turn,
        )
    return lookups


def _diagnose_prediction_rows(
    *,
    run_label: str,
    rows: Iterable[dict[str, Any]],
    lookups: dict[str, SampleLookup],
    top_k: int | None,
) -> list[dict[str, Any]]:
    diagnosed = []
    for row in rows:
        diagnosed.append(_diagnose_prediction_row(run_label, row, lookups, top_k))
    return diagnosed


def _diagnose_prediction_row(
    run_label: str,
    row: dict[str, Any],
    lookups: dict[str, SampleLookup],
    top_k: int | None,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "")
    question_id = str(row.get("question_id") or "")
    lookup = lookups.get(sample_id)
    qa = lookup.qa_by_id.get(question_id) if lookup is not None else None

    gold_dia_ids = _extract_dia_ids(row.get("evidence"))
    if not gold_dia_ids and qa is not None:
        gold_dia_ids = _extract_dia_ids(qa.evidence)
    gold_set = set(gold_dia_ids)

    retrieved = _retrieved_dia_rows(row, lookup, top_k)
    retrieved_dia_ids = [item["dia_id"] for item in retrieved if item.get("dia_id")]
    retrieved_set = set(retrieved_dia_ids)
    hit_ranks = [
        int(item["rank"])
        for item in retrieved
        if item.get("dia_id") in gold_set and item.get("rank") is not None
    ]
    first_rank = min(hit_ranks) if hit_ranks else None
    gold_hit_count = len(gold_set & retrieved_set)
    judge = row.get("judge")
    correct = judge.get("correct") if isinstance(judge, dict) else None

    return {
        "run": run_label,
        "run_id": row.get("run_id"),
        "sample_id": sample_id,
        "question_id": question_id,
        "category": row.get("category"),
        "question": row.get("question"),
        "judge_correct": correct if isinstance(correct, bool) else None,
        "gold_dia_ids": gold_dia_ids,
        "gold_count": len(gold_dia_ids),
        "retrieved_dia_ids": retrieved_dia_ids,
        "retrieved": retrieved,
        "gold_hit_count": gold_hit_count,
        "any_evidence_hit": bool(gold_set and hit_ranks),
        "all_evidence_hit": bool(gold_set) and gold_hit_count == len(gold_set),
        "evidence_recall": _safe_div(gold_hit_count, len(gold_set)),
        "first_evidence_rank": first_rank,
        "reciprocal_rank": _safe_div(1, first_rank) if first_rank is not None else 0.0,
    }


def _retrieved_dia_rows(
    row: dict[str, Any],
    lookup: SampleLookup | None,
    top_k: int | None,
) -> list[dict[str, Any]]:
    retrieved = row.get("retrieved_memories")
    if not isinstance(retrieved, list):
        return []
    rows = []
    for index, item in enumerate(retrieved, start=1):
        if not isinstance(item, dict):
            continue
        rank = _int_or_none(item.get("rank")) or index
        if top_k is not None and rank > top_k:
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        dia_id = _normalize_dia_id(metadata.get("dia_id") or item.get("dia_id"))
        if dia_id is None and lookup is not None:
            turn_id = metadata.get("turn_id") or item.get("turn_id") or item.get("id")
            if turn_id is not None:
                dia_id = lookup.dia_by_turn_id.get(str(turn_id))
        if dia_id is None and lookup is not None:
            session_id = metadata.get("session_id") or item.get("session_id")
            turn_index = _int_or_none(metadata.get("turn_index") or item.get("turn_index"))
            if session_id is not None and turn_index is not None:
                dia_id = lookup.dia_by_session_turn.get((str(session_id), turn_index))
        rows.append(
            {
                "rank": rank,
                "id": item.get("id"),
                "dia_id": dia_id,
                "turn_id": metadata.get("turn_id") or item.get("turn_id"),
                "score": item.get("score"),
                "distance": item.get("distance"),
                "memory": item.get("memory"),
            }
        )
    return rows


def _summarize_retrieval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_rows = [row for row in rows if row["gold_count"] > 0]
    judged = [row for row in rows if row.get("judge_correct") is not None]
    correct = [row for row in judged if row.get("judge_correct") is True]
    total_gold = sum(int(row["gold_count"]) for row in evidence_rows)
    total_gold_hits = sum(int(row["gold_hit_count"]) for row in evidence_rows)

    return {
        "question_count": len(rows),
        "questions_with_evidence": len(evidence_rows),
        "judged_count": len(judged),
        "accuracy": _safe_div(len(correct), len(judged)),
        "any_evidence_hit_rate": _safe_div(
            sum(1 for row in evidence_rows if row["any_evidence_hit"]),
            len(evidence_rows),
        ),
        "all_evidence_hit_rate": _safe_div(
            sum(1 for row in evidence_rows if row["all_evidence_hit"]),
            len(evidence_rows),
        ),
        "evidence_item_recall": _safe_div(total_gold_hits, total_gold),
        "mrr": _safe_div(sum(float(row["reciprocal_rank"]) for row in evidence_rows), len(evidence_rows)),
        "first_evidence_rank": _rank_summary(
            [row["first_evidence_rank"] for row in evidence_rows if row["first_evidence_rank"] is not None]
        ),
        "by_category": _summarize_groups(evidence_rows, key="category"),
        "by_judge": _summarize_groups(evidence_rows, key="judge_correct"),
    }


def _summarize_groups(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = str(row.get(key))
        groups.setdefault(label, []).append(row)
    return {
        label: {
            "question_count": len(group_rows),
            "any_evidence_hit_rate": _safe_div(
                sum(1 for row in group_rows if row["any_evidence_hit"]),
                len(group_rows),
            ),
            "evidence_item_recall": _safe_div(
                sum(int(row["gold_hit_count"]) for row in group_rows),
                sum(int(row["gold_count"]) for row in group_rows),
            ),
            "mrr": _safe_div(sum(float(row["reciprocal_rank"]) for row in group_rows), len(group_rows)),
        }
        for label, group_rows in sorted(groups.items())
    }


def _compare_retrieval_rows(
    jasper_rows: list[dict[str, Any]],
    qdrant_rows: list[dict[str, Any]],
    *,
    examples: int,
) -> dict[str, Any]:
    jasper_by_key = _rows_by_question_key(jasper_rows)
    qdrant_by_key = _rows_by_question_key(qdrant_rows)
    common_keys = sorted(set(jasper_by_key) & set(qdrant_by_key))
    pairs = [(jasper_by_key[key], qdrant_by_key[key]) for key in common_keys]

    qdrant_hit_jasper_miss = [pair for pair in pairs if pair[1]["any_evidence_hit"] and not pair[0]["any_evidence_hit"]]
    jasper_hit_qdrant_miss = [pair for pair in pairs if pair[0]["any_evidence_hit"] and not pair[1]["any_evidence_hit"]]
    qdrant_correct_jasper_wrong = [
        pair for pair in pairs if pair[1].get("judge_correct") is True and pair[0].get("judge_correct") is False
    ]
    jasper_correct_qdrant_wrong = [
        pair for pair in pairs if pair[0].get("judge_correct") is True and pair[1].get("judge_correct") is False
    ]

    both_ranked = [
        pair
        for pair in pairs
        if pair[0]["first_evidence_rank"] is not None and pair[1]["first_evidence_rank"] is not None
    ]
    rank_deltas = [
        int(jasper["first_evidence_rank"]) - int(qdrant["first_evidence_rank"])
        for jasper, qdrant in both_ranked
    ]
    overlaps = [_retrieval_jaccard(jasper["retrieved_dia_ids"], qdrant["retrieved_dia_ids"]) for jasper, qdrant in pairs]

    return {
        "common_question_count": len(common_keys),
        "qdrant_hit_jasper_miss_count": len(qdrant_hit_jasper_miss),
        "jasper_hit_qdrant_miss_count": len(jasper_hit_qdrant_miss),
        "qdrant_correct_jasper_wrong_count": len(qdrant_correct_jasper_wrong),
        "jasper_correct_qdrant_wrong_count": len(jasper_correct_qdrant_wrong),
        "first_evidence_rank_delta_jasper_minus_qdrant": _number_summary(rank_deltas),
        "retrieved_dia_id_jaccard": _number_summary(overlaps),
        "examples": {
            "qdrant_hit_jasper_miss": _comparison_examples(qdrant_hit_jasper_miss, examples),
            "jasper_hit_qdrant_miss": _comparison_examples(jasper_hit_qdrant_miss, examples),
            "qdrant_correct_jasper_wrong": _comparison_examples(qdrant_correct_jasper_wrong, examples),
        },
    }


def _comparison_examples(pairs: list[tuple[dict[str, Any], dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    sorted_pairs = sorted(
        pairs,
        key=lambda pair: (
            pair[0]["first_evidence_rank"] is None,
            pair[0]["first_evidence_rank"] or 10**9,
            pair[1]["first_evidence_rank"] is None,
            pair[1]["first_evidence_rank"] or 10**9,
        ),
    )
    examples = []
    for jasper, qdrant in sorted_pairs[:limit]:
        examples.append(
            {
                "sample_id": jasper["sample_id"],
                "question_id": jasper["question_id"],
                "category": jasper["category"],
                "question": jasper["question"],
                "gold_dia_ids": jasper["gold_dia_ids"],
                "jasper": _example_run_fields(jasper),
                "qdrant": _example_run_fields(qdrant),
            }
        )
    return examples


def _example_run_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "judge_correct": row["judge_correct"],
        "any_evidence_hit": row["any_evidence_hit"],
        "first_evidence_rank": row["first_evidence_rank"],
        "retrieved_dia_ids": row["retrieved_dia_ids"],
    }


def _extract_dia_ids(value: Any) -> list[str]:
    found: list[str] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, str):
            for match in DIA_ID_RE.finditer(node):
                found.append(f"D{int(match.group(1))}:{int(match.group(2))}")
            return
        if isinstance(node, dict):
            for nested in node.values():
                visit(nested)
            return
        if isinstance(node, Iterable) and not isinstance(node, (bytes, bytearray)):
            for nested in node:
                visit(nested)

    visit(value)
    return _unique(found)


def _turn_dia_id(turn: Turn) -> str | None:
    dia_id = _normalize_dia_id(turn.dia_id)
    if dia_id is not None:
        return dia_id
    if isinstance(turn.raw, dict):
        return _normalize_dia_id(turn.raw.get("dia_id") or turn.raw.get("dialogue_id"))
    return None


def _normalize_dia_id(value: Any) -> str | None:
    if value is None:
        return None
    match = DIA_ID_RE.search(str(value))
    if match is None:
        return None
    return f"D{int(match.group(1))}:{int(match.group(2))}"


def _compact_search_hit(hit: SearchHit) -> dict[str, Any]:
    payload = hit.payload if isinstance(hit.payload, dict) else {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return {
        "rank": hit.rank,
        "id": hit.id,
        "turn_id": metadata.get("turn_id") or payload.get("turn_id"),
        "dia_id": _normalize_dia_id(metadata.get("dia_id") or payload.get("dia_id")),
        "score": hit.score,
        "distance": hit.distance,
    }


def _predictions_path(path: Path) -> Path:
    if path.is_dir():
        return path / "predictions.jsonl"
    return path


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if isinstance(row, dict):
                yield row


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rows_by_question_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["sample_id"]), str(row["question_id"])): row for row in rows}


def _retrieval_jaccard(left: list[str], right: list[str]) -> float | None:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return None
    return len(left_set & right_set) / len(union)


def _overlap_ratio(left: list[str], right: list[str]) -> float | None:
    denominator = len(set(right))
    if denominator == 0:
        return None
    return len(set(left) & set(right)) / denominator


def _rank_summary(values: list[int]) -> dict[str, Any]:
    return _number_summary([float(value) for value in values])


def _number_summary(values: Iterable[float | int | None]) -> dict[str, Any]:
    numbers = sorted(float(value) for value in values if value is not None)
    if not numbers:
        return {"count": 0, "avg": None, "min": None, "p50": None, "max": None}
    return {
        "count": len(numbers),
        "avg": sum(numbers) / len(numbers),
        "min": numbers[0],
        "p50": numbers[len(numbers) // 2],
        "max": numbers[-1],
    }


def _safe_div(numerator: float | int, denominator: float | int | None) -> float | None:
    if denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
