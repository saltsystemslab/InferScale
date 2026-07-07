from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from .clients import build_openai_responses_judge_body, responses_output_text
from .config import BenchmarkConfig
from .data import QuestionAnswer
from .judging import failed_judge_payload, parsed_judge_payload, record_label
from .prompts import build_judge_messages
from .results import JsonlWriter, write_json


OPENAI_BATCH_ENDPOINT = "/v1/responses"
OPENAI_BATCH_COMPLETION_WINDOW = "24h"
OPENAI_BATCH_METADATA_PATH = "openai_judge_batch.json"
OPENAI_BATCH_INPUT_PATH = "openai_judge_batch_input.jsonl"
OPENAI_BATCH_OUTPUT_PATH = "openai_judge_batch_output.jsonl"
OPENAI_BATCH_ERRORS_PATH = "openai_judge_batch_errors.jsonl"

_TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


class OpenAIBatchJudgeError(RuntimeError):
    pass


@dataclass(slots=True)
class OpenAIBatchJudgeRequest:
    row_index: int
    custom_id: str
    request: dict[str, Any]


@dataclass(slots=True)
class OpenAIBatchApplyResult:
    applied_count: int
    error_count: int
    missing_count: int


@dataclass(slots=True)
class OpenAIBatchJudgeResult:
    batch_id: str | None
    submitted_count: int
    applied_count: int
    error_count: int
    missing_count: int


class OpenAIResponsesBatchJudgeClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install openai>=2.44,<3 to use --judge openai.") from exc

            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)
        self._client = client

    def judge_records(
        self,
        config: BenchmarkConfig,
        records: list[dict[str, Any]],
        *,
        poll_interval_s: float | None = None,
        max_wait_s: float | None = None,
    ) -> OpenAIBatchJudgeResult:
        return judge_records_with_openai_batch(
            config,
            records,
            client=self._client,
            poll_interval_s=poll_interval_s,
            max_wait_s=max_wait_s,
        )


def judge_records_with_openai_batch(
    config: BenchmarkConfig,
    records: list[dict[str, Any]],
    *,
    client: Any,
    poll_interval_s: float | None = None,
    max_wait_s: float | None = None,
) -> OpenAIBatchJudgeResult:
    targets = build_openai_batch_judge_requests(config, records)
    if not targets:
        return OpenAIBatchJudgeResult(
            batch_id=None,
            submitted_count=0,
            applied_count=0,
            error_count=0,
            missing_count=0,
        )

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / OPENAI_BATCH_INPUT_PATH
    output_path = run_dir / OPENAI_BATCH_OUTPUT_PATH
    errors_path = run_dir / OPENAI_BATCH_ERRORS_PATH
    metadata_path = run_dir / OPENAI_BATCH_METADATA_PATH

    _write_batch_input(input_path, targets)
    request_signature = _request_signature(targets)
    metadata = _load_matching_metadata(metadata_path, config, targets, request_signature)
    batch = _create_or_reuse_batch(client, config, input_path, metadata, request_signature, targets, metadata_path)
    batch = _poll_batch(
        client,
        batch,
        poll_interval_s=_poll_interval_s(poll_interval_s),
        max_wait_s=_max_wait_s(max_wait_s),
        metadata_path=metadata_path,
        existing_metadata=metadata,
    )

    batch_id = _string_attr(batch, "id")
    status = _string_attr(batch, "status")
    _save_batch_metadata(metadata_path, config, targets, request_signature, batch=batch)

    if status != "completed":
        _mark_batch_status_error(config, records, targets, status)
        raise OpenAIBatchJudgeError(f"OpenAI batch {batch_id or '<unknown>'} ended with status {status!r}.")

    output_lines = _download_batch_file(client, _string_attr(batch, "output_file_id"), output_path)
    error_lines = _download_batch_file(client, _string_attr(batch, "error_file_id"), errors_path)
    apply_result = apply_openai_batch_judge_results(
        config,
        records,
        targets,
        output_lines=output_lines,
        error_lines=error_lines,
    )
    if apply_result.error_count or apply_result.missing_count:
        raise OpenAIBatchJudgeError(
            "OpenAI batch judging completed with "
            f"{apply_result.error_count} request error(s) and {apply_result.missing_count} missing result(s)."
        )
    return OpenAIBatchJudgeResult(
        batch_id=batch_id,
        submitted_count=len(targets),
        applied_count=apply_result.applied_count,
        error_count=apply_result.error_count,
        missing_count=apply_result.missing_count,
    )


def build_openai_batch_judge_requests(
    config: BenchmarkConfig,
    records: list[dict[str, Any]],
) -> list[OpenAIBatchJudgeRequest]:
    requests: list[OpenAIBatchJudgeRequest] = []
    for row_index, record in enumerate(records):
        if not config.rejudge and _is_judged(record):
            continue
        messages = build_judge_messages(
            _qa_from_record(record),
            str(record.get("predicted_answer") or ""),
            structured=True,
        )
        custom_id = _custom_id(row_index, record)
        requests.append(
            OpenAIBatchJudgeRequest(
                row_index=row_index,
                custom_id=custom_id,
                request={
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": OPENAI_BATCH_ENDPOINT,
                    "body": build_openai_responses_judge_body(
                        model=config.judge_model,
                        messages=messages,
                        max_tokens=config.max_judge_tokens,
                    ),
                },
            )
        )
    return requests


def apply_openai_batch_judge_results(
    config: BenchmarkConfig,
    records: list[dict[str, Any]],
    targets: list[OpenAIBatchJudgeRequest],
    *,
    output_lines: Iterable[dict[str, Any]],
    error_lines: Iterable[dict[str, Any]] = (),
) -> OpenAIBatchApplyResult:
    target_by_id = {target.custom_id: target for target in targets}
    seen: set[str] = set()
    applied_count = 0
    error_count = 0

    for line in output_lines:
        custom_id = str(line.get("custom_id") or "")
        target = target_by_id.get(custom_id)
        if target is None:
            continue
        seen.add(custom_id)
        error = line.get("error")
        if error:
            _mark_request_error(config, records[target.row_index], error)
            error_count += 1
            continue
        response = line.get("response")
        if not _successful_response(response):
            _mark_request_error(config, records[target.row_index], response or line)
            error_count += 1
            continue
        body = response.get("body") if isinstance(response, dict) else None
        raw = responses_output_text(body or {})
        records[target.row_index]["judge"] = parsed_judge_payload(config, raw)
        applied_count += 1

    for line in error_lines:
        custom_id = str(line.get("custom_id") or "")
        target = target_by_id.get(custom_id)
        if target is None or custom_id in seen:
            continue
        seen.add(custom_id)
        _mark_request_error(config, records[target.row_index], line.get("error") or line)
        error_count += 1

    missing = [target for target in targets if target.custom_id not in seen]
    for target in missing:
        _mark_request_error(
            config,
            records[target.row_index],
            f"OpenAI batch result missing for {record_label(records[target.row_index])}",
        )
    return OpenAIBatchApplyResult(
        applied_count=applied_count,
        error_count=error_count,
        missing_count=len(missing),
    )


def _create_or_reuse_batch(
    client: Any,
    config: BenchmarkConfig,
    input_path: Path,
    metadata: dict[str, Any] | None,
    request_signature: str,
    targets: list[OpenAIBatchJudgeRequest],
    metadata_path: Path,
) -> Any:
    batch_id = metadata.get("batch_id") if metadata else None
    if isinstance(batch_id, str) and batch_id:
        logger.info("Reusing OpenAI judge batch {}", batch_id)
        return client.batches.retrieve(batch_id)

    logger.info("Uploading {} OpenAI judge request(s) for Batch API", len(targets))
    with input_path.open("rb") as input_file:
        file_obj = client.files.create(file=input_file, purpose="batch")
    input_file_id = _string_attr(file_obj, "id")
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=OPENAI_BATCH_ENDPOINT,
        completion_window=OPENAI_BATCH_COMPLETION_WINDOW,
        metadata={
            "run_id": config.run_id,
            "model": config.judge_model,
            "request_signature": request_signature,
        },
    )
    _save_batch_metadata(
        metadata_path,
        config,
        targets,
        request_signature,
        input_file_id=input_file_id,
        batch=batch,
    )
    return batch


def _poll_batch(
    client: Any,
    batch: Any,
    *,
    poll_interval_s: float,
    max_wait_s: float | None,
    metadata_path: Path,
    existing_metadata: dict[str, Any] | None,
) -> Any:
    started = time.monotonic()
    status = _string_attr(batch, "status")
    batch_id = _string_attr(batch, "id")
    while status not in _TERMINAL_BATCH_STATUSES:
        logger.info("OpenAI judge batch {} status={}", batch_id, status)
        if max_wait_s is not None and time.monotonic() - started >= max_wait_s:
            raise OpenAIBatchJudgeError(
                f"OpenAI batch {batch_id or '<unknown>'} still has status {status!r} after {max_wait_s:.0f}s."
            )
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)
        batch = client.batches.retrieve(batch_id)
        status = _string_attr(batch, "status")
        if existing_metadata:
            updated = dict(existing_metadata)
            updated["batch"] = _jsonable(batch)
            write_json(metadata_path, updated)
    logger.info("OpenAI judge batch {} terminal status={}", batch_id, status)
    return batch


def _download_batch_file(client: Any, file_id: str | None, path: Path) -> list[dict[str, Any]]:
    if not file_id:
        if path.exists():
            path.unlink()
        return []
    response = client.files.content(file_id)
    text = _file_response_text(response)
    path.write_text(text, encoding="utf-8")
    return _jsonl_lines(text, path)


def _write_batch_input(path: Path, targets: list[OpenAIBatchJudgeRequest]) -> None:
    with JsonlWriter(path) as writer:
        for target in targets:
            writer.write(target.request)


def _load_matching_metadata(
    path: Path,
    config: BenchmarkConfig,
    targets: list[OpenAIBatchJudgeRequest],
    request_signature: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    expected_custom_ids = [target.custom_id for target in targets]
    if (
        data.get("model") == config.judge_model
        and data.get("endpoint") == OPENAI_BATCH_ENDPOINT
        and bool(data.get("rejudge", False)) == config.rejudge
        and data.get("request_signature") == request_signature
        and data.get("custom_ids") == expected_custom_ids
    ):
        return data
    return None


def _save_batch_metadata(
    path: Path,
    config: BenchmarkConfig,
    targets: list[OpenAIBatchJudgeRequest],
    request_signature: str,
    *,
    input_file_id: str | None = None,
    batch: Any | None = None,
) -> None:
    batch_id = _string_attr(batch, "id") if batch is not None else None
    batch_input_file_id = _string_attr(batch, "input_file_id") if batch is not None else None
    write_json(
        path,
        {
            "version": 1,
            "provider": config.judge_provider,
            "model": config.judge_model,
            "rejudge": config.rejudge,
            "endpoint": OPENAI_BATCH_ENDPOINT,
            "completion_window": OPENAI_BATCH_COMPLETION_WINDOW,
            "request_count": len(targets),
            "request_signature": request_signature,
            "custom_ids": [target.custom_id for target in targets],
            "input_file_id": input_file_id or batch_input_file_id,
            "batch_id": batch_id,
            "batch": _jsonable(batch),
        },
    )


def _request_signature(targets: list[OpenAIBatchJudgeRequest]) -> str:
    payload = "\n".join(
        json.dumps(target.request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for target in targets
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _custom_id(row_index: int, record: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "row_index": row_index,
            "sample_id": record.get("sample_id"),
            "question_id": record.get("question_id"),
            "predicted_answer": record.get("predicted_answer"),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"judge-row-{row_index + 1:06d}-{digest}"


def _qa_from_record(record: dict[str, Any]) -> QuestionAnswer:
    return QuestionAnswer(
        sample_id=str(record.get("sample_id") or ""),
        question_id=str(record.get("question_id") or ""),
        question=str(record.get("question") or ""),
        answer=str(record.get("gold_answer") or ""),
        category=str(record.get("category") or ""),
        evidence=record.get("evidence"),
    )


def _successful_response(response: Any) -> bool:
    if not isinstance(response, dict):
        return False
    status_code = response.get("status_code")
    if not isinstance(status_code, int):
        return False
    return 200 <= status_code < 300


def _mark_batch_status_error(
    config: BenchmarkConfig,
    records: list[dict[str, Any]],
    targets: list[OpenAIBatchJudgeRequest],
    status: str | None,
) -> None:
    for target in targets:
        _mark_request_error(
            config,
            records[target.row_index],
            f"OpenAI batch ended with status {status!r}",
        )


def _mark_request_error(config: BenchmarkConfig, record: dict[str, Any], error: Any) -> None:
    message = json.dumps(error, sort_keys=True, ensure_ascii=False) if isinstance(error, dict) else str(error)
    record["judge"] = failed_judge_payload(RuntimeError(f"OpenAI batch request failed: {message}"), config)


def _is_judged(record: dict[str, Any]) -> bool:
    judge = record.get("judge")
    return isinstance(judge, dict) and isinstance(judge.get("correct"), bool)


def _jsonl_lines(text: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def _file_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, bytes):
        return response.decode("utf-8")
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):
        value = text()
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    read = getattr(response, "read", None)
    if callable(read):
        value = read()
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if isinstance(content, str):
        return content
    return str(response)


def _string_attr(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get(key)
    else:
        raw = getattr(value, key, None)
    return raw if isinstance(raw, str) and raw else None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except TypeError:
            return _jsonable(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict):
        return _jsonable(fields)
    return str(value)


def _poll_interval_s(value: float | None) -> float:
    if value is not None:
        return max(0.0, value)
    raw = os.environ.get("OPENAI_BATCH_POLL_INTERVAL_SECONDS")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 30.0


def _max_wait_s(value: float | None) -> float | None:
    if value is not None:
        return max(0.0, value)
    raw = os.environ.get("OPENAI_BATCH_MAX_WAIT_SECONDS")
    if raw:
        try:
            parsed = float(raw)
        except ValueError:
            return None
        return max(0.0, parsed) if parsed > 0 else None
    return None
