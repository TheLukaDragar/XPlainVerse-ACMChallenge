"""Submission validation helpers for the final XDD CodaBench task."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Mapping, Sequence


DETECTION_FILE = "detection.jsonl"
COMPLEX_FILE = "complex.jsonl"
SIMPLE_FILE = "simple.jsonl"
SUBMISSION_FILES = (DETECTION_FILE, COMPLEX_FILE, SIMPLE_FILE)


class SubmissionValidationError(ValueError):
    """Raised when a participant submission is malformed."""


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SubmissionValidationError(f"Malformed JSON in {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise SubmissionValidationError(f"Expected JSON object at {path}:{line_number}.")
            yield payload


def read_reference_rows(path: Path) -> list[Dict[str, Any]]:
    return list(iter_jsonl(Path(path)))


def reference_id_sets(reference_rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    all_ids: set[str] = set()
    explanation_ids: set[str] = set()
    for index, row in enumerate(reference_rows, start=1):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise SubmissionValidationError(f"Reference row {index} is missing a non-empty id.")
        if sample_id in all_ids:
            raise SubmissionValidationError(f"Duplicate reference id: {sample_id}")
        all_ids.add(sample_id)
        if row.get("score_explanations") is True:
            explanation_ids.add(sample_id)
    return all_ids, explanation_ids


def _strict_binary_int(value: Any, *, field: str, source: str, sample_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise SubmissionValidationError(f"{source}: id '{sample_id}' has invalid {field}; expected 0 or 1.")
    return value


def _read_zip_members(zip_path: Path) -> dict[str, str]:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            result: dict[str, str] = {}
            for expected in SUBMISSION_FILES:
                matches = [name for name in members if Path(name).name == expected]
                if len(matches) > 1:
                    raise SubmissionValidationError(f"Submission zip contains multiple {expected} files.")
                if matches:
                    raw = archive.read(matches[0])
                    result[expected] = raw.decode("utf-8")
            return result
    except zipfile.BadZipFile as exc:
        raise SubmissionValidationError(f"Malformed submission zip: {zip_path}") from exc
    except UnicodeDecodeError as exc:
        raise SubmissionValidationError(f"Submission files must be UTF-8 encoded: {zip_path}") from exc


def _read_directory_members(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for expected in SUBMISSION_FILES:
        candidates = sorted(path for path in directory.rglob(expected) if path.is_file())
        if len(candidates) > 1:
            raise SubmissionValidationError(f"Submission directory contains multiple {expected} files.")
        if candidates:
            result[expected] = candidates[0].read_text(encoding="utf-8")
    return result


def read_submission_files(submission: Path) -> dict[str, str]:
    """Read known submission files from a zip, a directory, or a CodaBench input dir."""
    submission = Path(submission)
    if submission.is_file():
        if submission.suffix.lower() != ".zip":
            raise SubmissionValidationError(f"Expected a submission zip, got: {submission}")
        return _read_zip_members(submission)

    if not submission.exists():
        raise SubmissionValidationError(f"Submission path does not exist: {submission}")

    zip_candidates = sorted(path for path in submission.rglob("*.zip") if path.is_file())
    if len(zip_candidates) > 1:
        raise SubmissionValidationError(f"Multiple submission zip files found in {submission}.")
    if zip_candidates:
        return _read_zip_members(zip_candidates[0])
    return _read_directory_members(submission)


def parse_jsonl_text(text: str, *, source: str) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise SubmissionValidationError(f"Malformed JSON in {source}:{line_number}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise SubmissionValidationError(f"Expected JSON object in {source}:{line_number}.")
        rows.append(payload)
    return rows


def _require_id(row: Mapping[str, Any], *, source: str, row_number: int) -> str:
    sample_id = row.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise SubmissionValidationError(f"{source}:{row_number} missing non-empty id.")
    return sample_id.strip()


def _id_aliases(sample_id: str) -> tuple[str, ...]:
    normalized = sample_id.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    aliases = [
        sample_id.strip(),
        normalized,
        path.name,
    ]
    if path.suffix:
        aliases.append(str(path.with_suffix("")))
        aliases.append(path.stem)
    else:
        aliases.append(path.stem)

    unique: list[str] = []
    for alias in aliases:
        if alias and alias not in unique:
            unique.append(alias)
    return tuple(unique)


def _build_reference_alias_map(reference_ids: set[str]) -> dict[str, str | None]:
    aliases: dict[str, str | None] = {}
    for canonical_id in reference_ids:
        for alias in _id_aliases(canonical_id):
            previous = aliases.get(alias)
            if previous is None and alias in aliases:
                continue
            if previous is not None and previous != canonical_id:
                aliases[alias] = None
            else:
                aliases[alias] = canonical_id
    return aliases


def _resolve_submission_id(
    sample_id: str,
    *,
    reference_ids: set[str],
    reference_aliases: Mapping[str, str | None],
) -> str | None:
    if sample_id in reference_ids:
        return sample_id

    matches = {
        canonical_id
        for alias in _id_aliases(sample_id)
        for canonical_id in [reference_aliases.get(alias)]
        if canonical_id is not None
    }
    if len(matches) == 1:
        return next(iter(matches))
    return None


def validate_detection_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_ids: set[str],
    source: str = DETECTION_FILE,
) -> dict[str, Dict[str, Any]]:
    rows_by_id: dict[str, Dict[str, Any]] = {}
    reference_aliases = _build_reference_alias_map(reference_ids)
    for row_number, row in enumerate(rows, start=1):
        sample_id = _require_id(row, source=source, row_number=row_number)
        canonical_id = _resolve_submission_id(
            sample_id,
            reference_ids=reference_ids,
            reference_aliases=reference_aliases,
        )
        if canonical_id is None:
            raise SubmissionValidationError(f"{source} contains unknown ids; first unknown id: {sample_id}")
        if canonical_id in rows_by_id:
            raise SubmissionValidationError(f"{source} contains duplicate id: {sample_id}")
        if "pred_label" not in row:
            raise SubmissionValidationError(f"{source}:{row_number} id '{sample_id}' missing pred_label.")
        rows_by_id[canonical_id] = {
            "id": canonical_id,
            "pred_label": _strict_binary_int(
                row["pred_label"],
                field="pred_label",
                source=source,
                sample_id=sample_id,
            ),
        }
    return rows_by_id


def validate_explanation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_ids: set[str],
    required_ids: set[str],
    field: str,
    source: str,
) -> dict[str, Dict[str, Any]]:
    rows_by_id: dict[str, Dict[str, Any]] = {}
    reference_aliases = _build_reference_alias_map(reference_ids)
    for row_number, row in enumerate(rows, start=1):
        sample_id = _require_id(row, source=source, row_number=row_number)
        canonical_id = _resolve_submission_id(
            sample_id,
            reference_ids=reference_ids,
            reference_aliases=reference_aliases,
        )
        if canonical_id is None:
            raise SubmissionValidationError(f"{source} contains unknown ids; first unknown id: {sample_id}")
        if canonical_id in rows_by_id:
            raise SubmissionValidationError(f"{source} contains duplicate id: {sample_id}")
        if field not in row:
            raise SubmissionValidationError(f"{source}:{row_number} id '{sample_id}' missing {field}.")
        explanation = row[field]
        if not isinstance(explanation, str):
            raise SubmissionValidationError(f"{source}:{row_number} id '{sample_id}' has invalid {field}.")
        if not explanation.strip():
            continue
        rows_by_id[canonical_id] = {"id": canonical_id, field: explanation}
    return rows_by_id


def validate_submission(
    submission: Path,
    reference_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Dict[str, Any]]]:
    """Validate a submission and return per-file rows indexed by id.

    Missing files and missing rows produce empty dictionaries or partial
    dictionaries. Malformed rows, duplicate canonical IDs, and IDs that cannot
    be resolved to the hidden reference fail validation.
    """
    reference_ids, explanation_ids = reference_id_sets(reference_rows)
    files = read_submission_files(Path(submission))

    validated: dict[str, dict[str, Dict[str, Any]]] = {
        "detection": {},
        "complex": {},
        "simple": {},
    }

    if DETECTION_FILE in files:
        rows = parse_jsonl_text(files[DETECTION_FILE], source=DETECTION_FILE)
        validated["detection"] = validate_detection_rows(rows, reference_ids=reference_ids)

    if COMPLEX_FILE in files:
        rows = parse_jsonl_text(files[COMPLEX_FILE], source=COMPLEX_FILE)
        validated["complex"] = validate_explanation_rows(
            rows,
            reference_ids=reference_ids,
            required_ids=explanation_ids,
            field="complex_explanation",
            source=COMPLEX_FILE,
        )

    if SIMPLE_FILE in files:
        rows = parse_jsonl_text(files[SIMPLE_FILE], source=SIMPLE_FILE)
        validated["simple"] = validate_explanation_rows(
            rows,
            reference_ids=reference_ids,
            required_ids=explanation_ids,
            field="simple_explanation",
            source=SIMPLE_FILE,
        )

    return validated
