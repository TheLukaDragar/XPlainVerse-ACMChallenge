#!/usr/bin/env python3
"""CodaBench scoring entry point for the XDD final evaluation."""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from metrics.bertscore_metric import compute_bertscore_f1
from metrics.detection import ZERO_DETECTION_METRICS, compute_detection_metrics
from metrics.sle_metric import compute_sle_raw, normalize_sle
from metrics.validation import read_reference_rows, validate_submission


DEFAULT_REFERENCE_PATH = Path("/app/data/xdd/references/final_reference.jsonl")
SCORE_KEYS = (
    "detection_macro_f1",
    "detection_accuracy",
    "detection_fake_f1",
    "detection_real_f1",
    "complex_bert_f1",
    "simple_bert_f1",
    "simple_sle_raw",
    "simple_sle_norm",
    "simple_overall_score",
    "explanation_score",
)
ZERO_SCORES = {key: 0.0 for key in SCORE_KEYS}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_html(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[xdd-scoring] {message}", file=sys.stderr)


def load_config(input_dir: Path) -> Dict[str, Any]:
    ref_dir = input_dir / "ref"
    config_path = ref_dir / "config.json"
    if not config_path.exists():
        candidates = sorted(path for path in ref_dir.rglob("config.json") if path.is_file())
        if not candidates:
            raise FileNotFoundError(f"Reference config not found at {config_path}.")
        if len(candidates) > 1:
            raise ValueError(f"Multiple reference config.json files found under {ref_dir}.")
        config_path = candidates[0]

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Reference config must be a JSON object: {config_path}")
    return dict(config)


def reference_path_from_config(config: Mapping[str, Any]) -> Path:
    raw_path = config.get("reference_path", str(DEFAULT_REFERENCE_PATH))
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Reference config must contain a non-empty reference_path string.")
    return Path(raw_path)


def locate_submission(input_dir: Path) -> Path:
    res_dir = input_dir / "res"
    if not res_dir.exists():
        raise FileNotFoundError(f"Submission directory not found: {res_dir}")

    zip_candidates = sorted(path for path in res_dir.rglob("*.zip") if path.is_file())
    if len(zip_candidates) > 1:
        raise ValueError(f"Multiple submission zip files found under {res_dir}.")
    if zip_candidates:
        return zip_candidates[0]
    return res_dir


def cuda_device_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def bert_model_type(config: Mapping[str, Any]) -> str:
    return str(config.get("bert_model", config.get("bertscore_model_type", "microsoft/deberta-xlarge-mnli")))


def bert_batch_size(config: Mapping[str, Any]) -> int:
    return int(config.get("bert_batch_size", config.get("bertscore_batch_size", 16)))


def bert_rescale_with_baseline(config: Mapping[str, Any]) -> bool:
    return bool(config.get("bertscore_rescale_with_baseline", config.get("rescale_with_baseline", False)))


def simple_weights(config: Mapping[str, Any]) -> tuple[float, float]:
    return (
        float(config.get("simple_bert_weight", 0.7)),
        float(config.get("simple_sle_weight", 0.3)),
    )


def explanation_rows(reference_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in reference_rows if row.get("score_explanations") is True]


def explanation_pairs(
    rows: Sequence[Mapping[str, Any]],
    predictions_by_id: Mapping[str, Mapping[str, Any]],
    *,
    prediction_field: str,
    reference_field: str,
) -> tuple[list[str], list[str]]:
    candidates: list[str] = []
    references: list[str] = []
    for row in rows:
        sample_id = str(row["id"])
        pred_row = predictions_by_id.get(sample_id)
        if pred_row is None:
            continue
        candidates.append(str(pred_row[prediction_field]))
        references.append(str(row[reference_field]))
    return candidates, references


def explanation_coverage(submitted_count: int, required_count: int) -> float:
    if required_count <= 0:
        return 1.0
    return float(max(0, min(submitted_count, required_count)) / required_count)


def denormalize_sle(norm: float, *, sle_min: float, sle_max: float) -> float:
    return float(sle_min + norm * (sle_max - sle_min))


def apply_explanation_coverage(
    scores: Dict[str, float],
    *,
    complex_count: int,
    simple_count: int,
    required_count: int,
    config: Mapping[str, Any],
) -> None:
    """Count missing explanation rows as zero contribution to averaged scores."""
    complex_factor = explanation_coverage(complex_count, required_count)
    simple_factor = explanation_coverage(simple_count, required_count)

    scores["complex_bert_f1"] = float(scores.get("complex_bert_f1", 0.0) * complex_factor)
    scores["simple_bert_f1"] = float(scores.get("simple_bert_f1", 0.0) * simple_factor)
    scores["simple_sle_norm"] = float(scores.get("simple_sle_norm", 0.0) * simple_factor)

    if simple_count > 0:
        sle_min = float(config.get("sle_min", -1.0))
        sle_max = float(config.get("sle_max", 4.0))
        scores["simple_sle_raw"] = denormalize_sle(
            scores["simple_sle_norm"],
            sle_min=sle_min,
            sle_max=sle_max,
        )

    bert_weight, sle_weight = simple_weights(config)
    scores["simple_overall_score"] = float(
        bert_weight * scores.get("simple_bert_f1", 0.0)
        + sle_weight * scores.get("simple_sle_norm", 0.0)
    )
    scores["explanation_score"] = float(
        0.5 * (scores.get("complex_bert_f1", 0.0) + scores.get("simple_overall_score", 0.0))
    )


def compute_complex_bert(
    candidates: Sequence[str],
    references: Sequence[str],
    *,
    config: Mapping[str, Any],
    device: str,
    mock_bert: bool,
) -> Dict[str, float]:
    if not candidates:
        return {"complex_bert_f1": 0.0}
    value = compute_bertscore_f1(
        candidates,
        references,
        model_type=bert_model_type(config),
        device=device,
        batch_size=bert_batch_size(config),
        rescale_with_baseline=bert_rescale_with_baseline(config),
        mock=mock_bert,
    )
    return {"complex_bert_f1": float(value)}


def compute_simple_metrics(
    candidates: Sequence[str],
    references: Sequence[str],
    *,
    config: Mapping[str, Any],
    device: str,
    mock_bert: bool,
    mock_sle: bool,
) -> Dict[str, float]:
    if not candidates:
        return {
            "simple_bert_f1": 0.0,
            "simple_sle_raw": 0.0,
            "simple_sle_norm": 0.0,
            "simple_overall_score": 0.0,
        }

    simple_bert = compute_bertscore_f1(
        candidates,
        references,
        model_type=bert_model_type(config),
        device=device,
        batch_size=bert_batch_size(config),
        rescale_with_baseline=bert_rescale_with_baseline(config),
        mock=mock_bert,
    )
    simple_sle_raw = compute_sle_raw(
        candidates,
        references=references,
        device=device,
        config=config,
        mock=mock_sle,
    )
    simple_sle_norm = normalize_sle(
        simple_sle_raw,
        sle_min=float(config.get("sle_min", -1.0)),
        sle_max=float(config.get("sle_max", 4.0)),
    )
    bert_weight, sle_weight = simple_weights(config)
    return {
        "simple_bert_f1": float(simple_bert),
        "simple_sle_raw": float(simple_sle_raw),
        "simple_sle_norm": float(simple_sle_norm),
        "simple_overall_score": float(bert_weight * simple_bert + sle_weight * simple_sle_norm),
    }


def metric_worker_payload(
    candidates: Sequence[str],
    references: Sequence[str],
    *,
    config: Mapping[str, Any],
    mock_bert: bool,
    mock_sle: bool,
) -> Dict[str, Any]:
    return {
        "candidates": list(candidates),
        "references": list(references),
        "config": dict(config),
        "mock_bert": bool(mock_bert),
        "mock_sle": bool(mock_sle),
        "device": "cuda:0",
    }


def start_metric_worker(task: str, payload: Mapping[str, Any], *, visible_device: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_device
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", task]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def collect_metric_worker(
    task: str,
    process: subprocess.Popen[str],
    payload: Mapping[str, Any],
) -> Dict[str, float]:
    stdout, stderr = process.communicate(input=json.dumps(payload))
    if process.returncode != 0:
        message = stderr.strip() or stdout.strip() or f"{task} worker exited with code {process.returncode}."
        raise RuntimeError(f"{task} metric worker failed: {message}")
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{task} metric worker returned malformed JSON.") from exc
    return {key: float(value) for key, value in parsed.items()}


def run_metric_worker(task: str, payload: Mapping[str, Any], *, visible_device: str) -> Dict[str, float]:
    process = start_metric_worker(task, payload, visible_device=visible_device)
    return collect_metric_worker(task, process, payload)


def run_metric_workers_parallel(tasks: Sequence[tuple[str, Mapping[str, Any], str]]) -> Dict[str, float]:
    running = [
        (task, payload, start_metric_worker(task, payload, visible_device=visible_device))
        for task, payload, visible_device in tasks
    ]
    results: Dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=len(running)) as executor:
        futures = [
            executor.submit(collect_metric_worker, task, process, payload)
            for task, payload, process in running
        ]
        for future in futures:
            results.update(future.result())
    return results


def run_two_gpu_metrics(
    complex_candidates: Sequence[str],
    complex_references: Sequence[str],
    simple_candidates: Sequence[str],
    simple_references: Sequence[str],
    *,
    config: Mapping[str, Any],
    mock_bert: bool,
    mock_sle: bool,
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    tasks: list[tuple[str, Mapping[str, Any], str]] = []
    if complex_candidates:
        tasks.append(
            (
                "complex",
                metric_worker_payload(
                    complex_candidates,
                    complex_references,
                    config=config,
                    mock_bert=mock_bert,
                    mock_sle=mock_sle,
                ),
                "0",
            )
        )
    else:
        results["complex_bert_f1"] = 0.0

    if simple_candidates:
        tasks.append(
            (
                "simple",
                metric_worker_payload(
                    simple_candidates,
                    simple_references,
                    config=config,
                    mock_bert=mock_bert,
                    mock_sle=mock_sle,
                ),
                "1",
            )
        )
    else:
        results.update(
            {
                "simple_bert_f1": 0.0,
                "simple_sle_raw": 0.0,
                "simple_sle_norm": 0.0,
                "simple_overall_score": 0.0,
            }
        )
    if tasks:
        results.update(run_metric_workers_parallel(tasks))
    return results


def run_worker(task: str) -> int:
    payload = json.load(sys.stdin)
    candidates = payload["candidates"]
    references = payload["references"]
    config = payload["config"]
    device = payload.get("device", "cuda:0")
    mock_bert = bool(payload.get("mock_bert", False))
    mock_sle = bool(payload.get("mock_sle", False))

    with contextlib.redirect_stdout(sys.stderr):
        if task == "complex":
            result = compute_complex_bert(
                candidates,
                references,
                config=config,
                device=device,
                mock_bert=mock_bert,
            )
        elif task == "simple":
            result = compute_simple_metrics(
                candidates,
                references,
                config=config,
                device=device,
                mock_bert=mock_bert,
                mock_sle=mock_sle,
            )
        else:
            raise ValueError(f"Unknown metric worker task: {task}")

    print(json.dumps(result, sort_keys=True))
    return 0


def score_submission(
    input_dir: Path,
    output_dir: Path,
    *,
    mock_bert: bool = False,
    mock_sle: bool = False,
    debug: bool = False,
) -> Dict[str, float]:
    config = load_config(input_dir)
    reference_path = reference_path_from_config(config)
    if not reference_path.exists():
        raise FileNotFoundError(f"Hidden reference file not found: {reference_path}")

    reference_rows = read_reference_rows(reference_path)
    scored_reference_rows = explanation_rows(reference_rows)
    submission = locate_submission(input_dir)
    debug_log(
        debug,
        (
            f"loaded {len(reference_rows)} reference rows; "
            f"{len(scored_reference_rows)} explanation-scored rows; submission={submission}"
        ),
    )

    validated = validate_submission(submission, reference_rows)
    scores: Dict[str, float] = dict(ZERO_SCORES)
    scores.update(compute_detection_metrics(reference_rows, validated["detection"]))

    complex_candidates: list[str] = []
    complex_references: list[str] = []
    if validated["complex"]:
        complex_candidates, complex_references = explanation_pairs(
            scored_reference_rows,
            validated["complex"],
            prediction_field="complex_explanation",
            reference_field="complex_reference",
        )

    simple_candidates: list[str] = []
    simple_references: list[str] = []
    if validated["simple"]:
        simple_candidates, simple_references = explanation_pairs(
            scored_reference_rows,
            validated["simple"],
            prediction_field="simple_explanation",
            reference_field="simple_reference",
        )

    gpu_count = cuda_device_count()
    debug_log(debug, f"torch cuda device count: {gpu_count}")
    if gpu_count >= 2 and (complex_candidates or simple_candidates):
        scores.update(
            run_two_gpu_metrics(
                complex_candidates,
                complex_references,
                simple_candidates,
                simple_references,
                config=config,
                mock_bert=mock_bert,
                mock_sle=mock_sle,
            )
        )
    else:
        device = "cuda:0" if gpu_count >= 1 else "cpu"
        if complex_candidates:
            scores.update(
                compute_complex_bert(
                    complex_candidates,
                    complex_references,
                    config=config,
                    device=device,
                    mock_bert=mock_bert,
                )
            )
        if simple_candidates:
            scores.update(
                compute_simple_metrics(
                    simple_candidates,
                    simple_references,
                    config=config,
                    device=device,
                    mock_bert=mock_bert,
                    mock_sle=mock_sle,
                )
            )

    apply_explanation_coverage(
        scores,
        complex_count=len(complex_candidates),
        simple_count=len(simple_candidates),
        required_count=len(scored_reference_rows),
        config=config,
    )

    details = {
        "reference_rows": len(reference_rows),
        "score_explanations_rows": len(scored_reference_rows),
        "submission_path": str(submission),
        "detection_rows": len(validated["detection"]),
        "complex_rows": len(validated["complex"]),
        "simple_rows": len(validated["simple"]),
        "complex_scored_rows": len(complex_candidates),
        "simple_scored_rows": len(simple_candidates),
        "mock_bert": mock_bert,
        "mock_sle": mock_sle,
        "cuda_device_count": gpu_count,
    }
    write_json(output_dir / "detailed_results.json", details)
    write_html(
        output_dir / "detailed_results.html",
        "<html><body><h1>XDD Final Evaluation</h1>"
        f"<p>Reference rows: {details['reference_rows']}</p>"
        f"<p>Explanation-scored rows: {details['score_explanations_rows']}</p>"
        f"<p>Detection rows submitted: {details['detection_rows']}</p>"
        f"<p>Complex explanation rows submitted: {details['complex_rows']}</p>"
        f"<p>Simple explanation rows submitted: {details['simple_rows']}</p>"
        f"<p>Complex explanation rows scored: {details['complex_scored_rows']}</p>"
        f"<p>Simple explanation rows scored: {details['simple_scored_rows']}</p>"
        "</body></html>\n",
    )
    return {key: float(scores.get(key, 0.0)) for key in SCORE_KEYS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score XDD CodaBench final submissions.")
    parser.add_argument("input_dir", nargs="?", default="/app/input")
    parser.add_argument("output_dir", nargs="?", default="/app/output")
    parser.add_argument("--mock-bert", action="store_true", help="Use deterministic local BERTScore mock.")
    parser.add_argument("--mock-sle", action="store_true", help="Use deterministic local SLE mock.")
    parser.add_argument("--debug", action="store_true", help="Print safe scorer progress to stderr.")
    parser.add_argument("--worker", choices=("complex", "simple"), help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        return run_worker(args.worker)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    try:
        scores = score_submission(
            input_dir,
            output_dir,
            mock_bert=args.mock_bert,
            mock_sle=args.mock_sle,
            debug=args.debug,
        )
    except Exception as exc:
        write_json(output_dir / "scores.json", ZERO_SCORES)
        write_html(
            output_dir / "detailed_results.html",
            "<html><body><h1>XDD Final Evaluation Failed</h1>"
            f"<p>{html.escape(str(exc))}</p>"
            "</body></html>\n",
        )
        print(f"Scoring failed: {exc}", file=sys.stderr)
        return 1

    write_json(output_dir / "scores.json", scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
