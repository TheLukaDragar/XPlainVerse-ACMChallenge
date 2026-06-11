#!/usr/bin/env python3
"""Build Pass-1 manifests for external Tier-1 datasets.

Outputs under manifests/external/:
  manifest_openfake_core_0-14.parquet
  manifest_dfbench_train.parquet
  manifest_genimage_train.parquet   (only if extracted tree exists)
  manifest_external_mix_v1.parquet  (optional combined mix)

  manifest_openfake_jpeg_0-14.parquet  (JPEG paths on /primoz after extract)
  manifest_all_v1.parquet              (XP pooled + OpenFake JPEG + DFBench train)

Run full prep (extract + manifests):
  ./scripts/prepare_external_training_lj.sh

Usage (login node — OpenFake only):
  python3 build_manifest_external.py --only openfake

Usage (primoz — DFBench / GenImage):
  ./scripts/lj_cpu_primoz_exec.sh python3 research/experiments/02_pass1_classifier/build_manifest_external.py --only dfbench
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

LABEL2INT = {"real": 0, "fake": 1}

OPENFAKE_ROOT = Path(os.environ.get("OPENFAKE_ROOT", "/home/jakob/luka/data/external/OpenFake"))
OPENFAKE_JPEG_ROOT = Path(os.environ.get("OPENFAKE_JPEG_ROOT", "/primoz/luka/external/OpenFake_jpeg"))
DFBENCH_ROOT = Path(os.environ.get("DFBENCH_ROOT", "/primoz/luka/external/DFBench"))
GENIMAGE_ROOT = Path(os.environ.get("GENIMAGE_ROOT", "/primoz/luka/external/GenImage"))


def repo_root() -> Path:
    env = os.environ.get("CODE_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def out_dir() -> Path:
    d = Path(os.environ.get("MANIFEST_DIR", repo_root() / "research/experiments/02_pass1_classifier/manifests/external"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_dfbench_label(row: dict) -> str | None:
    for turn in row.get("conversations", []):
        if turn.get("from") != "gpt":
            continue
        val = turn.get("value")
        if isinstance(val, list):
            for part in val:
                if part.get("role") == "assistant":
                    tok = str(part.get("content", "")).strip().upper()
                    if tok == "A":
                        return "real"
                    if tok == "B":
                        return "fake"
        elif isinstance(val, str):
            tok = val.strip().upper()
            if tok == "A":
                return "real"
            if tok == "B":
                return "fake"
    return None


def openfake_shards(max_group: int) -> list[Path]:
    core = OPENFAKE_ROOT / "core"
    if not core.is_dir():
        raise FileNotFoundError(f"OpenFake core dir missing: {core}")
    shards: list[Path] = []
    for p in sorted(core.glob("train-*-of-*.parquet")):
        m = re.match(r"train-(\d+)-of-", p.name)
        if not m:
            continue
        if int(m.group(1)) > max_group:
            continue
        if p.stat().st_size > 1024:
            shards.append(p)
    return shards


def build_openfake(max_group: int = 14) -> pd.DataFrame:
    shards = openfake_shards(max_group)
    if not shards:
        raise RuntimeError(f"no OpenFake shards found under {OPENFAKE_ROOT}/core (groups 0-{max_group})")

    rows: list[dict] = []
    for shard in shards:
        table = pq.read_table(shard, columns=["label", "model"])
        labels = table.column("label").to_pylist()
        models = table.column("model").to_pylist()
        rel = shard.relative_to(OPENFAKE_ROOT)
        for i, (lab, model) in enumerate(zip(labels, models)):
            label = str(lab).strip().lower()
            if label not in LABEL2INT:
                continue
            rows.append({
                "sample_id": f"openfake_{rel.as_posix().replace('/', '_')}_{i}",
                "image_path": f"openfake://{rel.as_posix()}#{i}",
                "label": label,
                "label_int": LABEL2INT[label],
                "source": "openfake",
                "generator": str(model) if model is not None else "",
            })

    df = pd.DataFrame(rows)
    print(
        f"openfake groups 0-{max_group}: {len(shards)} shards, {len(df)} rows; "
        f"class counts: {df.label.value_counts().to_dict()}"
    )
    return df


def build_openfake_jpeg(max_group: int = 14, require_exists: bool = False) -> pd.DataFrame:
    shards = openfake_shards(max_group)
    if not shards:
        raise RuntimeError(f"no OpenFake shards found under {OPENFAKE_ROOT}/core (groups 0-{max_group})")

    rows: list[dict] = []
    missing = 0
    for shard in shards:
        table = pq.read_table(shard, columns=["label", "model"])
        labels = table.column("label").to_pylist()
        models = table.column("model").to_pylist()
        rel = shard.relative_to(OPENFAKE_ROOT)
        jpeg_dir = OPENFAKE_JPEG_ROOT / rel.parent / rel.stem
        for i, (lab, model) in enumerate(zip(labels, models)):
            label = str(lab).strip().lower()
            if label not in LABEL2INT:
                continue
            image_path = str(jpeg_dir / f"{i:06d}.jpg")
            exists = Path(image_path).is_file() and Path(image_path).stat().st_size > 1024
            if not exists:
                missing += 1
                if require_exists:
                    continue
            rows.append({
                "sample_id": f"openfake_{rel.as_posix().replace('/', '_')}_{i}",
                "image_path": image_path,
                "label": label,
                "label_int": LABEL2INT[label],
                "source": "openfake",
                "generator": str(model) if model is not None else "",
                "file_exists": exists,
            })

    df = pd.DataFrame(rows)
    print(
        f"openfake_jpeg groups 0-{max_group}: {len(df)} rows; "
        f"class counts: {df.label.value_counts().to_dict()}; files_missing={missing}"
    )
    return df


def build_dfbench(split: str = "train", require_exists: bool = False) -> pd.DataFrame:
    jsonl = DFBENCH_ROOT / ("img_train.jsonl" if split == "train" else "img_test.jsonl")
    if not jsonl.is_file():
        raise FileNotFoundError(jsonl)

    rows: list[dict] = []
    missing = 0
    for row in load_jsonl(jsonl):
        label = parse_dfbench_label(row)
        rel = row.get("image", "")
        if label is None or not rel:
            continue
        # jsonl paths look like DFBench/Kolors/000002.jpg
        image_path = str(DFBENCH_ROOT / rel)
        exists = Path(image_path).is_file()
        if not exists:
            missing += 1
            if require_exists:
                continue
        if Path(rel).name.startswith("._"):
            continue
        parts = Path(rel).parts
        generator = parts[1] if len(parts) >= 2 else "unknown"
        sid = f"dfbench_{generator}_{Path(rel).stem}"
        rows.append({
            "sample_id": sid,
            "image_path": image_path,
            "label": label,
            "label_int": LABEL2INT[label],
            "source": "dfbench",
            "generator": generator,
            "file_exists": exists,
        })

    df = pd.DataFrame(rows)
    print(
        f"dfbench {split}: {len(df)} rows; class counts: {df.label.value_counts().to_dict()}; "
        f"files_missing={missing} (unzip required before training)"
    )
    return df


def genimage_extracted_roots() -> list[Path]:
    """Return train/ai and train/nature dirs if any generator is extracted."""
    roots: list[Path] = []
    if not GENIMAGE_ROOT.is_dir():
        return roots
    for gen_dir in sorted(GENIMAGE_ROOT.iterdir()):
        if not gen_dir.is_dir():
            continue
        for sub in ("train/ai", "train/nature"):
            p = gen_dir / sub
            if p.is_dir() and any(p.iterdir()):
                roots.append(p)
    return roots


def build_genimage() -> pd.DataFrame:
    roots = genimage_extracted_roots()
    if not roots:
        print(f"genimage: no extracted train/ai|nature trees under {GENIMAGE_ROOT} — skip (unzip first)")
        return pd.DataFrame()

    rows: list[dict] = []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for folder in roots:
        generator = folder.parent.parent.name
        label = "fake" if folder.name == "ai" else "real"
        for img in folder.rglob("*"):
            if not img.is_file() or img.suffix.lower() not in exts:
                continue
            rows.append({
                "sample_id": f"genimage_{generator}_{label}_{img.stem}",
                "image_path": str(img),
                "label": label,
                "label_int": LABEL2INT[label],
                "source": "genimage",
                "generator": generator,
                "file_exists": True,
            })

    df = pd.DataFrame(rows)
    print(f"genimage: {len(df)} rows; class counts: {df.label.value_counts().to_dict()}")
    return df


def build_mix_v1(
    xplainverse_pooled: Path,
    openfake_cap: int,
    dfbench_fake_cap: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Conservative mix v1: all XPlainVerse pooled + capped external fakes only."""
    rng = random.Random(seed)
    parts: list[pd.DataFrame] = []

    xp = pd.read_parquet(xplainverse_pooled)
    xp = xp.assign(source="xplainverse", generator="xplainverse", file_exists=True)
    parts.append(xp)
    print(f"mix base xplainverse: {len(xp)} rows")

    of_path = out_dir() / "manifest_openfake_core_0-14.parquet"
    if of_path.is_file() and openfake_cap > 0:
        of = pd.read_parquet(of_path)
        of_fake = of[of.label == "fake"]
        n = min(openfake_cap, len(of_fake))
        of_sample = of_fake.sample(n=n, random_state=seed).reset_index(drop=True)
        of_sample = of_sample.assign(file_exists=True)
        parts.append(of_sample)
        print(f"mix + openfake fakes: {n}")

    df_path = out_dir() / "manifest_dfbench_train.parquet"
    if df_path.is_file() and dfbench_fake_cap > 0:
        df = pd.read_parquet(df_path)
        df_fake = df[(df.label == "fake") & (df.get("file_exists", True))]
        n = min(dfbench_fake_cap, len(df_fake))
        if n > 0:
            idx = df_fake.index.tolist()
            rng.shuffle(idx)
            df_sample = df_fake.loc[idx[:n]].reset_index(drop=True)
            parts.append(df_sample)
            print(f"mix + dfbench fakes (existing files only): {len(df_sample)}")

    mix = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"mix_v1 total: {len(mix)} rows; class counts: {mix.label.value_counts().to_dict()}")
    return mix


def _load_openfake_jpeg_manifest(require_exists: bool) -> pd.DataFrame:
    cached = out_dir() / "manifest_openfake_jpeg_0-14.parquet"
    if cached.is_file() and require_exists:
        df = pd.read_parquet(cached)
        if "file_exists" in df.columns:
            df = df[df.file_exists].copy()
        return df
    return build_openfake_jpeg(require_exists=require_exists)


def _load_dfbench_manifest(require_exists: bool) -> pd.DataFrame:
    cached = out_dir() / "manifest_dfbench_train.parquet"
    jsonl = DFBENCH_ROOT / "img_train.jsonl"
    if jsonl.is_file():
        return build_dfbench("train", require_exists=require_exists)
    if cached.is_file():
        df = pd.read_parquet(cached)
        if require_exists and "file_exists" in df.columns:
            df = df[df.file_exists].copy()
        print(f"dfbench: loaded cached manifest ({len(df)} rows)")
        return df
    raise FileNotFoundError(f"dfbench jsonl missing ({jsonl}) and no cached {cached}")


def build_all_v1(xplainverse_pooled: Path, seed: int = 0, require_exists: bool = True) -> pd.DataFrame:
    """Full combined training manifest: XP pooled + OpenFake JPEG 0-14 + DFBench train."""
    parts: list[pd.DataFrame] = []

    xp = pd.read_parquet(xplainverse_pooled)
    xp = xp.assign(source="xplainverse", generator="xplainverse", file_exists=True)
    parts.append(xp)
    print(f"all_v1 base xplainverse: {len(xp)} rows")

    of = _load_openfake_jpeg_manifest(require_exists)
    if len(of):
        parts.append(of.drop(columns=["file_exists"], errors="ignore"))
        print(f"all_v1 + openfake_jpeg: {len(of)}")

    df = _load_dfbench_manifest(require_exists)
    if len(df):
        parts.append(df.drop(columns=["file_exists"], errors="ignore"))
        print(f"all_v1 + dfbench: {len(df)}")

    all_df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"all_v1 total: {len(all_df)} rows; class counts: {all_df.label.value_counts().to_dict()}")
    return all_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build external dataset manifests for Pass-1")
    parser.add_argument(
        "--only",
        choices=["openfake", "openfake_jpeg", "dfbench", "genimage", "mix", "all_v1", "all"],
        default="all_v1",
    )
    parser.add_argument("--require-exists", action="store_true", default=False)
    parser.add_argument("--openfake-max-group", type=int, default=14)
    parser.add_argument("--openfake-cap", type=int, default=150_000)
    parser.add_argument("--dfbench-fake-cap", type=int, default=100_000)
    parser.add_argument(
        "--xplainverse-pooled",
        type=Path,
        default=repo_root() / "research/experiments/02_pass1_classifier/manifests/manifest_trainval_pooled.parquet",
    )
    args = parser.parse_args()

    od = out_dir()
    print(f"manifest out: {od}")

    req = args.require_exists

    if args.only in {"openfake", "all"}:
        df = build_openfake(args.openfake_max_group)
        path = od / "manifest_openfake_core_0-14.parquet"
        df.to_parquet(path)
        print(f"wrote {path}")

    if args.only in {"openfake_jpeg", "all_v1", "all"}:
        df = build_openfake_jpeg(args.openfake_max_group, require_exists=req)
        path = od / "manifest_openfake_jpeg_0-14.parquet"
        df.to_parquet(path)
        print(f"wrote {path}")

    if args.only in {"dfbench", "all"}:
        df = build_dfbench("train", require_exists=req)
        path = od / "manifest_dfbench_train.parquet"
        df.to_parquet(path)
        print(f"wrote {path}")

    if args.only in {"genimage", "all"}:
        df = build_genimage()
        if len(df):
            path = od / "manifest_genimage_train.parquet"
            df.to_parquet(path)
            print(f"wrote {path}")

    if args.only in {"mix", "all"}:
        if not args.xplainverse_pooled.is_file():
            print(f"skip mix: missing {args.xplainverse_pooled}")
        else:
            df = build_mix_v1(args.xplainverse_pooled, args.openfake_cap, args.dfbench_fake_cap)
            path = od / "manifest_external_mix_v1.parquet"
            df.to_parquet(path)
            print(f"wrote {path}")

    if args.only in {"all_v1", "all"}:
        if not args.xplainverse_pooled.is_file():
            raise FileNotFoundError(args.xplainverse_pooled)
        df = build_all_v1(args.xplainverse_pooled, require_exists=req)
        path = od / "manifest_all_v1.parquet"
        df.to_parquet(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
