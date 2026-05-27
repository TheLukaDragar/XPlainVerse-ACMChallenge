# Agent Task: Lj Multi-GPU VLM Training Setup (Apptainer)

## Context

Repository: `/home/jakob/luka/code/XPlainVerse-ACMChallenge`
Container entry: `~/xplainverse_exec.sh` (Apptainer SIF at `~/containers/xplainverse-acmchallenge.sif`)
GPU node: `elixir-lj-gpu-01` — 4× A100 80GB, Torch 2.2.2+cu121 inside container
Data: `/home/jakob/luka/data/XPlainVerse` (bind-mounted via `$HOME`)
Code in container: `/workspace/XPlainVerse-ACMChallenge`

Reference script: `scripts/train_vlm_full.sh` (shared-workspace cluster)
Draft lj script may exist: `scripts/train_vlm_full_lj.sh` — verify and fix if needed.

## Plan (execute in order)

1. **Inspect environment**
   - Read `scripts/train_vlm_full.sh`, `scripts/train_vlm_full_lj.sh`, `~/xplainverse_exec.sh`
   - Inside container: verify `swift`, `flash_attn`, `torch.cuda.device_count()==4`
   - Check dataset files: `dataset/train_vlm.jsonl`, `dataset/val_vlm.jsonl`

2. **Fix lj training script** (`scripts/train_vlm_full_lj.sh`)
   - Paths: code `/workspace/...` or `/home/jakob/luka/code/...`, data `/home/jakob/luka/data/XPlainVerse`, output `/home/jakob/luka/runs/vlm_full`
   - Default 4-GPU: `NPROC_PER_NODE=4`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `PER_DEVICE_BS=2`, `GRAD_ACCUM=4`
   - CUDA lib auto-detect (cu12/cu121, not hard-coded cu13)
   - `ATTN_IMPL` fallback if flash_attn missing
   - Do NOT overwrite `train_vlm_full.sh`

3. **Build JSONL if missing**
   ```bash
   ~/xplainverse_exec.sh bash -lc '
     cd /workspace/XPlainVerse-ACMChallenge
     python3 dataset/build_swift_jsonl.py \
       --data-root /home/jakob/luka/data/XPlainVerse \
       --output-dir dataset
   '
   ```
   Skip if files already exist and are non-empty.

4. **Smoke test** (do NOT start full 450k training)
   - Dry-run: invoke script with env overrides for a tiny run OR verify `swift sft --help` and that train jsonl paths resolve
   - If feasible, run 1-step sanity: `NUM_EPOCHS` tiny + `VAL_SLICE=4` + `SAVE_STEPS=999999` + `EVAL_STEPS=999999` for ~1 minute to confirm multi-GPU launch without OOM
   - If smoke too risky/slow, document why and confirm launch command only

5. **Write runbook**
   - Append short section to `scripts/train_vlm_full_lj.sh` header OR create `scripts/LJ_TRAINING.md` with exact commands
   - Include fallback: `ATTN_IMPL=sdpa PACKING=false REPORT_TO=tensorboard`

## Success criteria (all must pass)

- [ ] `scripts/train_vlm_full_lj.sh` exists, is executable, lj paths correct
- [ ] `dataset/train_vlm.jsonl` and `dataset/val_vlm.jsonl` exist (built or pre-existing)
- [ ] Container check passes: 4 GPUs visible, `swift` on PATH
- [ ] Multi-GPU launch verified (smoke test started OR explicit evidence deepspeed/torchrun sees 4 ranks)
- [ ] Runbook command documented for full training:
  `~/xplainverse_exec.sh bash /workspace/XPlainVerse-ACMChallenge/scripts/train_vlm_full_lj.sh`

## Constraints

- Run commands via `~/xplainverse_exec.sh` for anything needing GPUs/container
- Do not commit unless changes are clearly ready; no git push
- Do not start a multi-hour full training run
- Minimize scope — only files needed for lj multi-GPU setup

## Deliverable

Reply with a structured report:
1. What you changed (files + summary)
2. Verification results (GPU count, swift, jsonl row counts)
3. Exact command to start full 4-GPU training
4. Any blockers or recommended env overrides
