# Experiment 03 — GRPO on Pass-2 complex explanation

Optimise the **leaderboard complex metric (BERTScore F1)** directly with GRPO,
starting from the v2 SFT LoRA (`ckpt-1655`). SFT plateaued (ROUGE-L flat ~30
across the epoch); GRPO trains against the actual scored metric to push past it.

## Why this reward
The official Qwen entity/facts scorer is too slow to run per rollout (~hours /
1000). The public leaderboard scores complex with **BERTScore**, which is cheap
(one DeBERTa forward), so we reward exactly that, plus a light verdict/format
term. GRPO's group-relative advantage handles BERTScore's low variance — the
signal is the spread among the N rollouts of one image.

| File | Purpose |
|------|---------|
| `grpo_reward.py` | ms-swift `ORM` reward funcs: `xpv_complex_bert` (BERTScore-F1, leaderboard config) + `xpv_verdict_format` |
| `build_grpo_jsonl.py` | v2 SFT jsonl → prompt-only rows + `reference_complex`/`solution` columns |
| `../../../scripts/train_vlm_v2_grpo_lj.sh` | 4×A100 GRPO launcher (continues from SFT LoRA) |

## Run
```bash
# 1) build GRPO dataset (once)
./scripts/lj_ghcr_image_exec.sh bash -c \
  'python3 research/experiments/03_grpo/build_grpo_jsonl.py \
     --in dataset/train_vlm_v2.jsonl --out dataset/train_grpo.jsonl'

# 2) GRPO (continues from ckpt-1655)
LJ_GPU_GRES=gpu:4 LJ_GPU_TIME=24:00:00 \
  ./scripts/lj_ghcr_image_exec.sh bash scripts/train_vlm_v2_grpo_lj.sh
```

## Reward
`reward = 1.0·BERTScore_F1(candidate_complex, reference_complex) + 0.3·verdict_format`
- BERTScore: `microsoft/deberta-xlarge-mnli`, `lang=en`, no rescale — identical
  to `evaluation/evaluate_val.py`, so the reward IS the leaderboard metric.
- `verdict_format`: +0.5 well-formed `Verdict:` line, +0.5 if it matches the gold label.

## Notes
- The lj image has no vLLM, so rollouts use HF generate (`USE_VLLM=false`) —
  correct but slow; keep `NUM_GENERATIONS`/batch modest. Add vLLM for speed.
- KL `--beta` anchors to the SFT model to prevent drift / extra hallucination.
- ViT stays frozen (the unfreeze experiment showed no gain).
- Validate periodically with the full scorer on a small slice — never per rollout.
