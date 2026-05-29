# MS-Swift distributed training on Frida

This note documents the setup used for the XPlainVerse prompt-v2 Qwen3-VL SFT run.

## MS-Swift distributed environment

MS-Swift launches distributed training through torchrun when these environment
variables are present:

| Variable | Meaning |
|---|---|
| `NPROC_PER_NODE` | GPU processes per node, normally one per visible GPU |
| `NNODES` | total number of nodes |
| `NODE_RANK` | this node's rank, from `0` to `NNODES - 1` |
| `MASTER_ADDR` | hostname/IP of rank-0 node |
| `MASTER_PORT` | rendezvous port |

The Frida Slurm wrapper `scripts/sbatch_train_vlm_v2_h100.sbatch` launches one
task per node with `srun`; `scripts/train_vlm_v2_frida.sh` derives the variables
above from `SLURM_NNODES`, `SLURM_PROCID`, and `SLURM_JOB_NODELIST`.

## Current Frida target

The requested production target is one full H100 node:

```text
partition: frida
node:      ixh when available
GPUs:      8 × H100 80GB
CPUs:      128 requested
memory:    1500G requested
time:      7 days
```

Frida currently has only one H100 node in this partition, so this is multi-GPU
single-node training. The scripts are multi-node-capable, but true multi-node
H100 training would require additional H100 nodes or a different GPU type.

## Prompt-v2 SFT configuration

The run starts from the base model, not the cancelled v1 adapter checkpoint:

```text
model: Qwen/Qwen3-VL-8B-Instruct
train: dataset/train_vlm_v2.jsonl
val:   dataset/val_vlm_v2.jsonl#1000
```

Default batch calculation:

```text
per_device_train_batch_size = 4
NPROC_PER_NODE              = 8
gradient_accumulation_steps = 1
NNODES                      = 1
effective batch             = 4 × 8 × 1 × 1 = 32
```

With 363,602 prompt-v2 training rows:

```text
steps per epoch ≈ ceil(363602 / 32) = 11363
```

Fallback if H100 memory is insufficient:

```bash
PER_DEVICE_BS=2 GRAD_ACCUM=2
```

This preserves effective batch 32 while reducing per-rank activation memory.

## Validation policy

The cancelled v1 run spent too much wall time validating every 400 steps on 2000
generated examples. Prompt-v2 defaults reduce that overhead:

```text
VAL_SLICE=1000
EVAL_STEPS=1000
SAVE_STEPS=1000
PREDICT_WITH_GENERATE=true
WANDB_SAMPLE_N=16
```

`PREDICT_WITH_GENERATE=true` still generates validation outputs so the W&B
callback logs 16 fixed image/GT/prediction examples per eval. The full generated
metric slice is reduced to 1000 rows and runs only every 1000 optimizer steps.

## Submit commands

Smoke test (single A100, tiny slices, no generation eval):

```bash
sbatch --job-name=xpv-v2-smoke --gres=gpu:A100:1 --cpus-per-task=16 \
  --mem=80G --time=00:45:00 \
  --export=ALL,NPROC_PER_NODE=1,CUDA_VISIBLE_DEVICES=0,PER_DEVICE_BS=1,GRAD_ACCUM=1,TRAIN_SLICE=32,VAL_SLICE=16,MAX_STEPS=2,EVAL_STEPS=999999,SAVE_STEPS=999999,PREDICT_WITH_GENERATE=false,REPORT_TO=tensorboard,OUTPUT_DIR=/shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/runs/vlm_v2_smoke \
  scripts/sbatch_train_vlm_v2_h100.sbatch
```

Production H100 run:

```bash
sbatch scripts/sbatch_train_vlm_v2_h100.sbatch
```

Useful overrides:

```bash
# More conservative memory, same effective batch.
sbatch --export=ALL,PER_DEVICE_BS=2,GRAD_ACCUM=2 scripts/sbatch_train_vlm_v2_h100.sbatch

# Faster validation if queue time is more valuable than continuous ROUGE.
sbatch --export=ALL,EVAL_STEPS=2000,SAVE_STEPS=1000 scripts/sbatch_train_vlm_v2_h100.sbatch
```
