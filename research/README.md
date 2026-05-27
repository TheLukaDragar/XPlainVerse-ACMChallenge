# Research notes — XPlainVerse pipeline design

Working notes that motivate the architectural decisions for our XPlainVerse submission. Every quantitative claim cites the source paper (arXiv ID + venue + year) so we can drop these straight into the ACM-MM writeup.

## Files

| File | Contents |
|------|----------|
| `01_problem_statement.md` | Concrete failure mode of the current VLM (numbers from our own runs) |
| `02_literature_review.md` | Each cited paper with the specific accuracy/AUC numbers we rely on |
| `03_zero_shot_experiment.md` | Today's experiment: 6/7 simplicityprevails baselines on XPlainVerse val |
| `04_strategy.md` | Recommended pipeline + literature support for each design choice |
| `05_open_questions.md` | Things I claimed without direct experimental support — flagged honestly |
| `06_training_data_v2.md` | Research-grounded design for the new training set (`train_vlm_v2.jsonl`) — prompt-mix and fake-GT filter, each cited |
| `07_gt_vocabulary_analysis.md` | Mining of 2000 fake + 2000 real GTs from `train_vlm.jsonl` — opening templates, transitions, distinctive vocabulary, and the v2.1 prompt revisions they motivate |
| `references.bib` | BibTeX for everything |

## TL;DR (numbers only)

- **Current VLM** (Qwen3-VL-8B + LoRA, ckpt-3200, 13% through 1 epoch): val verdict acc **0.720** (real 0.92 / fake 0.55) on 2000-sample val; complex_overall **0.391** on 32-sample official eval (best ckpt-2400)
- **Zero-shot AIGC detectors** on XPlainVerse val (1000 sample, our experiment, 26 May 2026): best AUC **0.726** (PE-CLIP-Linear), best calibrated acc **0.679** — **all 6 worse than the VLM**
- **Published in-distribution AIGC detection accuracy** (target for fine-tuning): DINOv3-Linear 96.5% on GenImage (Simplicity Prevails 2026), SigLIP2+DINOv2 ensemble 99.10% on OpenFake (Bombek1 2025), NTIRE 2026 winners 0.9974 ROC-AUC clean
- **Published lift from decoupled detection→explanation in forensic VLM**: M2F2-Det (CVPR 2025) ablation shows **−10.12% accuracy** when forgery features are removed
