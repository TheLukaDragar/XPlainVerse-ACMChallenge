#!/usr/bin/env python3
"""Unit-test the compressor GRPO reward plugin (BERT + SLE + gate)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compressor_reward as cr  # noqa: E402

ref = ("The ground looks like a flat white sheet, like someone pasted paper over "
       "the park. Real snow has bumps and soft spots, but this is too even.")

completions = [
    "The snow looks like a flat white sheet someone pasted over the park.",  # good, simple
    ref,                                                                      # exact ref
    "This image shows synthetic generation artifacts and inconsistency.",     # jargon (gate)
    "Fake.",                                                                  # degenerate
    "The ground surface appears unnaturally smooth and uniformly white, "
    "resembling a digitally generated snow effect lacking subtle texture "
    "variation across the entire field which is unusual for outdoor scenes.", # long/complex
]
kwargs = {"reference_simple": [ref] * len(completions),
          "label": ["fake"] * len(completions)}

bert = cr.SimpleBertReward()(completions, **kwargs)
sle = cr.SimpleSleReward()(completions, **kwargs)
gate = cr.SimpleGateReward()(completions, **kwargs)

print(f"{'bert':>6} {'sle_n':>6} {'gate':>6} {'0.7b+0.3s':>9}  text")
for c, b, s, g in zip(completions, bert, sle, gate):
    combo = 0.7 * b + 0.3 * s
    print(f"{b:6.3f} {s:6.3f} {g:6.2f} {combo:9.3f}  {c[:60]}")

assert all(0.0 <= b <= 1.0 for b in bert), "bert out of range"
assert all(0.0 <= s <= 1.0 for s in sle), "sle out of range"
assert bert[1] > 0.95, "exact-ref BERT should be ~1.0"
assert gate[2] < 0, "jargon completion should be gated"
assert gate[3] < 0, "degenerate completion should be gated"
print("\nOK: reward plugin sane")
