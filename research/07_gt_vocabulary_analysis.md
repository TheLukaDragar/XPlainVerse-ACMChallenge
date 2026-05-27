# GT vocabulary analysis — what the references actually look like

Reproducible mining of vocabulary, opening templates, transitions, and structural patterns from the XPlainVerse train complex GTs. Source: `dataset/train_vlm.jsonl` (450k rows, complex_explanation embedded in the assistant target). Script: `dataset/analyze_gt_vocab.py` (~1.4 s on 4000 samples).

**Sample:** 2000 fake + 2000 real GTs sampled in file order from `train_vlm.jsonl` (which is already shuffled by `build_swift_jsonl.py` with seed=42).

**Why this matters for v2:** the metric (Entity F1 + Evidence F1 = 70% of complex score) is computed by Qwen3.5-4B coverage matching against the reference. Coverage is *semantic* (paraphrases count), but our model's vocabulary still drifts toward whatever the SFT target distribution emphasizes. We want the training-prompt instructions to nudge the model toward the **same vocabulary the references use**, so the model has the right words available when describing evidence.

---

## 1. FAKE GT patterns

### 1.1 Statistics

| Metric | Value |
|--------|------:|
| Sentences (median / mean) | **4 / 4.2** |
| Words (median / mean) | **113 / 121** |
| Artifact-seed hits per sample | **2.0** (avg) |
| Authenticity-seed hits per sample | 0.13 (avg) |

(Artifact seeds = words like *distorted, warped, smudged, synthetic, unnatural, inconsistent*. Authenticity seeds = *natural, realistic, plausible, sharp, crisp, authentic*. Lists are in `analyze_gt_vocab.py`.)

### 1.2 Top 12 opening 6-grams

```
178  several visual cues make the image                # template A
138  the image exhibits several digital artifacts
135  several visual cues make this image
 95  the image contains several visual cues
 44  several visual cues make the image                # already counted
 43  several visual cues make this image
 38  the generated image looks synthetic due
 36  the image looks synthetic due to
 30  the image exhibits clear signs of
 23  the image contains several physical and
 23  the image exhibits several synthetic qualities
 20  the image exhibits several digital artifacts
 17  several visual cues make the scene
 16  the image contains several unnatural elements
 15  the image exhibits several tell-tale signs
```

Roughly **three template families** dominate:

| Family | Frequency in 2000 | Example |
|--------|------------------:|---------|
| "Several visual cues make this/the image …" | ~400 | *"Several visual cues make this image look synthetic. The most notable is …"* |
| "The image exhibits/contains several …" | ~376 | *"The image exhibits clear signs of synthetic generation, primarily due to …"* |
| "The (generated) image looks synthetic due to …" | ~84 | *"The generated image looks synthetic due to several inconsistencies."* |

### 1.3 Top transitions (first 3 words of sentences 2..N)

```
312  the lighting on
231  additionally the lighting
193  most notably the          ← unique to FAKE
161  in the background
159  there is also
103  the skin texture
 92  additionally there is
 80  additionally the background
 79  around the face
 74  there is a
 74  the most prominent        ← unique to FAKE
 54  the man in
 52  additionally the text
 51  the texture of
 50  additionally the overall
 47  her facial expression
 45  his facial expression
 44  his facial features
 43  her facial features
 43  additionally the edges
 41  the edges of
 40  these elements combine
 39  the man on
 38  the woman in
 36  the overall lighting
```

The **distinctive transition family** is `Most notably …` / `The most prominent …` (267 occurrences) used to introduce **evidence #1**, followed by repeated `Additionally …` chains for evidence 2..N. Our current prompt only mentions `Additionally / Furthermore`.

### 1.4 Distinctive vocabulary not yet in the prompt

Words that appear ≥ 250 times per 2000 fake GTs and are essentially absent from real GTs (ratio score). These are high-leverage:

```
word               score        fake#   real#
appears                 798.3     1652       0     ← dominant verb
several                 765.9     1585       0
unnaturally             627.7     1299       0
synthetic               572.1     1184       0
overly                  549.9     1138       0     ← qualifier
inconsistent            539.3     1116       0
appear                  509.8     1055       0
visual                  457.1      946       0
lacks                   445.5      922       0     ← "lacks fine detail"
unnatural               433.5      897       0
inconsistencies         419.0      867       0
facial                  396.3      820       0
manipulation            386.6      800       0
contains                378.4      783       0
lacking                 337.8      699       0     ← "lacking definition"
distorted               335.8      695       0
lack                    331.0      685       0
creating                328.1      679       0
compared                301.1      623       0     ← "compared to surrounding"
text                    294.8      610       0
surrounding             286.1      592       0
blending                279.8      579       0     ← "blending into the shirt"
suggest                 273.5      566       0
appearance              250.8      519       0
digital                 245.5      508       0
subtle                  245.0      507       0     ← used for soft artifacts
appearing               232.0      480       0
particularly            223.7      463       0     ← intensifier mid-sentence
```

**The five missing-from-prompt high-frequency patterns:**

1. **`appears / appear / appearing`** (3187 hits ≈ 1.6 / sample) — the dominant verb linking evidence object to artifact claim ("the puppy's fur **appears** overly soft")
2. **`overly`** (1138, 0.57 / sample) — qualifier for soft/smooth/sharp ("overly smooth", "overly defined")
3. **`lacks / lack / lacking`** (2306 ≈ 1.15 / sample) — negation pattern ("lacks fine detail", "lacking definition")
4. **`blending`** (579) + **`compared to surrounding`** (623 + 592) — describes objects merging into context
5. **`particularly`** (463) + **`subtle`** (507) — qualifiers for borderline artifacts

---

## 2. REAL GT patterns

### 2.1 Statistics

| Metric | Value |
|--------|------:|
| Sentences (median / mean) | **2 / 1.9** |
| Words (median / mean) | **41 / 41.2** |
| Artifact-seed hits per sample | 0.13 |
| Authenticity-seed hits per sample | 1.90 |

Real GTs are **~3× shorter** than fakes and written in **plain English**, not forensic prose.

### 2.2 Top 10 opening 6-grams (54% of all reals open with one of these)

```
690  this picture looks real because the
199  this picture looks real because of
196  this picture is real because you
117  this looks like a real picture
 81  you can tell this picture is
 77  this picture looks real because you
 37  you can tell this is a
 31  this picture is real because the
 25  this looks real because you can
 19  this looks like a real photo
```

### 2.3 Top transitions

```
660  you can see                ← dominant transition for REAL
 54  you can also
 51  if you look
 35  that's what happens
 27  it looks like
 22  real clothes get
 14  see how the
 13  it's not perfectly
```

The pattern `you can see how …` (660 sentence-internal + 14 standalone) is the canonical real transition. Almost nothing in common with the fake transition family.

### 2.4 Distinctive REAL vocabulary

```
word               score        real#   fake#
happens                2918.8      482       0   ← "real X happens when..."
really                 2258.8      373       0
lots                    908.4      150       0
wrinkly                 902.3      149       0   ← imperfection vocab
move                    859.9      142       0
tell                    757.0      125       0
things                  684.3      113       0
exactly                 666.1      110       0
gets                    520.8       86       0
outside                 504.7      214       1
sticking                484.5       80       0
bits                    460.2       76       0
what                    439.3      300       2
happen                  423.9       70       0
bumpy                   408.0      173       1
fast                    405.7       67       0
bumps                   393.6       65       0
bunched                 393.6       65       0
lady                    327.0       54       0
dirty                   296.7       49       0
walk                    254.3       42       0
super                   254.3       42       0
crinkles                248.3       41       0
kids                    242.2       40       0
messy                   233.9      644      10
clothes                 223.4      326       5
life                    219.3       93       1
play                    213.8      146       2
```

Vocabulary is essentially **everyday physical-world English**: *wrinkly, bumpy, crinkles, messy, dirty, sticking, walking, lots, super, kids, lady*. No forensic terminology at all.

**Authenticity cues that GTs use as "this looks real because …":**

- Wrinkles / crinkles / messy fabric
- Dirt / scuff marks / wear
- Motion blur / things moving fast
- Lots of small details that you couldn't make up
- Imperfect alignment / "not perfectly straight"
- Natural facial expressions (laughing, surprised, candid)

These are *imperfections framed as evidence of reality* — the exact opposite framing from fake GTs (where imperfections are evidence of synthesis).

---

## 3. Mismatches with current v2 prompts

### 3.1 `VLM_USER_PROMPT_HYPOTHETICAL_FAKE` (current)

> *"…describe the synthesis artifacts visible in each — distorted text, warped geometry, smudged or merged textures, anatomical errors, inconsistent lighting or shadows, or unnatural object boundaries. Use connectives such as 'Additionally' and 'Furthermore' to chain the observations together."*

**What's missing per the mined data:**

| Missing from prompt | GT freq / sample | Why it matters |
|---------------------|-----------------:|----------------|
| The verb "appears" | 1.6 | Standard evidence→claim link |
| Qualifier "overly" | 0.57 | Modifies "smooth/soft/sharp" |
| Negation "lacks/lacking" | 1.15 | "lacks fine detail" pattern |
| Body intensifier "particularly" | 0.23 | Mid-sentence pointer to a region |
| Transition "Most notably / The most prominent" | 0.13 | Introduces evidence #1 |
| Phrase "blending into surrounding" | 0.30 | Specific merge-artifact phrasing |
| "Compared to (surrounding / rest of)" | 0.31 | Comparison framing |

### 3.2 `VLM_USER_PROMPT_HYPOTHETICAL_REAL` (current)

> *"…describe the authentic photographic cues visible in each — natural texture detail, plausible lighting and shadows, consistent geometry, and physically coherent object boundaries."*

**This is forensic prose**. Real GTs do not use any of: "authentic photographic cues", "plausible lighting", "consistent geometry", "physically coherent object boundaries". They use:

- `you can see how the X is wrinkly / messy / bumpy / dirty`
- `that's exactly what happens when …`
- `lots of small details`
- `the way the cloth moves / falls / bunches`

The current prompt asks the model to write in a register that **does not match the reference at all**. Even with semantic-coverage matching, this hurts BERT (which is contextual cosine) and probably hurts entity recall too, because the model writes about wrong cue types.

### 3.3 `VLM_USER_PROMPT` (primary, used 50% of v2 train + all VLM-only inference)

Mostly fine — it's the joint "decide" prompt. Two small additions are worth making:

- Mention "overly smooth / overly defined" and "blending into surroundings" as artifact patterns
- Mention "wrinkles, messy textures, dirt, motion blur" as real-cue patterns (since for the primary prompt the model decides which class, both cue families are relevant)

---

## 4. Where the v2 build job stood (job 91267)

Submitted on 2026-05-27 22:57:39 UTC with the *old* prompt_v2.txt (FORENSIC-ANALYSIS preamble but no vocabulary alignment). Per user decision after seeing this analysis: **cancel and resubmit** with the v2.1 prompt revisions described in §5 below.

---

## 5. Planned v2.1 prompt revisions

To be written into `dataset/prompt_v2.txt`. Direct quotes are taken from this analysis (most frequent GT openers / transitions) so the new prompt teaches the model the actual reference vocabulary.

### 5.1 Primary `VLM_USER_PROMPT` (50% of train + VLM-only fallback inference)

Keep the structure. Add two clauses to the artifact list and add a real-cue clause:

> *"…describe the visual evidence visible in each — distorted text, warped geometry, smudged or merged textures, **overly soft or plastic skin/fabric**, **regions that lack fine detail or blend into their surroundings**, anatomical errors, inconsistent lighting or shadows, unnatural object boundaries, or, for real images, **authentic everyday cues such as wrinkles, motion blur, dirt, messy textures, lots of small natural details**. Use connectives such as **'Most notably' and 'Additionally'** to chain the observations together."*

Why these specific additions, by the numbers:
- "overly soft / lack fine detail" — 1138 + 2306 = ~3400 hits / 2000 fakes
- "blending into surroundings" — 579 + 592 = ~1170 hits
- "wrinkles, motion blur, dirt, messy textures" — direct copies of top REAL distinctive words
- "Most notably / Additionally" — replaces "Additionally / Furthermore" (193 + 1000+ hits vs current "Furthermore" being rare)

### 5.2 `VLM_USER_PROMPT_HYPOTHETICAL_FAKE` (Pass-2 conditional)

Keep the FORENSIC-ANALYSIS preamble. Rewrite the body to mirror the mined fake-GT register:

> *"…in one coherent paragraph, describe how each region **appears unnatural, overly smooth, lacks fine detail, blends into surrounding textures, or shows distorted text / warped geometry / anatomical errors / inconsistent lighting**. Start the paragraph with **'Most notably'** for the strongest piece of evidence and use **'Additionally'** to chain the others."*

Why:
- `appears / appear` is in 80%+ of fake GTs as the main verb — naming it in the prompt licenses the model to use it
- `Most notably ... Additionally ...` is the exact template family ~270 fake GTs use

### 5.3 `VLM_USER_PROMPT_HYPOTHETICAL_REAL` (Pass-2 conditional, soft-match per user)

Keep the FORENSIC-ANALYSIS preamble but switch the body register from forensic prose to plain English with concrete authentic-cue vocabulary:

> *"In plain everyday language, describe several specific things in the image that make it look like a real photograph — for example wrinkles or crinkles in clothing, messy or dirty surfaces, hair or fabric moving, candid facial expressions, motion blur, or lots of small natural details that would be hard to invent. Aim for one short paragraph (roughly 1–3 sentences) similar in length to a casual description."*

Why:
- "wrinkles / crinkles / messy / dirty / motion blur / lots of small details" are the top distinctive REAL words
- "1–3 sentences" matches the 2-sentence median real GT length
- "casual description" prevents the model from drifting into forensic register

### 5.4 What we explicitly do NOT change

Per PGT brittleness finding (arXiv 2506.11031 — −10.9 macro F1 from dropping the phrase "synthesis artifacts"):

- Keep `"Examine the style and the synthesis artifacts"` exactly in the primary prompt
- Keep the verbatim list of forensic-category nouns (distorted text, warped geometry, etc.) — FakeVLM 2025 reported +12.8 F1 from naming categories
- Keep the terminal `Verdict: real / fake` line on its own (the regex in `build_submission.py` depends on it)
- Keep the FORENSIC-ANALYSIS preamble verbatim — matches the +4.2 gold-verdict experiment

---

## 6. Caveats

- **The 2000-sample analysis** is a representative slice, not the whole 320k+130k. We don't expect the head of the distribution to shift, but the long-tail vocabulary might. The current numbers explain >50% coverage of each pattern in the sample, which is enough signal.
- **`appears` is high-frequency but a generic verb.** Telling the model to use it doesn't directly improve EntityF1 — it just keeps the model close to the GT vocabulary surface, which helps BERT.
- **"Most notably"** as an opener is not a *guarantee* of higher EntityF1 — but it correlates strongly in the GT distribution, so models that mimic it should also mimic the multi-region enumeration that follows.
- **Real GT plain-English prompt** is the biggest register change. Risk: model collapses to too-short outputs on reals if we don't also keep the "list several specific things" instruction. We keep that instruction in §5.3.

---

## 7. Reproducing this analysis

```bash
cd /shared/workspace/lrv/luka/XPlainVerse-ACMChallenge/code/XPlainVerse-ACMChallenge
python3 dataset/analyze_gt_vocab.py --per-class 2000
```

Runs in ~1.4 s on the prebuilt `train_vlm.jsonl`. Login-node-safe (text-only, no image reads).
