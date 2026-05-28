# Provenance — Codabench scorer extraction

## Source

- **Image:** `docker://abhijeet1317/xdd-scorer:2026-v5`
- **Docker Hub:** https://hub.docker.com/r/abhijeet1317/xdd-scorer/tags
- **Challenge site:** https://explainable-deepfake-detection.github.io/
- **Extracted paths:** `/app/program/` and `/app/data/` (empty reference placeholders only)

## Extraction command (Elixir Lj GPU node)

```bash
DEST=/home/jakob/luka/code/XPlainVerse-ACMChallenge/evaluator_codabench
mkdir -p "$DEST"
apptainer exec --no-home ~/containers/xdd-scorer_2026-v5.sif \
  tar -C /app -cf - program data | tar -C "$DEST" -xf -
```

## Image metadata

- **Entrypoint:** `python3 /app/program/scoring.py /app/input /app/output`
- **Compressed size (Docker Hub):** ~5 GB
- **Local SIF:** `~/containers/xdd-scorer_2026-v5.sif` (~4.4 GB)
- **Pushed:** ~27 May 2026 (tag `2026-v5`)

## Code authorship

Python files under `program/` are **organizer code** shipped inside the Docker image, not original work in this repository. They are vendored here for offline inspection, diffing against `evaluation/`, and local dry runs. Do not modify scoring logic unless reconciling with a newer Docker tag from the organizers.

## Re-sync when organizers release a new tag

```bash
apptainer pull ~/containers/xdd-scorer_2026-v5.sif docker://abhijeet1317/xdd-scorer:2026-v6   # example
# re-run extraction tar command above, then diff program/
```

Compare `metadata.yaml` and `SCORE_KEYS` in `scoring.py` after each pull — metric definitions may change between tags.
