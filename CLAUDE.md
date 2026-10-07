# CLAUDE.md

Guidance for AI agents working in this repository.

## What this is

Voice-MAD (鬼畜调音) prototype: render a target song's vocal melody using an
anime character's spoken lines as the codebook, without vocoders — the
"unit-selection VQ" formulation. Target song of record: 春日影 (MyGO!!!!!),
rendered with 长崎素世 (Soyo) lines.

## Repo layout

- `prototype/engine/token_dp.py` — current engine (v8): keyframe-constrained
  segment Viterbi DP over raw clips. Start here.
- `prototype/engine/` — older v3–v7 engines (audio_match, mosaic, unit_select,
  sing_melody, sampler_match, phrase_match, nucleus_bank, diagnose_*).
- `prototype/lib/` — library building (build_library.py), clip index JSONs,
  hires audio loading. `samples/` (audio) is NOT in the repo.
- `prototype/legacy/` — v1–v7 engines + results, kept for reference only.
- `prototype/make_review.py` — generates `review.html` from `out/*.metrics.json`.
- `scripts/` — dataset downloaders (Genshin / Star Rail / anime dry voice).
- `materials/` (top level, gitignored) — raw voice packs, ~134 GB locally.

## Commands

```bash
./setup.sh                                   # venv + deps
python3 prototype/engine/token_dp.py --help  # current engine entry
python3 prototype/make_review.py             # rebuild review.html after renders
```

## Hard rules for agents

- Never commit audio (`*.wav/*.mp3/materials/`). See CONTRIBUTING.md.
- Do not reintroduce vocoders (WORLD/PSOLA/neural resynthesis) into the main
  render path — v3–v5 were rejected by ear for this. Allowed modifications of
  a clip: constant gain (closed-form), 5 ms fades, syllable-boundary crops
  (>=70% kept), bounded resample speed shifts for low notes, choke cuts at
  keyframes (>=40% kept).
- The objective is frame-additive so segment DP is exact; do not replace the
  solver with MCTS/beam search unless the loss becomes non-additive.
- Metrics that matter (see `*.metrics.json`): M1 pitch accuracy ±50 cents,
  M2 voicing recall/false-alarm, M3 onset timing (±30 ms share, late share),
  M5 dynamics correlation, distinct_lines (diversity), crop_mean.
- After editing engine code, run a short-window render and check metrics
  before claiming success; the author judges final quality by ear.
