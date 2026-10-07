# Contributing

This is a personal research prototype that is being developed in public.
Issues and pull requests are welcome, but please keep the following in mind:

1. **No copyrighted audio in PRs.** Never commit voice clips, songs, stems, or
   rendered audio (`*.wav`, `*.mp3`, `materials/`, `prototype/lib/samples/`).
   Metrics/cues JSON and diagnostic PNGs are fine.
2. **Keep the formulation.** The current engine (v8, `prototype/engine/token_dp.py`)
   deliberately uses unmodified source clips plus constant gain and short fades —
   no vocoders, no pitch shifting of speech beyond bounded resample shifts.
   If you want to explore a different formulation, open an issue first.
3. **Run a smoke render before submitting engine changes** and attach the
   `*.metrics.json` diff.
4. Old engines live in `prototype/legacy/` and are kept for reference only;
   do not extend them.
