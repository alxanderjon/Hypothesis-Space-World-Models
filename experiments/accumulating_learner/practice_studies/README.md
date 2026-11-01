# Practice-half studies (Trial 27 build, 31 Oct – 1 Nov 2026)

Scripts behind the practice-half results in `docs/research-log.md` (development half only; never run on the evaluation half).
They were run from a scratch directory (paths inside point there) and are kept here as the record of what was run.

- `probe_scratch.py`, `probe_accum.py`: strong from-scratch baseline at M = 8–64; static-library accumulation probe.
- `probe_pattern.py`, `pattern_report.py`: option (b) — equation-pattern library vs strong scratch, pieces-only, decoys,
  copy-the-closest, removal check and provenance.
- `bench_export.py`, `bench_pysr.py`, `bench_eval.py`, `bench_eval_part.py`, `bench_compare.py`: PySR 2.7.2 benchmark
  (identical data and budgets; high-precision finish; external exactness test).
- `pysr_sanity.py`: operator-set sanity check; `pysr_feed_demo.py`: feeding discovered pieces / patterns into PySR.
