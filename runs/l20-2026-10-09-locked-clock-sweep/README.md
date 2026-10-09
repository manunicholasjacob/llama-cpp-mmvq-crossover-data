# L20 locked-clock sweep after #29953 (9 October 2026)

This follow-up measures the `n=8` MMQ/MMVQ throughput crossover on one L20 at
llama.cpp `fc9ce6b9d52a8504edcb262abc92737c2289f96c`, including #29941 and #29953.
It uses the same three byte-verified pure GGUFs as the preceding
[810 MHz / default-clock pilot](../l20-2026-10-09-post-29953-clock-pilot/README.md).
The 810 MHz timing and sanitizer gate were **not repeated** and their rounds are
not pooled into this sweep.

## Results

Paired geometric mean of cutoff7 / selected throughput; six outer rounds,
100 internal samples per row. Positive changes favor MMQ.

| Locked SM clock | 8B Q4_0 change | 8B Q8_0 change |
|---|---:|---:|
| 1200 MHz | +5.60% | -6.98% |
| 1500 MHz | -5.25% | -10.19% |
| 1800 MHz | -13.57% | -11.54% |
| 2100 MHz | -16.62% | -11.68% |

All eight valid 8B cells have the same direction in all six rounds. Q4_0 changes
from a material MMQ win at 1200 MHz to a material loss at 1500 MHz. This brackets
the observed transition; it does not locate an exact equality frequency.
Q8_0 has no win in the valid 1200-2100 MHz range. Its positive 810 MHz result from
the separate pilot motivates testing 810-1200 MHz, but is not a same-run bracket.
The 3% materiality threshold is this experiment's screening criterion, not an
upstream acceptance requirement. Full ratios, confidence intervals and controls
are in `summary.csv` and `analysis.json`.

1B Q8_0 changes are -11.02%, -15.30% and -16.52% at 1200, 1500 and 2100 MHz.
There is no 1B Q4_0 timing in this run, so the small-model Q8_0 results cannot
establish which Q4_0 matrix shapes account for the 8B gain.

## Lock acceptance and exclusions

The preflight probe accepted 2520 MHz, but the longer primary measurements did
not hold that pin: 12 selected/null-rebuild Q4_0 windows failed the registered
+/-15 MHz tolerance. **The entire 2520 MHz point is excluded from the valid
locked-clock curve**, including Q8_0. Its raw data and derived values remain
visible and are marked invalid in `summary.csv`; they are not relabeled as a
lower-frequency point. The highest valid primary lock is 2100 MHz. No lower
high-end fallback was measured after the formal-run failure.

All primary busy samples at the four retained locks match their targets exactly.
Memory clock remained 9001 MHz. Across the recorded formal windows, maximum
temperature was 61 C; at least three busy 100 ms telemetry samples were available
per invocation. Telemetry covers the full invocation including load/warmup, not
individual kernels. The maximum unchanged-route control deviation was 0.0528%.
Route controls were measured once at 1200 and the requested 2520 MHz, as scheduled;
there is no separate 2100 MHz route-control invocation. Those unchanged-route
controls are not substitutes for valid n=8 lock acceptance.

## Method and coverage

- One exclusive L20, SM 8.9, 92 SMs; NVCC 13.0.88; fixed source and model hashes.
- `selected` and `null_rebuild` are independent builds of unchanged source.
  `cutoff7` only adds an Ada dense Q4_0/Q8_0 return of `ne11 <= 7`.
- Flags: `-ngl 999 -fa 1 -n 0 -embd 1 -t 4 -b 2048 -ub 512 -r 100 -o json`;
  `-p 8` is a separate invocation; route controls use `-p 1,16`.
- 246 formal invocations / 258 rows / 25,800 internal timing samples, excluding
  ten preflight probe invocations. Primary clocks, files and binaries rotate
  by round. The outer six rounds are the statistical unit.
- Bootstrap percentile intervals use 10,000 paired-round resamples, seed 0.
- This is a short embedding/prompt benchmark, not production server decoding.
  No per-operator timing, MoE coverage, held-out routing validation, or new
  correctness gate was performed. It does not establish a portable clock rule
  or support changing the static table by GPU model or SM count.

## Files and independent reproduction

`raw/` retains every per-invocation JSON, stderr log, telemetry CSV and window,
including probes and the excluded high-clock data. `build/` contains the recorded
source files, configure/build logs and CMake caches. `protocol.json`, schedules,
model/source locks and the frozen `sweep.py` preserve the experiment specification.
No model weights or executable binaries are included; their recorded hashes
are available. Scripts in `code/` include preparation helpers and the independent
standard-library-only reviewer.

From the repository root, with Python 3.9 or newer:

```sh
python3 runs/l20-2026-10-09-locked-clock-sweep/code/review_sweep_archive.py runs/l20-2026-10-09-locked-clock-sweep
```

This verifies original/public hash mappings, public checksums, the isolated source
patch, recorded model hashes, complete schedules, raw/combined agreement,
individual samples, paired estimates and confidence intervals, achieved clocks,
and route controls. It writes a sibling report outside the sealed run directory.
`independent-review.json` records the review of the original private archive.

Only transient host/GPU identifiers in metadata are redacted, with explicit
original/public hashes in `PUBLIC-PROVENANCE.json`. Every other collected file is
byte-identical, including raw timing and telemetry. The original 1,064-entry
`SHA256SUMS.json` is preserved unchanged. `SHARE_SHA256SUMS` covers the derived
public package; the private transfer archive remains local.
