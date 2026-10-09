# MMVQ vs MMQ crossover data, five NVIDIA architectures (llama.cpp #28090, 2 Sep 2026)

The measurements behind the 2 Sep 2026 comment on ggml-org/llama.cpp #28090: 28 tables
(5 GPUs, up to 3 models, Q4_0 and Q8_0), 8,064 llama-bench rows. Nothing here was re-measured
for this package; it is the raw output plus tables regenerated from it by
`notebooks/make_data_package.py`, which first checks every ratio, round count, noise floor and
crossing against the reduction the comment was written from.

## Method

- llama.cpp `b96806d96` (`b96806d96061049a5b574269b049bf6241d63d46`, master on 1 Sep 2026).
- Three binaries per card, all built from that commit:
  - `selected`: unmodified.
  - `force_mmq`: in `ggml_cuda_should_use_mmvq()` the final fall-through returns `ne11 <= 1`
    (`patches/force_mmq_fallthrough_only.patch`). For the L4 the four inner `default:` returns
    (Ada, Blackwell, DGX Spark, CDNA2) were changed the same way
    (`patches/force_mmq_with_inner_defaults.patch`), because an L4 reaches the Ada branch's own
    `default:` and never the final fall-through; on the other four cards those inner defaults are
    dead code, so both patches route identically there.
  - `null_rebuild`: the same source as `selected`, configured and built a second time.
- Build: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<60|75|80|86|89> -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler`.
  Binary SHA-256 and the memory pool used are in `meta/<arch>_binaries.json`.
- Models: unsloth Llama-3.1-8B-Instruct (BF16 GGUF), Llama-3.2-3B-Instruct and
  Llama-3.2-1B-Instruct (F16 GGUFs), each requantized with
  `llama-quantize --pure --allow-requantize` to Q4_0 and Q8_0. `models.csv` has the source URL
  pinned to the Hugging Face commit, the source size and SHA-256, and llama-bench's `model_size`
  for every quantized file.
- Command: `llama-bench -m <model> -ngl 999 -fa 1 -p 1,2,...,16 -n 0 -embd 1 -r 30 -o jsonl`.
  `-t` was not set (llama-bench default); `-b` and `-ub` were the defaults, 2048 and 512.
- Order: six rounds; in each round every binary runs every model file, and the binary order rotates
  by one position per round.
- Telemetry: nvidia-smi at 1 Hz during each llama-bench invocation (SM clock mean, p10, p50,
  temperature, power). Clocks were not pinned on any card.

## Cards

| arch | GPU | SMs | where | driver | nvcc | pool | SM clock, mean per file (MHz) |
|---|---|---:|---|---|---|---|---|
| sm_60 | Tesla P100-PCIE-16GB | 56 | Kaggle | 580.159.04 | 12.8 | legacy (`GGML_CUDA_NO_VMM=ON`) | 1327 to 1328 |
| sm_75 | Tesla T4 | 40 | Kaggle | 580.159.04 | 12.8 | legacy (`GGML_CUDA_NO_VMM=ON`) | 1341 to 1545 |
| sm_80 | A100-SXM4-40GB | 108 | Colab | 580.82.07 | 12.8 | stock VMM | 1400 to 1410 |
| sm_86 | RTX 3050 Laptop 4 GB | 16 | local, WSL2 | 596.08 | 13.2 | stock VMM | 802 to 873 |
| sm_89 | L4 | 58 | Colab | 580.82.07 | 12.8 | stock VMM | 1336 to 1789 |

## Files

| path | what |
|---|---|
| `summary.csv` | one row per table: `force_mmq/selected` at ne11 = 5..8, the ne11 = 8 paired geometric mean with its 95% CI, rounds won, both noise floors and both first-crossing values, the ne11 9..16 control range, SM clock and max temperature |
| `tables/<arch>_<gpu>_<model>_<quant>.csv` | 28 per-ne11 tables, ne11 = 1..16: median t/s of `selected`, `force_mmq` and `null_rebuild`; `force_mmq/selected` and `null_rebuild/selected` as ratio of medians and as paired geometric mean with 95% bootstrap CI; rounds won |
| `raw/bench_<arch>.jsonl` | every llama-bench row: binary, round, position in the round, model, quant, ne11, avg and stddev t/s and ns, model_size, and the invocation's telemetry summary |
| `telemetry/telemetry_per_invocation.csv` | the telemetry summary once per invocation (504 invocations) |
| `meta/<arch>_binaries.json`, `meta/<arch>_env.json` | commit, binary SHA-256, pool; GPU, driver, nvcc, Python, platform, start time |
| `models.csv` | model sources and SHA-256, quantized `model_size` |
| `patches/` | the `force_mmq` change, both forms |
| `code/mmvq_study.py`, `code/reduce.py` | the driver that ran every card, and the reducer the 2 Sep comment's numbers came from |
| `SHA256SUMS` | checksums of every file here |

## The two statistics

- Ratio of medians (the 2 Sep comment): median t/s over the six rounds per binary, then the ratio.
  Noise floor = max |null_rebuild / selected - 1| over ne11 = 2..8; crossing = the first ne11 in
  2..8 where `force_mmq / selected` exceeds 1 + floor. ne11 = 1 is excluded because both binaries
  run MMVQ there.
- Paired geometric mean (zhihz's format): the per-round ratio `x / selected` within each round, the
  geometric mean over the six rounds, and a 95% percentile bootstrap over rounds (10,000 resamples,
  seed 0). `summary.csv` also applies the same floor-and-crossing rule to this statistic
  (`paired_noise_floor`, `first_ne11_beyond_paired_floor`).

The two agree in direction on every table, and on the first crossing everywhere except the T4 1B Q8_0,
the RTX 3050 and the L4 8B Q8_0 (all three under caveats). At ne11 = 8 they are close to each
other on the P100, T4 and A100 (0.7% at most); on the L4 8B the paired geometric mean is lower than the ratio of medians
(Q4_0 1.142 [1.107, 1.165] against 1.161; Q8_0 1.060 [1.024, 1.080] against 1.076).

`force_mmq` at ne11 = 8 routes exactly as a cutoff of 7 would (MMQ at 8, nothing else changed at 8),
so the ne11 = 8 column is directly comparable with a `cutoff7 / selected` measurement.

## Caveats

- **Kaggle builds (P100, T4) used `GGML_CUDA_NO_VMM=ON`.** The Kaggle image has no `libcuda.so`
  where CMake looks, so configure failed and the driver retried with the legacy pool (recorded in
  `meta/sm60_binaries.json` and `meta/sm75_binaries.json`). That changes the temporary-buffer pool,
  not kernel selection; both binaries in each pair share it and the ne11 9..16 controls are flat, so
  the ratios stand, but absolute t/s on those two cards is not a stock build.
- **T4 1B Q8_0 has a 9.0% noise floor** (ratio of medians): `null_rebuild` ran 3 to 9% faster than
  `selected` at ne11 = 1..3 across rounds. Its small-batch rows should not be read; it is reported as
  "never" crosses under that floor (3.0% floor and a crossing at 8 under the paired statistic).
- **Clocks were not pinned.** Cloud runs had no permission; the per-invocation clock record is the
  control instead. The L4 ran between 1336 MHz (8B) and 1789 MHz (1B) mean, power-limited at 72 W.
- **RTX 3050 Laptop (4 GB):** no 8B (does not fit); about 800 MHz under sustained load; every
  invocation started at or below 72 C (a cooldown gate added after the first attempt throttled from
  1519 to 712 MHz). The laptop's state still varied across rounds, so paired-round ratios scatter
  widely: the paired noise floor is 8 to 32% against 0.4 to 3.2% for the ratio of medians, and the
  crossings under the paired statistic move later (3B Q8_0 from 5 to 6, 1B Q4_0 from 4 to 7,
  1B Q8_0 from 4 to 5; 3B Q4_0 stays at 6). The direction (MMQ wins below 8 on this card) holds under both; the exact crossing
  on this card is soft.
- **L4 8B Q8_0:** crosses at 7 under the ratio of medians (1.0072 against a 0.59% floor, a margin of
  0.13%) and at 8 under the paired statistic (1.55% floor). Read it as 7 or 8. The 8B Q4_0 crosses at
  7 under both.
- **The first L4 run is not included.** In it `force_mmq` only changed the final fall-through, which an
  L4 never reaches, so the two binaries were identical (ratio 1.00 at every ne11). The L4 data here is
  the rerun with the Ada default changed.
- Model files: the 2 Sep runs did not record SHA-256 of the quantized files. The sources are the files
  in `models.csv` (same repos, file names and byte sizes; the repos' last commits are from May 2025 for
  the 1B and 8B and November 2025 for the 3B, all before these runs), and `model_size` per quantized file is in the raw rows.
- CMakeCache files were not kept for these builds; the configure flags above are the complete set
  passed by the driver (`code/mmvq_study.py`, `phase_build`).
- Differences from zhihz's V100/L20 protocol: no `-t 4` (llama-bench default threads; with every
  layer offloaded the thread count should not matter, but it is a difference), `force_mmq` instead
  of `cutoff7` (identical routing at ne11 = 8, see above).

## License

Data, tables and documentation: CC BY 4.0 (see `LICENSE`). Scripts under `code/`: MIT.
Please cite as: Manu Nicholas Jacob, "MMVQ vs MMQ crossover data, five NVIDIA architectures", 2026,
https://github.com/manunicholasjacob/llama-cpp-mmvq-crossover-data

## October 2026 L4 re-runs (`runs/`)

Same commit (b96806d96) and models, four binaries (`selected`, `force_mmq`, zhihz's `cutoff7`, `null_rebuild`),
`llama-bench -ngl 999 -fa 1 -n 0 -embd 1 -t 4 -b 2048 -ub 512 -p 1..16 -r 30`, six rounds, Colab L4, driver 580.82.07.

- `runs/l4-2026-10-07-unpinned-run1/`: clocks not locked. Under 8B load the 72 W L4 sits on `sw_power_cap`, and the
  MMQ build ran at a higher mean SM clock than the MMVQ build (8B Q4_0: 1295 vs 1130 MHz).
- `runs/l4-2026-10-07-pinned-810MHz/`: `nvidia-smi -lgc 810,810`. Every pin from 1395 to 900 MHz hit the power cap
  under 8B load; at 810 the MMQ build holds the pin exactly and the MMVQ path stays at 91 to 97% of samples.
  Summary in `out/SUMMARY.md`; telemetry is per llama-bench invocation.


## October 2026 L20 at 810 MHz (zhihz)

[`runs/l20-2026-10-07-pinned-810MHz/`](runs/l20-2026-10-07-pinned-810MHz/README.md) adds the complete requested L20 sweep (92 SMs, 144 invocations, 2,304 rows) with six byte-matched October model files, four builds, n=1..16 and six outer rounds. Both-GPU paired reductions and figures use the existing pinned L4 raw data. At n8, the L20 8B force_mmq ratios are 1.07116 (Q4_0) and 1.11463 (Q8_0); small-model outcomes are mixed.

**These are exploratory stock-pool timings:** the independent exact-allocation MMQ diagnostic reported 385 errors, while the original selected/exact-pool and force_mmq/stock-pool controls passed. The run README preserves this limitation, achieved-clock scope and the remaining device confounders. This contribution does not establish a universal SM-count rule or source-code correctness acceptance.

The original collected 515-file manifest is preserved inside the run; its `SHARE_SHA256SUMS` covers added documentation/code/figures as well. The repository-level `SHA256SUMS` is regenerated to cover the current checkout, including both October L4 runs and this L20 contribution.


## October 2026 L20 after #29953: clock comparison (zhihz)

[`runs/l20-2026-10-09-post-29953-clock-pilot/`](runs/l20-2026-10-09-post-29953-clock-pilot/README.md) adds a separate fixed-source study at `fc9ce6b9` (including #29941 and #29953): a passing focused dense safety gate, 216 stock-pool timing invocations, and 324 rows over six outer rounds.

At n=8, cutoff7 / selected throughput changes were **+17.92% (8B Q4_0) and +11.30% (8B Q8_0) at 810 MHz**, versus **−17.82% and −11.20% at measured default clocks**. The fixed exact-allocation diagnostic had zero memcheck errors and passed 168/168 numerical cases; its pre-fix control reproduced invalid reads and aborted. The run documents the explicit admission amendment, preserves the original failed gate, and includes a CPU-only independent reviewer and checksums.

This supports a focused clock-aware investigation while keeping the static table unchanged. It is not full-model correctness acceptance or a portable selection rule, and does not replace the historical b968 data. See the run README for measured default frequencies, workload controls, metadata redactions and provenance.


## October 2026 L20 locked-clock follow-up (zhihz)

[`runs/l20-2026-10-09-locked-clock-sweep/`](runs/l20-2026-10-09-locked-clock-sweep/README.md) adds the `fc9ce6b9` sweep: 246 formal invocations / 258 rows, six outer rounds, valid locks at 1200/1500/1800/2100 MHz. At n=8, 8B Q4_0 changes from +5.60% at 1200 MHz to -5.25% at 1500 MHz; 8B Q8_0 is already -6.98% at 1200 MHz. The requested 2520 MHz point failed sustained lock acceptance and is excluded, with all evidence retained.

The run includes raw outputs, telemetry, source/build provenance, complete paired reductions and a CPU-only independent reviewer. The prior 810 MHz point and safety gate are not repeated or pooled. These results narrow the next investigation to quantization-specific frequency intervals and matrix-shape controls; they do not establish a portable default routing rule.
