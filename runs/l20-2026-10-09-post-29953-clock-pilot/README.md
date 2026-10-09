# L20 post-#29953: focused dense safety gate and clock comparison

Collected by zhihz on 9 October 2026 for [llama.cpp #28090](https://github.com/ggml-org/llama.cpp/issues/28090). Fixed source: [`fc9ce6b9d52a8504edcb262abc92737c2289f96c`](https://github.com/ggml-org/llama.cpp/commit/fc9ce6b9d52a8504edcb262abc92737c2289f96c), including #29941 and #29953.

**Reviewed outcome: the fixed source passed the focused dense admission gate, and n=8 MMQ wins on 8B at 810 MHz but loses at default clocks. This does not establish full-model correctness, a crossover frequency or a portable selector.**

## Conditions

- One NVIDIA L20, SM89, 92 SMs; driver 580.126.09, nvcc 13.0.88, Compute Sanitizer 2025.3.1.0.
- Three pure GGUF files: Llama-3.1-8B Q4_0/Q8_0 and Llama-3.2-1B Q8_0. Sizes and SHA256 match the corresponding October inputs; no model weights are included. `models.lock.json` and `models.verified.json` record identity.
- Timing builds: stock-pool `selected`, `cutoff7`, independent same-source `null_rebuild`. `cutoff7` overrides Ada dense Q4_0/Q8_0 selection to use MMVQ only through n=7; its n=8 intervention is MMQ. This is an experimental selector, not a proposed upstream rule.
- Six outer rounds; clock order alternates, and three file/build orders rotate. Each invocation has 100 internal repeats. n=8 runs separately; n=1/n=16 share control invocations. 216 invocations, 324 unique rows.
- Flags: `-ngl 999 -fa 1 -n 0 -embd 1 -t 4 -b 2048 -ub 512 -r 100 -o json`, with `-p 8` or `-p 1,16`.

## n=8 results

Paired geometric-mean **throughput** change, cutoff7 / selected. Positive values favor MMQ. 95% percentile bootstrap intervals resample the six outer rounds (10,000 resamples, seed 0); they are descriptive intervals.

| Input | Locked 810 MHz [95% CI] | Default clocks [95% CI] |
|---|---:|---:|
| 8B Q4_0 | +17.915% [+17.903%, +17.927%] | −17.818% [−17.934%, −17.735%] |
| 8B Q8_0 | +11.298% [+11.277%, +11.319%] | −11.204% [−11.223%, −11.189%] |
| 1B Q8_0 | +0.566% [+0.542%, +0.592%] | −16.279% [−16.336%, −16.230%] |

Both 8B low-clock gains and all default-clock losses held across all six paired rounds. The 1B low-clock gain is below the predefined 3% materiality threshold. `summary.csv` and `analysis.json` contain the complete estimates, paired ratios and control checks.

All 54 low-clock n=8 windows held 810 MHz in every busy sample (minimum 6 busy samples/window). Default window means ranged from 2462.5 to 2520 MHz (minimum 4 busy samples/window). For 8B Q4_0, selected averaged 2473.3 MHz versus cutoff7 at 2520 MHz: **default means reset/unlocked, not both paths pinned to 2520 MHz**. Other file/build default-group means were 2520 MHz. Memory clocks stayed at 9001 MHz; maximum busy temperature was 56°C.

The maximum paired same-source rebuild deviation over n={1,8,16} was 0.0432%; maximum unchanged-route cutoff7 deviation at n={1,16} was 0.0479%. Both pass the predefined 2% limit. Telemetry is invocation-wide at 100 ms, includes loading/warmup, and does not isolate individual timed repeats.

## Focused safety gate and explicit amendment

Two additional diagnostic builds force MMQ at n>=2 with an exact `cudaMalloc` pool and CUDA Graphs disabled. The pre-fix control restores only the immediate parent's `mmq.cu`/`mmq.cuh`; `source-provenance.json` records their hashes.

- Pre-#29953 control: 1,842 memcheck errors, including MMQ invalid global reads, followed by an abort; it did not complete all numerical cases.
- Fixed exact-allocation build: **zero memcheck errors and 168/168 CPU-reference numerical cases passed**.
- Stock-pool selected and cutoff7: each passed 168/168 numerical cases.

Coverage is Q4_0/Q8_0 × n=2..8 × 12 dense attention/FFN geometries. It excludes full-model numerical acceptance, output-head coverage and MoE.

The original v1 admission incorrectly required the intentionally faulty parent both to report errors and to complete all numerical cases. It stopped after the parent aborted. **Before any model timing observations**, v2 explicitly allowed that positive control to abort after MMQ invalid-read reports and nonzero error count/exit. Fixed-code requirements and all performance/analysis criteria remained unchanged. See `gate/amendment.json`, `gate/gate.json`, and the retained failed v1 `gates/1791514484179779852/`. This is not a claim that the original v1 gate passed. The recorded sanitizer command was not relaxed; see [NVIDIA error actions](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#error-actions).

`protocol.json`, `pilot.py`, and `analysis.json` are frozen records: the protocol still says v1, while the accompanying amendment defines the actual v2 admission. The machine-readable analysis decision is a collector workflow label; the scientific interpretation is stated here.

## Interpretation and limits

Keep the static cutoff table unchanged. These results support testing a clock-aware hypothesis with workload/shape controls. Only two frequency conditions, three files and n={1,8,16} were timed; intermediate clock crossover points, n=2..7 performance, other quantizations and portability remain untested.

The historical b968 run remains separate. Differences in its Q4_0 gain cannot be attributed solely to #29953 because source/environment changes separate the runs. The old diagnostic failure remains valid evidence of that old run; the new focused gate does not retroactively turn it into correctness acceptance.

## Evidence, privacy and checksums

- `raw/`: original per-invocation JSON, combined JSONL, stderr, timestamped telemetry and invocation windows.
- `build/`, `build.json`, `host-metadata/`: source excerpts, logs, CMake flags, binary hashes and environment records.
- `gate/`, `gates/`: amended fixed/stock checks and the original failed positive-control gate.
- `pilot.py`, `code/`: recorded collector, harness, input preparation and independent CPU-only reviewer. Source archives and build binaries are not included, so the recorded collector is not a turnkey GPU deployment bundle.
- `SHA256SUMS.json`: unchanged original 923-file manifest. **Seven metadata files have public redactions**, so use `PUBLIC-PROVENANCE.json` to map their original and published hashes; all other collected files remain byte-identical.
- `SHARE_SHA256SUMS`: hashes of every public file here, including additions and redacted metadata. The repository-level manifest is also updated.

Only transient hostnames and GPU UUID/serial fields are redacted. Benchmark inputs, measurements, timing windows, numerical diagnostics and statistical outputs are unchanged. The private transfer archive remains local; its hash is recorded only for provenance. No server address, credentials or model weights are published.

## Recompute without a GPU

From the repository root, using Python 3.11+ and only its standard library:

```sh
python3 runs/l20-2026-10-09-post-29953-clock-pilot/code/review_evidence.py \
  runs/l20-2026-10-09-post-29953-clock-pilot --out /tmp/l20-fixed-review.json
```

The reviewer validates public checksums and the original/redacted provenance, checks source and model identities and all gate logs, verifies 216 invocations / 324 unique rows / 100 internal samples, compares original JSON with the combined rows, and independently recomputes paired estimates, intervals, controls, clocks and the decision. It also verifies that the amendment precedes the first timing window.

Data and documentation follow the repository's CC BY 4.0 license; new scripts follow MIT. Included llama.cpp source retains its upstream licensing.
