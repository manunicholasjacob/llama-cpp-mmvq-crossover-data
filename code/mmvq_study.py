#!/usr/bin/env python3
"""MMVQ/MMQ crossover study for ggml-org/llama.cpp #28090, plus the MMQ
mul_mat_id tail-padding over-read check for #27792 / PR #27044.

One script, run unchanged on Kaggle (P100 sm_60, T4 sm_75), Colab (A100 sm_80,
L4 sm_89) and a local RTX 3050 (sm_86). Everything it measures is written as
JSONL under <workdir>/results so the reduction is done by reduce.py, never by hand.

Phases (each skippable, each idempotent):

  probe      record GPU, driver, nvcc, compute capability
  build      clone llama.cpp at the pinned commit and build
               selected      unmodified source
               force_mmq     final fall-through of ggml_cuda_should_use_mmvq() -> ne11 <= 1
               null_rebuild  byte-identical source, configured and built a second time
               sanitizer     selected + exact-size pool patch, GGML_CUDA_NO_VMM=ON
               sanitizer_fixed  the same + PR #27044's one-line padding fix
  models     download F16 GGUFs and requantize with --pure to Q4_0 and Q8_0
  bench      interleaved llama-bench sweep, ne11 = 1..16, rotating binary order each round
  sanitize   compute-sanitizer memcheck over test-backend-ops MUL_MAT_ID, unfixed vs fixed

Usage:
  python mmvq_study.py --workdir /root/mmvq --phases probe,build,models,bench,sanitize
"""

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

PIN = "b96806d96"  # ggml-org/llama.cpp master, 1 Sep 2026
PIN_FULL = "b96806d96061049a5b574269b049bf6241d63d46"
REPO = "https://github.com/ggml-org/llama.cpp.git"

# Model sources. F16 so that --pure requantization produces a clean single-type file.
MODELS = {
    "llama3.2-1b": ("https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-F16.gguf", 2479595168),
    "llama3.2-3b": ("https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-F16.gguf", 6433687616),
    "llama3.1-8b": ("https://huggingface.co/unsloth/Llama-3.1-8B-Instruct-GGUF/resolve/main/Llama-3.1-8B-Instruct-BF16.gguf", 16068895872),
}
QUANTS = ["Q4_0", "Q8_0"]

# The one-line change #28090 used for its force_mmq binary.
FORCE_MMQ_OLD = "    return ne11 <= MMVQ_MAX_BATCH_SIZE;\n}\n\n// Device constexpr"
FORCE_MMQ_NEW = "    return ne11 <= 1;\n}\n\n// Device constexpr"

# PR #27044: size the ids-path tail padding from the flattened row count.
FIX_OLD = "        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);\n    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);\n    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());\n    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {\n        src1_scale.alloc(ne12*n_expert_used);"
FIX_NEW = "        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12*n_expert_used) * sizeof(block_q8_1_mmq);\n    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);\n    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());\n    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {\n        src1_scale.alloc(ne12*n_expert_used);"

# Diagnostic-only patch for the legacy pool: with GGML_CUDA_POOL_EXACT set, every pool
# allocation is its own cudaMalloc of exactly the requested size and is freed immediately,
# so compute-sanitizer sees the true buffer bounds instead of the 5% look-ahead slack.
POOL_ALLOC_OLD = "    void * alloc(size_t size, size_t * actual_size) override {\n#ifdef DEBUG_CUDA_MALLOC"
POOL_ALLOC_NEW = """    void * alloc(size_t size, size_t * actual_size) override {
        if (getenv("GGML_CUDA_POOL_EXACT") != nullptr) {
            void * ptr_exact = nullptr;
            ggml_cuda_set_device(device);
            CUDA_CHECK(ggml_cuda_device_malloc(&ptr_exact, size > 0 ? size : 1, device));
            *actual_size = size;
            return ptr_exact;
        }
#ifdef DEBUG_CUDA_MALLOC"""
POOL_FREE_OLD = "    void free(void * ptr, size_t size) override {\n        for (int i = 0; i < MAX_BUFFERS; ++i) {\n"
POOL_FREE_NEW = """    void free(void * ptr, size_t size) override {
        if (getenv("GGML_CUDA_POOL_EXACT") != nullptr) {
            ggml_cuda_set_device(device);
            CUDA_CHECK(cudaFree(ptr));
            return;
        }
        for (int i = 0; i < MAX_BUFFERS; ++i) {
"""


def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, cwd=None, check=True, capture=False, env=None, log_path=None):
    if log_path:
        with open(log_path, "ab") as fh:
            fh.write(f"\n$ {cmd}\n".encode())
            p = subprocess.run(cmd, shell=True, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    else:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                           stdout=subprocess.PIPE if capture else None,
                           stderr=subprocess.STDOUT if capture else None, text=capture)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}\n{p.stdout if capture else ''}")
    return p.stdout if capture else p.returncode


def nvsmi(query):
    out = sh(f"nvidia-smi --query-gpu={query} --format=csv,noheader,nounits", capture=True)
    return [x.strip() for x in out.strip().splitlines()[0].split(",")]


def patch_file(path, old, new):
    s = pathlib.Path(path).read_text()
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"expected exactly one match in {path}, found {n}")
    pathlib.Path(path).write_text(s.replace(old, new))


class Sampler:
    """1 Hz nvidia-smi sampler so each llama-bench invocation carries its own clock and
    temperature record. #28090 reported average SM clock per binary for the same reason."""

    def __init__(self):
        self.rows = []
        self._stop = threading.Event()
        self._t = None

    def start(self):
        self.rows = []
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                v = nvsmi("clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active")
                self.rows.append(v)
            except Exception:
                pass
            self._stop.wait(1.0)

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)
        sm = [float(r[0]) for r in self.rows if r[0].replace('.', '', 1).isdigit()]
        tmp = [float(r[2]) for r in self.rows if r[2].replace('.', '', 1).isdigit()]
        pw = [float(r[3]) for r in self.rows if r[3].replace('.', '', 1).isdigit()]
        def q(xs, p):
            if not xs:
                return None
            xs = sorted(xs)
            return xs[min(len(xs) - 1, int(p * len(xs)))]
        return {
            "samples": len(self.rows),
            "sm_mhz_mean": statistics.fmean(sm) if sm else None,
            "sm_mhz_p10": q(sm, 0.10), "sm_mhz_p50": q(sm, 0.50),
            "temp_c_mean": statistics.fmean(tmp) if tmp else None,
            "temp_c_max": max(tmp) if tmp else None,
            "power_w_mean": statistics.fmean(pw) if pw else None,
        }


# ----------------------------------------------------------------------------- phases

def phase_probe(a):
    name, cc, drv, mem = nvsmi("name,compute_cap,driver_version,memory.total")
    nvcc = sh(f"{a.nvcc} --version | tail -1", capture=True).strip()
    env = {
        "gpu": name, "compute_cap": cc, "driver": drv, "vram_mib": mem, "nvcc": nvcc,
        "host": os.uname().nodename, "when": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pin": PIN, "python": sys.version.split()[0],
        "platform": a.platform,
    }
    (a.results / "env.json").write_text(json.dumps(env, indent=1))
    log(f"probe: {env}")
    return env


def cuda_arch(a):
    if a.cuda_arch:
        return a.cuda_arch
    return nvsmi("compute_cap")[0].replace(".", "")


def phase_build(a):
    arch = cuda_arch(a)
    src = a.work / "llama.cpp"
    if not (src / ".git").exists():
        # Colab runtimes have refused the partial clone (exit 128) where Kaggle accepted it,
        # so try it, then fall back to a shallow fetch of the pinned commit by full sha.
        ok = False
        for attempt in range(3):
            if sh(f"git clone --filter=blob:none {REPO} {src}", log_path=a.logs / "clone.log", check=False) == 0:
                ok = True
                break
            shutil.rmtree(src, ignore_errors=True)
            time.sleep(10)
        if not ok:
            log("partial clone failed three times, shallow-fetching the pinned commit instead")
            src.mkdir(parents=True, exist_ok=True)
            sh(f"git -C {src} init -q && git -C {src} remote add origin {REPO} && "
               f"git -C {src} fetch --depth 1 origin {PIN_FULL} && git -C {src} checkout -q FETCH_HEAD",
               log_path=a.logs / "clone.log")
    if sh(f"git -C {src} cat-file -e {PIN_FULL}^{{commit}}", check=False) != 0:
        sh(f"git -C {src} fetch origin master", log_path=a.logs / "clone.log", check=False)
    sh(f"git -C {src} checkout -q {PIN_FULL}", log_path=a.logs / "clone.log")
    head = sh(f"git -C {src} rev-parse HEAD", capture=True).strip()
    log(f"llama.cpp at {head}")

    def copy_src(name):
        d = a.work / f"src_{name}"
        if not d.exists():
            shutil.copytree(src, d, symlinks=True, ignore=shutil.ignore_patterns(".git"))
        return d

    src_force = copy_src("force_mmq")
    marker = src_force / ".patched"
    if not marker.exists():
        patch_file(src_force / "ggml/src/ggml-cuda/mmvq.cu", FORCE_MMQ_OLD, FORCE_MMQ_NEW)
        # Ada (sm_89), Blackwell and DGX Spark have their own branches whose `default:` case
        # returns the same MMVQ_MAX_BATCH_SIZE; an L4 never reaches the final fall-through, so
        # force_mmq was a no-op there. Flip those defaults too. Dead code on every other card,
        # so the binary means the same thing everywhere: MMQ from ne11 = 2 for any type that
        # has no explicit entry.
        p = src_force / "ggml/src/ggml-cuda/mmvq.cu"
        s = p.read_text()
        inner_old = "            default:\n                return ne11 <= MMVQ_MAX_BATCH_SIZE;"
        inner_new = "            default:\n                return ne11 <= 1;"
        n = s.count(inner_old)
        if n != 4:  # Ada, Blackwell, DGX Spark, CDNA2 at b96806d96 (CDNA1 is indented deeper and untouched)
            raise RuntimeError(f"expected 4 inner default fall-throughs in mmvq.cu, found {n}")
        p.write_text(s.replace(inner_old, inner_new))
        marker.write_text("force_mmq: final fall-through plus the Ada, Blackwell, DGX Spark and CDNA2 branch defaults")
    src_san = copy_src("sanitizer")
    marker = src_san / ".patched"
    if not marker.exists():
        patch_file(src_san / "ggml/src/ggml-cuda/ggml-cuda.cu", POOL_ALLOC_OLD, POOL_ALLOC_NEW)
        patch_file(src_san / "ggml/src/ggml-cuda/ggml-cuda.cu", POOL_FREE_OLD, POOL_FREE_NEW)
        marker.write_text("pool_exact")

    # Kaggle's image has no libcuda.so stub under the toolkit, so CMake never creates the
    # CUDA::cuda_driver target that ggml-cuda links for the VMM pool. Point it at the real
    # driver library instead; falling back to GGML_CUDA_NO_VMM would change the pool the
    # bench binaries run with, which is not the same experiment.
    driver_lib = ""
    stub = pathlib.Path(a.nvcc).resolve().parent.parent / "lib64/stubs/libcuda.so"
    found = sh("ldconfig -p | grep -m1 'libcuda.so' | awk '{print $NF}'", capture=True, check=False).strip()
    import glob as _glob
    globbed = sorted(_glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*") + _glob.glob("/usr/local/nvidia/lib64/libcuda.so*")
                     + _glob.glob("/usr/local/cuda*/lib64/stubs/libcuda.so") + _glob.glob("/usr/local/cuda*/targets/x86_64-linux/lib/stubs/libcuda.so"))
    for cand in [stub, pathlib.Path(found) if found else None] + [pathlib.Path(g) for g in globbed]:
        if cand and cand.exists():
            driver_lib = f"-DCUDA_cuda_driver_LIBRARY={cand}"
            log(f"passing driver library {cand}")
            break
    common = (f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch} -DCMAKE_BUILD_TYPE=Release "
              f"-DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_COMPILER={a.nvcc} "
              f"-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler {driver_lib} {a.cmake_extra}")
    a.vmm_note = "vmm pool (stock)"

    def build(name, srcdir, extra="", targets="llama-bench llama-quantize test-backend-ops"):
        bdir = a.work / f"build_{name}"
        done = bdir / ".done"
        if done.exists():
            log(f"build {name}: cached")
            return bdir
        t0 = time.time()
        log(f"build {name}: configure")
        rc = sh(f"cmake -S {srcdir} -B {bdir} {common} {extra}", log_path=a.logs / f"build_{name}.log", check=False)
        if rc != 0 and "GGML_CUDA_NO_VMM" not in extra:
            # Last resort so a session is not lost: legacy pool. Recorded in binaries.json
            # because it is a deviation from the stock build.
            log(f"build {name}: configure failed, retrying with GGML_CUDA_NO_VMM=ON")
            shutil.rmtree(bdir, ignore_errors=True)
            sh(f"cmake -S {srcdir} -B {bdir} {common} {extra} -DGGML_CUDA_NO_VMM=ON", log_path=a.logs / f"build_{name}.log")
            a.vmm_note = "legacy pool (GGML_CUDA_NO_VMM=ON fallback, configure failed with CUDA::cuda_driver)"
        elif rc != 0:
            raise RuntimeError(f"configure failed for {name}, see {a.logs / f'build_{name}.log'}")
        log(f"build {name}: compile ({a.jobs} jobs)")
        sh(f"cmake --build {bdir} --config Release -j {a.jobs} --target {targets}", log_path=a.logs / f"build_{name}.log")
        done.write_text(f"{time.time() - t0:.0f}s")
        log(f"build {name}: done in {time.time() - t0:.0f}s")
        return bdir

    bins = a.work / "bins"
    bins.mkdir(exist_ok=True)
    if "bench" in a.phases or "models" in a.phases:
        build("selected", src)
        build("force_mmq", src_force)
        build("null_rebuild", src)
    if "sanitize" in a.phases:
        b = build("sanitizer", src_san, extra="-DGGML_CUDA_NO_VMM=ON", targets="test-backend-ops")
        shutil.copy2(b / "bin/test-backend-ops", bins / "test-backend-ops.unfixed")
        marker = src_san / ".fixed"
        if not marker.exists():
            patch_file(src_san / "ggml/src/ggml-cuda/mmq.cu", FIX_OLD, FIX_NEW)
            marker.write_text("pr27044")
            (a.work / "build_sanitizer/.done").unlink(missing_ok=True)
        build("sanitizer", src_san, extra="-DGGML_CUDA_NO_VMM=ON", targets="test-backend-ops")
        shutil.copy2(b / "bin/test-backend-ops", bins / "test-backend-ops.fixed")

    # Record binary hashes so a run can be tied to exactly what it measured.
    hashes = {}
    for name in ["selected", "force_mmq", "null_rebuild"]:
        p = a.work / f"build_{name}/bin/llama-bench"
        if p.exists():
            hashes[name] = sh(f"sha256sum {p}", capture=True).split()[0]
    for name in ["unfixed", "fixed"]:
        p = bins / f"test-backend-ops.{name}"
        if p.exists():
            hashes[f"test-backend-ops.{name}"] = sh(f"sha256sum {p}", capture=True).split()[0]
    (a.results / "binaries.json").write_text(json.dumps({"head": head, "arch": arch, "pool": a.vmm_note, "sha256": hashes}, indent=1))


def phase_models(a):
    mdir = a.work / "models"
    mdir.mkdir(exist_ok=True)
    quant = a.work / "build_selected/bin/llama-quantize"
    for key in a.models:
        url, size = MODELS[key]
        f16 = mdir / f"{key}-f16.gguf"
        if not f16.exists() or f16.stat().st_size != size:
            log(f"download {key} ({size / 1e9:.1f} GB)")
            sh(f"curl -sL --retry 5 -o {f16} {url}")
            if f16.stat().st_size != size:
                raise RuntimeError(f"{f16} is {f16.stat().st_size} bytes, expected {size}")
        for q in QUANTS:
            out = mdir / f"{key}-pure-{q.lower()}.gguf"
            if out.exists():
                continue
            log(f"quantize {key} -> pure {q}")
            sh(f"{quant} --pure --allow-requantize {f16} {out} {q}", log_path=a.logs / "quantize.log")
        if a.delete_f16:
            f16.unlink()


def llama_bench_rows(a, binary, model, sampler):
    """One llama-bench invocation, all ne11 values, returns the parsed JSONL rows."""
    plist = ",".join(str(i) for i in range(1, a.ne11_max + 1))
    cmd = (f"{binary} -m {model} -ngl 999 -fa 1 -p {plist} -n 0 -embd 1 -r {a.reps} -o jsonl "
           f"2>/dev/null")
    sampler.start()
    t0 = time.time()
    out = sh(cmd, capture=True, check=False)
    wall = time.time() - t0
    tele = sampler.stop()
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows, tele, wall, out


def phase_bench(a):
    env = json.loads((a.results / "env.json").read_text())
    binaries = {n: a.work / f"build_{n}/bin/llama-bench" for n in ["selected", "force_mmq", "null_rebuild"]}
    for n, p in binaries.items():
        if not p.exists():
            raise RuntimeError(f"missing {p}")
    models = []
    for key in a.models:
        for q in QUANTS:
            p = a.work / f"models/{key}-pure-{q.lower()}.gguf"
            if p.exists():
                models.append((key, q, p))
    log(f"bench: {len(models)} model files x 3 binaries x {a.rounds} rounds")

    # A quick fit check so a model that does not fit in VRAM is dropped up front, not mid-round.
    fitted = []
    smp = Sampler()
    for key, q, p in models:
        rows, _, _, raw = llama_bench_rows(argparse.Namespace(ne11_max=1, reps=1), binaries["selected"], p, smp)
        if rows:
            fitted.append((key, q, p))
        else:
            log(f"bench: {p.name} did not run, dropping it. Tail of output:\n{raw[-800:]}")
            (a.results / f"dropped_{p.name}.log").write_text(raw)
    models = fitted

    out_path = a.results / f"bench_{a.tag}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["round"], r["binary"], r["model"], r["quant"]))
            except Exception:
                pass
    order_names = list(binaries)
    with open(out_path, "a") as fh:
        for rnd in range(a.rounds):
            rot = order_names[rnd % 3:] + order_names[:rnd % 3]
            for bname in rot:
                for key, q, p in models:
                    if (rnd, bname, key, q) in done:
                        continue
                    if a.cool_to:
                        # Laptop cards throttle within seconds (the RTX 3050 fell from 1519 to
                        # 712 MHz between round 0 and round 1). Start every invocation from the
                        # same thermal state so the binary comparison is not confounded with heat.
                        t_wait = time.time()
                        while float(nvsmi("temperature.gpu")[0]) > a.cool_to and time.time() - t_wait < 600:
                            time.sleep(5)
                    log(f"round {rnd} {bname:12s} {key} {q} (start temp {nvsmi('temperature.gpu')[0]} C)")
                    rows, tele, wall, raw = llama_bench_rows(a, binaries[bname], p, smp)
                    if not rows:
                        log(f"  no rows! tail: {raw[-400:]}")
                        continue
                    for r in rows:
                        rec = {
                            "tag": a.tag, "gpu": env["gpu"], "compute_cap": env["compute_cap"],
                            "round": rnd, "order": rot.index(bname), "binary": bname,
                            "model": key, "quant": q, "ne11": r["n_prompt"],
                            "avg_ts": r["avg_ts"], "stddev_ts": r["stddev_ts"],
                            "avg_ns": r["avg_ns"], "stddev_ns": r["stddev_ns"],
                            "build_commit": r.get("build_commit"), "model_size": r.get("model_size"),
                            "n_ubatch": r.get("n_ubatch"), "flash_attn": r.get("flash_attn"),
                            "wall_s": wall, "telemetry": tele,
                            "when": dt.datetime.now(dt.timezone.utc).isoformat(),
                        }
                        fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    log(f"  ne11=8 -> {[r['avg_ts'] for r in rows if r['n_prompt'] == 8]} t/s, "
                        f"sm {tele['sm_mhz_mean'] and round(tele['sm_mhz_mean'])} MHz, temp {tele['temp_c_max']} C")


SAN_READ = re.compile(r"Invalid __global__ read of size (\d+) bytes")
SAN_KERNEL = re.compile(r"^\s*=+\s+at (?:0x[0-9a-f]+ in )?(\S+)")
SAN_AFTER = re.compile(r"is ([\d,]+) bytes after the nearest allocation at 0x[0-9a-f]+ of size ([\d,]+) bytes")
SAN_BEFORE = re.compile(r"is ([\d,]+) bytes before the nearest allocation")
SAN_SUMMARY = re.compile(r"ERROR SUMMARY: (\d+) error")


def parse_sanitizer(text):
    errors = []
    cur = None
    for line in text.splitlines():
        m = SAN_READ.search(line)
        if m:
            cur = {"size": int(m.group(1)), "kernel": None, "after": None, "alloc": None}
            errors.append(cur)
            continue
        if cur is None:
            continue
        if cur["kernel"] is None:
            m = re.search(r"at (?:0x[0-9a-f]+ in )?(?:void )?([A-Za-z_][\w:]*(?:<[^()]*>)?)\(", line)
            if m:
                cur["kernel"] = m.group(1)
        m = SAN_AFTER.search(line)
        if m:
            cur["after"] = int(m.group(1).replace(",", ""))
            cur["alloc"] = int(m.group(2).replace(",", ""))
    summ = SAN_SUMMARY.search(text)
    kernels = {}
    for e in errors:
        k = e["kernel"] or "?"
        d = kernels.setdefault(k, {"count": 0, "max_after": 0, "allocs": set()})
        d["count"] += 1
        if e["after"] is not None:
            d["max_after"] = max(d["max_after"], e["after"])
            d["allocs"].add(e["alloc"])
    for d in kernels.values():
        d["allocs"] = sorted(d["allocs"])[:20]
    return {
        "reported_errors": len(errors),
        "summary_errors": int(summ.group(1)) if summ else None,
        "by_kernel": kernels,
    }


def phase_sanitize(a):
    env = json.loads((a.results / "env.json").read_text())
    bins = a.work / "bins"
    san = shutil.which("compute-sanitizer") or str(pathlib.Path(a.nvcc).parent / "compute-sanitizer")
    result = {"tag": a.tag, "gpu": env["gpu"], "compute_cap": env["compute_cap"], "runs": {}}
    runenv = dict(os.environ, GGML_CUDA_POOL_EXACT="1")
    for variant in ["unfixed", "fixed"]:
        exe = bins / f"test-backend-ops.{variant}"
        for filt in a.san_filters:
            label = f"{variant}:{filt}"
            logp = a.logs / f"sanitizer_{variant}_{re.sub(r'[^a-z0-9]+', '_', filt)}.log"
            # --padding: memcheck tracks a guard region after every allocation, so an over-read
            # that would otherwise land in a neighbouring live buffer is still reported.
            # --destroy-on-device-error kernel: keep the context alive after a precise error so
            # every test case runs, instead of dying on the first one (the default, context).
            cmd = (f"{san} --tool memcheck --print-limit {a.san_print_limit} --show-backtrace device "
                   f"--padding {a.san_padding} --destroy-on-device-error kernel "
                   f"{exe} test -b CUDA0 -o MUL_MAT_ID -p '{filt}'")
            log(f"sanitize {label}")
            t0 = time.time()
            p = subprocess.run(cmd, shell=True, env=runenv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            logp.write_text(p.stdout)
            parsed = parse_sanitizer(p.stdout)
            plain = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout)
            m = re.search(r"(\d+)/(\d+) tests passed", plain)
            tests_ok = int(m.group(1)) if m else 0
            tests_fail = (int(m.group(2)) - int(m.group(1))) if m else 0
            parsed["sanitizer_unsupported"] = bool(re.search(r"Device not supported|Failed to initialize WDDM", plain))
            parsed.update({"exit": p.returncode, "wall_s": time.time() - t0, "tests_ok": tests_ok, "tests_fail": tests_fail})
            result["runs"][label] = parsed
            log(f"  {parsed['reported_errors']} invalid reads reported, summary={parsed['summary_errors']}, "
                f"tests ok={tests_ok} fail={tests_fail}, {time.time() - t0:.0f}s")
    # Control: the same binary without the exact-pool env, i.e. the stock legacy pool with 5% slack.
    exe = bins / "test-backend-ops.unfixed"
    filt = a.san_filters[0]
    cmd = (f"{san} --tool memcheck --print-limit {a.san_print_limit} --destroy-on-device-error kernel "
           f"{exe} test -b CUDA0 -o MUL_MAT_ID -p '{filt}'")
    log("sanitize unfixed, stock legacy pool, no padding (control)")
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (a.logs / "sanitizer_unfixed_stockpool.log").write_text(p.stdout)
    parsed = parse_sanitizer(p.stdout)
    parsed["exit"] = p.returncode
    result["runs"]["unfixed:stockpool:" + filt] = parsed
    (a.results / f"sanitizer_{a.tag}.json").write_text(json.dumps(result, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--results", default=None, help="results dir (default <workdir>/results)")
    ap.add_argument("--tag", default=None, help="short arch tag, default sm<cc>")
    ap.add_argument("--platform", default=os.environ.get("MMVQ_PLATFORM", "unknown"))
    ap.add_argument("--phases", default="probe,build,models,bench,sanitize")
    ap.add_argument("--models", default="llama3.2-1b,llama3.2-3b,llama3.1-8b")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--ne11-max", type=int, default=16)
    ap.add_argument("--cool-to", type=float, default=None, help="wait until GPU temp <= this (C) before each invocation")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    ap.add_argument("--cuda-arch", default=None)
    ap.add_argument("--cmake-extra", default="")
    ap.add_argument("--delete-f16", action="store_true")
    ap.add_argument("--san-filters", default="type_a=q4_0;type_a=q8_0;type_a=q4_K")
    ap.add_argument("--san-print-limit", type=int, default=3000)
    ap.add_argument("--san-padding", type=int, default=65536)
    a = ap.parse_args()
    a.work = pathlib.Path(a.workdir).expanduser().resolve()
    a.work.mkdir(parents=True, exist_ok=True)
    a.results = pathlib.Path(a.results).expanduser().resolve() if a.results else a.work / "results"
    a.results.mkdir(parents=True, exist_ok=True)
    a.logs = a.work / "logs"
    a.logs.mkdir(exist_ok=True)
    a.phases = [p.strip() for p in a.phases.split(",") if p.strip()]
    a.models = [m.strip() for m in a.models.split(",") if m.strip()]
    a.san_filters = [f.strip() for f in a.san_filters.split(";") if f.strip()]
    if not a.tag:
        a.tag = "sm" + nvsmi("compute_cap")[0].replace(".", "")
    log(f"tag={a.tag} phases={a.phases} work={a.work}")
    for ph in a.phases:
        globals()[f"phase_{ph}"](a)
    log("all phases done")


if __name__ == "__main__":
    main()
