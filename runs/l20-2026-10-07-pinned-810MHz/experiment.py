#!/usr/bin/env python3
"""Matched four-world L20 study. No remote credentials or third-party Python code."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import tarfile
import threading
import time

from prepare_models import sha

ROOT = Path(__file__).resolve().parent
COMMIT = 'b96806d96061049a5b574269b049bf6241d63d46'
ARCHIVE_SHA = '265bbeada8caa673e2577bdf4fb2236b10f7fdcdacd5b686d5d47132d01b2a39'
WORLDS = ('selected', 'force_mmq', 'cutoff7', 'null_rebuild')
DIAGNOSTIC = 'force_mmq_exact_pool'
RAW_SELECTOR_SHA = 'f5dac9e161803399281be1a63757afb5847fc590969b4fc4daa3cca6289bd7b9'

def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)

def call(cmd, log=None, env=None):
    cmd = list(map(str, cmd))
    print(' '.join(cmd), flush=True)
    if log:
        with open(log, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=True)
    else:
        subprocess.run(cmd, env=env, check=True)

def output(cmd):
    return subprocess.check_output(list(map(str, cmd)), text=True, timeout=30).strip()

def clean_env():
    bad = [k for k in os.environ if k.startswith('GGML_CUDA_') or k in ('CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING')]
    if bad:
        raise RuntimeError(f'Remove CUDA override variables before running: {bad}')
    return dict(os.environ)

def source():
    if sha(ROOT / 'llama.cpp.tar.gz') != ARCHIVE_SHA:
        raise RuntimeError('Pinned source archive SHA256 mismatch')
    dest = ROOT / 'work' / 'baseline'
    if not dest.exists():
        dest.mkdir(parents=True)
        with tarfile.open(ROOT / 'llama.cpp.tar.gz') as archive:
            for item in archive.getmembers():
                parts = Path(item.name).parts[1:]
                if not parts:
                    continue
                if Path(item.name).is_absolute() or '..' in parts or not (item.isfile() or item.isdir()):
                    raise RuntimeError('Unsafe source archive entry')
                item.name = str(Path(*parts))
                archive.extract(item, dest)
    if sha(dest / 'ggml/src/ggml-cuda/mmvq.cu') != RAW_SELECTOR_SHA:
        raise RuntimeError('Baseline selector mismatch')
    expected = json.loads((ROOT / 'source-files.sha256.json').read_text())
    actual = {str(f.relative_to(dest)): sha(f) for f in dest.rglob('*') if f.is_file()}
    if actual != expected:
        raise RuntimeError('Baseline source files differ from the pinned archive')
    return dest

def exact_pool_patch(text):
    anchor = 'std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device('
    if text.count(anchor) != 1:
        raise RuntimeError('Diagnostic pool anchor mismatch')
    pool = '''struct issue28090_exact_pool : public ggml_cuda_pool {
    int device;
    explicit issue28090_exact_pool(int value) : device(value) {}
    void * alloc(size_t size, size_t * actual_size) override {
        ggml_cuda_set_device(device);
        void * ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        *actual_size = size;
        return ptr;
    }
    void free(void * ptr, size_t) override {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
    }
};

'''
    text = text.replace(anchor, pool + anchor)
    start = text.index(anchor)
    brace = text.index('{', start)
    end = text.index('\n}\n', brace)
    return text[:brace + 1] + '\n    return std::unique_ptr<ggml_cuda_pool>(new issue28090_exact_pool(device));' + text[end:]

def build(args):
    env = clean_env()
    cpu = args.cpu
    if not cpu and 'V13.0.88' not in output(['nvcc', '--version']):
        raise RuntimeError('Install CUDA Toolkit 13.0.88 to match the October L4 run')
    baseline = source()
    worlds = ('cpu_preview',) if cpu else (*WORLDS, DIAGNOSTIC)
    records = {}
    for world in worlds:
        wd = ROOT / 'work' / world
        sd = wd / 'src'
        bd = wd / 'build'
        if bd.exists():
            raise RuntimeError(f'Existing build directory: {bd}; preserve it and use a fresh bundle for a new build')
        shutil.copytree(baseline, sd)
        intervention = 'force_mmq' if world == DIAGNOSTIC else world
        if intervention in ('force_mmq', 'cutoff7'):
            with open(wd / 'patch.log', 'w') as log:
                subprocess.run(['patch', '-p1', '-d', str(sd), '-i', str(ROOT / f'{intervention}.patch')],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        if world == DIAGNOSTIC:
            f = sd / 'ggml/src/ggml-cuda/ggml-cuda.cu'
            f.write_text(exact_pool_patch(f.read_text()))
        flags = ['-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF',
                 '-DGGML_BACKEND_DL=OFF', '-DGGML_NATIVE=ON', '-DGGML_CCACHE=OFF',
                 '-DGGML_CUDA_NO_VMM=OFF', '-DGGML_CUDA_FA_ALL_QUANTS=OFF',
                 '-DGGML_CUDA_FORCE_MMQ=OFF', '-DGGML_CUDA_FORCE_CUBLAS=OFF',
                 '-DLLAMA_BUILD_TESTS=ON', '-DLLAMA_BUILD_EXAMPLES=ON',
                 '-DLLAMA_BUILD_COMMON=ON',
                 '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_SERVER=OFF',
                 '-DLLAMA_BUILD_APP=OFF', '-DLLAMA_BUILD_MTMD=OFF', '-DLLAMA_OPENSSL=OFF',
                 f'-DLLAMA_BUILD_COMMIT={COMMIT}', '-DLLAMA_BUILD_NUMBER=0',
                 f'-DEXPERIMENT_SOURCE={sd}']
        flags += ['-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DGGML_OPENMP=OFF'] if cpu else [
            '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=89',
            '-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler']
        if world == DIAGNOSTIC:
            flags.append('-DGGML_CUDA_GRAPHS=OFF')
        start = time.time()
        call(['cmake', '-S', ROOT, '-B', bd, '-G', 'Ninja', *flags], wd / 'configure.log', env)
        targets = ['dense-gate'] if world == DIAGNOSTIC else ['llama-bench', 'llama-quantize', 'dense-gate']
        call(['cmake', '--build', bd, '--target', *targets, '-j', args.jobs], wd / 'build.log', env)
        records[world] = {'flags': flags, 'wall_s': time.time() - start,
                          'selector_sha256': sha(sd / 'ggml/src/ggml-cuda/mmvq.cu'),
                          'pool_source_sha256': sha(sd / 'ggml/src/ggml-cuda/ggml-cuda.cu'),
                          'binaries': {name: sha(binary(world, name)) for name in targets}}
        print(f'Built {world}', flush=True)
    if not cpu:
        call(['nvcc', ROOT / 'device_probe.cu', '-o', ROOT / 'work/device_probe'])
    save(ROOT / 'work' / ('cpu-build.json' if cpu else 'build.json'), {
        'runtime_commit': COMMIT, 'source_archive_sha256': ARCHIVE_SHA,
        'cuda': None if cpu else output(['nvcc', '--version']), 'worlds': records})

def binary(world, name):
    bd = ROOT / 'work' / world / 'build'
    return bd / 'bin' / name

def preflight():
    clean_env()
    if 'V13.0.88' not in output(['nvcc', '--version']):
        raise RuntimeError('Toolchain differs from the pinned protocol')
    name = output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).splitlines()
    if len(name) != 1 or name[0] != 'NVIDIA L20':
        raise RuntimeError(f'Require one full L20: {name}')
    probe = ROOT / 'work/device_probe'
    if not probe.exists():
        probe.parent.mkdir(exist_ok=True)
        call(['nvcc', ROOT / 'device_probe.cu', '-o', probe])
    device = json.loads(output([probe]))
    # A full 48 GB L20 exposes about 44.39 GiB via totalGlobalMem with this
    # driver/ECC configuration. Keep the full-device 92-SM check and allow
    # CUDA/driver reserved memory; 45 GiB incorrectly rejects this full card.
    if device['cc'] != '8.9' or device['sm_count'] != 92 or device['memory_bytes'] < 44 * 1024**3:
        raise RuntimeError(f'Unexpected L20 properties: {device}')
    if output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
        raise RuntimeError('GPU already in use')
    q = output(['nvidia-smi', '-q'])
    if re.search(r'Virtualization Mode\s*:\s*VGPU', q, re.I) or re.search(r'MIG Mode\s*\n\s*Current\s*:\s*Enabled', q):
        raise RuntimeError('Virtual or partitioned GPU is not valid')
    return device

def artifact_hashes(folder):
    return {str(f.relative_to(folder)): sha(f) for f in sorted(folder.rglob('*'))
            if f.is_file() and f.name != 'SHA256SUMS.json'}

def gate():
    device = preflight()
    folder = ROOT / 'gates' / f'{time.time_ns()}'
    folder.mkdir(parents=True)
    commands = []
    report = {'status': 'RUNNING', 'device': device, 'build_sha256': sha(ROOT / 'work/build.json'),
              'harness_sha256': sha(ROOT / 'dense_gate.cpp'), 'expected_cases_per_command': 168,
              'coverage': 'Q4_0/Q8_0 n2..8, 12 attention/FFN geometries; excludes output head and full-model correctness',
              'commands': commands}
    save(folder / 'gate.json', report)
    try:
        for world in ('selected', 'force_mmq', 'cutoff7', DIAGNOSTIC):
            cmd = [binary(world, 'dense-gate')]
            if world == DIAGNOSTIC:
                cmd = ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99',
                       '--target-processes', 'all', *cmd]
            logfile = folder / f'{world}.log'
            commands.append({'world': world, 'command': list(map(str, cmd)), 'log': logfile.name})
            call(cmd, logfile)
            content = logfile.read_text()
            if 'DENSE_GATE_RESULT passed=168 total=168 cpu_smoke=0' not in content:
                raise RuntimeError('Gate did not execute all expected cases')
            if world == DIAGNOSTIC and 'ERROR SUMMARY: 0 errors' not in content:
                raise RuntimeError('Sanitizer did not confirm zero errors')
        report['status'] = 'PASSED'
    except BaseException as e:
        report.update(status='FAILED', error=str(e))
        raise
    finally:
        save(folder / 'gate.json', report)
        save(folder / 'SHA256SUMS.json', artifact_hashes(folder))
        save(ROOT / 'work/latest-gate.json', {'folder': str(folder.relative_to(ROOT)), **report})

def validate_rows(rows, model_name=None):
    if len(rows) != 16 or sorted(x['n_prompt'] for x in rows) != list(range(1, 17)):
        raise RuntimeError('Missing or duplicate prompt sizes')
    for row in rows:
        expected = {'n_gen': 0, 'embeddings': True, 'flash_attn': 1, 'n_gpu_layers': 999,
                    'n_ubatch': 512, 'n_batch': 2048, 'n_threads': 4}
        if any(row.get(k) != v for k, v in expected.items()):
            raise RuntimeError('Benchmark configuration mismatch')
        if not math.isfinite(row['avg_ts']) or row['avg_ts'] <= 0 or len(row['samples_ts']) != 30:
            raise RuntimeError('Invalid throughput or sample count')
        if any(not math.isfinite(x) or x <= 0 for x in row['samples_ts']):
            raise RuntimeError('Invalid inner sample')
        if not COMMIT.startswith(row.get('build_commit', 'invalid')) or not row.get('build_commit'):
            raise RuntimeError('Benchmark reports the wrong runtime commit')
        if model_name and Path(row['model_filename']).name != model_name:
            raise RuntimeError('Benchmark model mismatch')

def bench_args(model, repeats=30):
    return ['-m', model, '-ngl', '999', '-fa', '1', '-n', '0', '-embd', '1', '-t', '4',
            '-b', '2048', '-ub', '512', '-p', ','.join(map(str, range(1, 17))),
            '-r', str(repeats), '-o', 'json']

TELEMETRY_FIELDS = 'clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active'

def collect_telemetry(path, stop, ready, errors):
    with open(path, 'w') as f:
        while not stop.is_set():
            start = time.time()
            try:
                value = output(['nvidia-smi', f'--query-gpu={TELEMETRY_FIELDS}', '--format=csv,noheader,nounits'])
                f.write(json.dumps({'start': start, 'end': time.time(), 'csv': value}) + '\n')
                f.flush()
                ready.set()
            except Exception as error:
                errors.append(str(error))
                ready.set()
                break
            stop.wait(.1)

def stock_pool_admission(latest, build_info):
    """Admit an explicitly labelled study after real allocator controls.

    This does not turn the failed exact-pool gate into a passing PR gate.
    The performance binaries and their stock pools remain unchanged.
    """
    controls_dir = ROOT / 'allocator-controls'
    controls = json.loads((controls_dir / 'controls.json').read_text())
    if (latest['status'] != 'FAILED' or controls['status'] != 'PASSED' or
            controls['original_diagnostic_gate'] != latest or
            controls['build_sha256'] != sha(ROOT / 'work/build.json') or
            controls['harness_sha256'] != sha(ROOT / 'dense_gate.cpp') or
            not controls['original_diagnostic_source_and_binary_restored']):
        raise RuntimeError('Missing matching allocator controls for exploratory admission')
    gate_dir = ROOT / latest['folder']
    manifest = json.loads((gate_dir / 'SHA256SUMS.json').read_text())
    for world in ('selected', 'force_mmq', 'cutoff7', DIAGNOSTIC):
        name = world + '.log'
        if sha(gate_dir / name) != manifest[name]:
            raise RuntimeError('Original gate evidence changed')
        text = (gate_dir / name).read_text()
        if world == DIAGNOSTIC:
            if 'Invalid __global__ read' not in text:
                raise RuntimeError('Unexpected diagnostic failure; investigate before timing')
        elif 'DENSE_GATE_RESULT passed=168 total=168 cpu_smoke=0' not in text:
            raise RuntimeError('Formal numerical comparison failed')
    for name in ('selected_exact_pool', 'force_mmq_stock_pool'):
        record = controls['controls'][name]
        logfile = controls_dir / record['log']
        content = logfile.read_text()
        if (record['status'] != 'PASSED' or record['exit_code'] != 0 or
                sha(logfile) != record['log_sha256'] or
                'DENSE_GATE_RESULT passed=168 total=168 cpu_smoke=0' not in content or
                'ERROR SUMMARY: 0 errors' not in content):
            raise RuntimeError('Allocator control did not pass all expected checks')
    if controls['controls']['force_mmq_stock_pool']['binary_sha256'] != build_info['worlds']['force_mmq']['binaries']['dense-gate']:
        raise RuntimeError('Default-pool control used a different binary')
    if sha(controls_dir / 'selected-exact-pool.bin') != controls['controls']['selected_exact_pool']['binary_sha256']:
        raise RuntimeError('Selected allocator-control binary changed')
    selector_record = (controls_dir / 'selected-exact-pool.sources.sha256').read_text().splitlines()[0].split()[0]
    if selector_record != RAW_SELECTOR_SHA:
        raise RuntimeError('Allocator control did not use the selected selector')
    return {'status': 'EXPLORATORY_STOCK_POOL_ONLY',
            'strict_exact_pool_gate': 'FAILED',
            'default_pool_memcheck': 'PASSED_168_CASES_ZERO_ERRORS',
            'baseline_exact_pool_memcheck': 'PASSED_168_CASES_ZERO_ERRORS',
            'purpose': 'Requested same-source L4/L20 performance comparison using unchanged stock-pool binaries.',
            'limitation': 'Forced small-n MMQ depends on allocator padding. These timings are not PR correctness acceptance.',
            'controls_sha256': sha(controls_dir / 'controls.json')}

def run_study(exploratory_stock_pool=False):
    device = preflight()
    latest = json.loads((ROOT / 'work/latest-gate.json').read_text())
    if latest['build_sha256'] != sha(ROOT / 'work/build.json') or latest['harness_sha256'] != sha(ROOT / 'dense_gate.cpp'):
        raise RuntimeError('Gate does not match the current build')
    build_info = json.loads((ROOT / 'work/build.json').read_text())
    admission = None
    if exploratory_stock_pool:
        admission = stock_pool_admission(latest, build_info)
    elif latest['status'] != 'PASSED':
        raise RuntimeError('No passing gate for the current build')
    for world, record in build_info['worlds'].items():
        for name, expected in record['binaries'].items():
            if sha(binary(world, name)) != expected:
                raise RuntimeError('Binary changed after build or gate')
    call(['python3', ROOT / 'prepare_models.py'])
    telemetry_probe = output(['nvidia-smi', f'--query-gpu={TELEMETRY_FIELDS}', '--format=csv,noheader,nounits'])
    if 'Not Supported' in telemetry_probe or 'N/A' in telemetry_probe:
        raise RuntimeError('Required telemetry is unavailable')
    folder = ROOT / 'results' / f'l20-810MHz-{time.time_ns()}'
    (folder / 'raw/telemetry').mkdir(parents=True)
    (folder / 'build').mkdir()
    for f in ('protocol.json', 'models.lock.json'):
        shutil.copy2(ROOT / f, folder / f)
    shutil.copytree(ROOT / latest['folder'], folder / 'gate')
    shutil.copy2(ROOT / 'models/verified.json', folder / 'models.verified.json')
    if admission:
        save(folder / 'exploratory-admission.json', admission)
        control_target = folder / 'allocator-controls'
        control_target.mkdir()
        for path in (ROOT / 'allocator-controls').iterdir():
            if path.is_file() and path.suffix in ('.json', '.log', '.sha256'):
                shutil.copy2(path, control_target / path.name)
    metadata = ROOT / 'host-metadata'
    if metadata.exists():
        metadata_target = folder / 'host-metadata'
        metadata_target.mkdir()
        for name in ('cuda-components.json', 'nvcc.txt', 'compute-sanitizer.txt', 'gcc.txt',
                     'uname.txt', 'cuda-libraries.sha256', 'lscpu.txt', 'network.json', 'device.json'):
            if (metadata / name).exists():
                shutil.copy2(metadata / name, metadata_target / name)
    shutil.copy2(Path(__file__), folder / 'experiment.py')
    shutil.copy2(ROOT / 'work/build.json', folder / 'build/build.json')
    for world in (*WORLDS, DIAGNOSTIC):
        for name in ('configure.log', 'build.log'):
            shutil.copy2(ROOT / 'work' / world / name, folder / 'build' / f'{world}-{name}')
        shutil.copy2(ROOT / 'work' / world / 'build/CMakeCache.txt', folder / 'build' / f'{world}-CMakeCache.txt')
        for relative in ('ggml/src/ggml-cuda/mmvq.cu', 'ggml/src/ggml-cuda/ggml-cuda.cu'):
            shutil.copy2(ROOT / 'work' / world / 'src' / relative, folder / 'build' / f'{world}-{Path(relative).name}')
    state = {'status': 'RUNNING', 'device': device, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'host': output(['uname', '-a']), 'telemetry_fields': TELEMETRY_FIELDS, 'target_sm_mhz': 810,
             'clock_scope': 'per invocation, not individual n8 slices', 'invocations_completed': 0}
    if admission:
        state['admission'] = admission
    save(folder / 'execution.json', state)
    (folder / 'nvidia-smi-before.txt').write_text(output(['nvidia-smi', '-q']))
    pinned = False
    lock = json.loads((ROOT / 'models.lock.json').read_text())
    try:
        call(['nvidia-smi', '-pm', '1'])
        call(['nvidia-smi', '-lgc', '810,810'])
        pinned = True
        with open(folder / 'raw/bench.jsonl', 'w') as combined, open(folder / 'raw/windows.jsonl', 'w') as windows:
            for model in ('8b', '3b', '1b'):
                for quant in ('Q4_0', 'Q8_0'):
                    file = lock[model]['quantized'][quant]['file']
                    model_path = ROOT / 'models' / file
                    for world in WORLDS:
                        call([binary(world, 'llama-bench'), *bench_args(model_path, repeats=3)],
                             folder / 'raw' / f'warmup-{model}-{quant}-{world}.log')
                    for round_id in range(6):
                        order = WORLDS[round_id % 4:] + WORLDS[:round_id % 4]
                        for position, world in enumerate(order):
                            tag = f'{model}-{quant}-r{round_id}-{world}'
                            telemetry = folder / 'raw/telemetry' / f'{tag}.jsonl'
                            stop = threading.Event()
                            ready = threading.Event()
                            errors = []
                            thread = threading.Thread(target=collect_telemetry, args=(telemetry, stop, ready, errors))
                            thread.start()
                            if not ready.wait(10) or errors:
                                stop.set()
                                thread.join()
                                raise RuntimeError(f'Telemetry startup failed: {errors}')
                            start = time.time()
                            cmd = [binary(world, 'llama-bench'), *bench_args(model_path)]
                            try:
                                with open(folder / 'raw' / f'{tag}.json', 'w') as out, open(folder / 'raw' / f'{tag}.stderr.log', 'w') as err:
                                    subprocess.run(list(map(str, cmd)), stdout=out, stderr=err, check=True)
                            finally:
                                end = time.time()
                                stop.set()
                                thread.join()
                                windows.write(json.dumps({'tag': tag, 'start': start, 'end': end,
                                    'model': model, 'quant': quant, 'round': round_id, 'binary': world,
                                    'order': position, 'command': list(map(str, cmd))}) + '\n')
                                windows.flush()
                            if errors or not telemetry.stat().st_size:
                                raise RuntimeError(f'Telemetry failed: {errors}')
                            rows = json.loads((folder / 'raw' / f'{tag}.json').read_text())
                            validate_rows(rows, file)
                            for row in rows:
                                row.update(model=model, quant=quant, binary=world, round=round_id,
                                           order=position, ne11=row['n_prompt'], wall_s=end-start,
                                           telemetry=str(telemetry.relative_to(folder)))
                                combined.write(json.dumps(row) + '\n')
                            combined.flush()
                            state['invocations_completed'] += 1
                            save(folder / 'execution.json', state)
                            print(f"{state['invocations_completed']}/144 {tag}", flush=True)
        analyze(folder)
        state['status'] = ('COMPLETE_EXPLORATORY_REQUIRES_CLOCK_AND_PADDING_REVIEW' if admission
                           else 'COMPLETE_REQUIRES_CLOCK_AND_RESULT_REVIEW')
    except BaseException as error:
        state.update(status='FAILED_OR_INTERRUPTED', error=str(error))
        raise
    finally:
        if pinned:
            try:
                subprocess.run(['nvidia-smi', '-rgc'], check=False, timeout=30)
            except Exception as reset_error:
                state['clock_reset_error'] = str(reset_error)
        save(folder / 'execution.json', state)
        try:
            after = output(['nvidia-smi', '-q'])
        except Exception as query_error:
            after = 'Post-run GPU query failed: ' + str(query_error)
        (folder / 'nvidia-smi-after.txt').write_text(after)
        save(folder / 'SHA256SUMS.json', artifact_hashes(folder))
        with tarfile.open(str(folder) + '.tar.gz', 'w:gz') as archive:
            archive.add(folder, arcname=folder.name)
        print(f'PRESERVED_ARTIFACT {folder}.tar.gz', flush=True)

def gm(values):
    return math.exp(statistics.fmean(math.log(x) for x in values))

def reduce_rows(rows):
    expected = {(model, quant, n, round_id, world)
                for model in ('8b', '3b', '1b') for quant in ('Q4_0', 'Q8_0')
                for n in range(1, 17) for round_id in range(6) for world in WORLDS}
    values = {}
    for row in rows:
        key = (row['model'], row['quant'], row['ne11'], row['round'], row['binary'])
        if key in values or key not in expected:
            raise RuntimeError('Duplicate or unexpected measurement')
        value = row['avg_ts']
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError('Invalid throughput')
        values[key] = value
    if set(values) != expected:
        raise RuntimeError(f'Incomplete sweep: {len(values)}/2304 rows')
    result = []
    for model in ('8b', '3b', '1b'):
        for quant in ('Q4_0', 'Q8_0'):
            floor = max(abs(gm([values[model, quant, n, r, 'null_rebuild'] /
                              values[model, quant, n, r, 'selected'] for r in range(6)]) - 1)
                        for n in range(2, 9))
            for world in WORLDS[1:]:
                curve = []
                for n in range(1, 17):
                    ratios = [values[model, quant, n, r, world] / values[model, quant, n, r, 'selected'] for r in range(6)]
                    rng = random.Random(0)
                    boots = sorted(gm(rng.choices(ratios, k=6)) for _ in range(10000))
                    curve.append({'model': model, 'quant': quant, 'world': world, 'n': n,
                                  'paired_gm': gm(ratios), 'ci95_low': boots[249], 'ci95_high': boots[9749],
                                  'wins': sum(x > 1 for x in ratios), 'paired_null_floor_n2_to_8': floor})
                crossing = next((x['n'] for x in curve if 2 <= x['n'] <= 8 and x['paired_gm'] > 1 + floor), None)
                for row in curve:
                    row['first_above_null_floor_n2_to_8'] = crossing
                result.extend(curve)
    return result

def clock_report(folder):
    reports = []
    for path in sorted((folder / 'raw/telemetry').glob('*.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        parsed = [list(csv.reader([row['csv']]))[0] for row in rows]
        busy = [x for x in parsed if float(x[5]) > 0]
        if not busy:
            raise RuntimeError(f'No busy telemetry: {path}')
        sm = [float(x[0]) for x in busy]
        mem = [float(x[1]) for x in busy]
        reports.append({'file': path.name, 'scope': 'invocation-wide busy samples, includes model load and all n',
                        'samples': len(parsed), 'busy_samples': len(busy), 'sm_mean': statistics.fmean(sm),
                        'sm_min': min(sm), 'sm_max': max(sm), 'sm_810_fraction': sum(x == 810 for x in sm) / len(sm),
                        'memory_mean': statistics.fmean(mem), 'power_mean': statistics.fmean(float(x[2]) for x in busy),
                        'temperature_max': max(float(x[4]) for x in busy), 'reason_bitmasks': sorted(set(x[6].strip() for x in busy))})
    if len(reports) != 144:
        raise RuntimeError('Missing telemetry invocations')
    return reports

def analyze(folder):
    rows = [json.loads(line) for line in (folder / 'raw/bench.jsonl').read_text().splitlines()]
    reduced = reduce_rows(rows)
    clocks = clock_report(folder)
    with open(folder / 'summary.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(reduced[0]))
        writer.writeheader()
        writer.writerows(reduced)
    save(folder / 'analysis.json', {'rows': reduced, 'primary_n8': [x for x in reduced if x['n'] == 8],
                                  'status': 'PAIRED_REDUCTION_COMPLETE_REQUIRES_CLOCK_REVIEW',
                                  'bootstrap': 'Python random.Random(0), choices, 10000 resamples, indices 249 and 9749',
                                  'interpretation': 'Six outer rounds; descriptive confidence intervals. Same clock does not isolate SM count.'})
    save(folder / 'clocks.json', clocks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'preflight', 'gate', 'run', 'analyze', 'offline-check'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--exploratory-stock-pool', action='store_true',
                        help='Keep failed exact-pool evidence; require passing real allocator controls; label timings exploratory')
    args = parser.parse_args()
    if args.jobs < 1:
        raise RuntimeError('jobs must be positive')
    if args.action == 'build':
        build(args)
    elif args.action == 'preflight':
        print(json.dumps(preflight(), indent=2))
    elif args.action == 'gate':
        gate()
    elif args.action == 'run':
        run_study(args.exploratory_stock_pool)
    elif args.action == 'analyze':
        analyze(args.results)
    else:
        baseline = source()
        original = (baseline / 'ggml/src/ggml-cuda/ggml-cuda.cu').read_text()
        changed = exact_pool_patch(original)
        assert changed.count('struct issue28090_exact_pool') == 1
        for world in ('force_mmq', 'cutoff7'):
            call(['patch', '--dry-run', '-p1', '-d', baseline, '-i', ROOT / f'{world}.patch'])
        print('Pinned source and patch anchors verified. CUDA execution is not tested locally.')

if __name__ == '__main__':
    main()
