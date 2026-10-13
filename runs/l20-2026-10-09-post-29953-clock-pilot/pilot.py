#!/usr/bin/env python3
"""Single-L20 post-29953 admission study; no third-party Python dependencies."""
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
import time

ROOT = Path(__file__).resolve().parent
FIXED = 'fc9ce6b9d52a8504edcb262abc92737c2289f96c'
PARENT = 'c35b66744f13cb0dcc476af063e112122eee9355'
ARCHIVE_SHA = '733a52df76205ced1285801a32121d35b9d0a2c16db386034bd134f1689391a5'
WORLDS = ('selected', 'cutoff7', 'null_rebuild')
DIAGNOSTICS = ('fixed_force_exact', 'parent_force_exact')
CASES = (('8b', 'Q4_0'), ('8b', 'Q8_0'), ('1b', 'Q8_0'))
NS = (1, 8, 16)
FIELDS = 'timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def call(cmd, log=None, timeout=1800, check=True):
    cmd = list(map(str, cmd))
    print(' '.join(cmd), flush=True)
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, 'w') as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=check)
    return subprocess.run(cmd, timeout=timeout, check=check)


def output(cmd):
    return subprocess.check_output(list(map(str, cmd)), text=True, timeout=30).strip()


def clean_env():
    bad = [k for k in os.environ if k.startswith('GGML_CUDA_') or k in ('CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING')]
    if bad:
        raise RuntimeError(f'Remove CUDA overrides: {bad}')


def exact_pool(text):
    anchor = 'std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device('
    if text.count(anchor) != 1:
        raise RuntimeError('Exact-pool anchor mismatch')
    pool = '''struct pilot_exact_pool : public ggml_cuda_pool {
    int device;
    explicit pilot_exact_pool(int value) : device(value) {}
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
    return text[:brace + 1] + '\n    return std::unique_ptr<ggml_cuda_pool>(new pilot_exact_pool(device));' + text[end:]


def selector(text, cutoff):
    anchor = 'bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {'
    if text.count(anchor) != 1:
        raise RuntimeError('Selector anchor mismatch')
    return text.replace(anchor, anchor + f'''
    if (cc == GGML_CUDA_CC_ADA_LOVELACE &&
        (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0)) {{
        return ne11 <= {cutoff};
    }}''')


def source():
    archive = ROOT / 'assets/llama-fc9ce6b9.tar.gz'
    if sha(archive) != ARCHIVE_SHA:
        raise RuntimeError('Source archive mismatch')
    dest = ROOT / 'work/baseline'
    if not dest.exists():
        dest.mkdir(parents=True)
        with tarfile.open(archive) as t:
            for item in t:
                parts = Path(item.name).parts[1:]
                if not parts:
                    continue
                if Path(item.name).is_absolute() or '..' in parts or not (item.isfile() or item.isdir()):
                    raise RuntimeError('Unsafe source member')
                item.name = str(Path(*parts))
                t.extract(item, dest)
    expected = json.loads((ROOT / 'source.lock.json').read_text())
    actual = {str(p.relative_to(dest)): sha(p) for p in dest.rglob('*') if p.is_file()}
    if actual != expected:
        raise RuntimeError('Extracted baseline changed')
    provenance = json.loads((ROOT / 'source-provenance.json').read_text())
    if provenance['fixed_commit'] != FIXED or provenance['parent_commit'] != PARENT:
        raise RuntimeError('Wrong parent-control provenance')
    for name, info in provenance['parent_blobs'].items():
        if sha(ROOT / 'assets' / ('parent-' + name)) != info['sha256']:
            raise RuntimeError('Parent-control blob changed')
    return dest


def binary(world, name):
    return ROOT / 'work' / world / 'build/bin' / name


def build(args):
    clean_env()
    baseline = source()
    worlds = ('cpu_smoke',) if args.cpu else WORLDS + DIAGNOSTICS
    if not args.cpu and 'V13.0.88' not in output(['nvcc', '--version']):
        raise RuntimeError('Use CUDA toolkit 13.0.88; do not silently change toolchains')
    records = {}
    for world in worlds:
        wd = ROOT / 'work' / world
        sd, bd = wd / 'src', wd / 'build'
        if wd.exists():
            raise RuntimeError(f'Preserve existing work and use a fresh bundle: {wd}')
        shutil.copytree(baseline, sd)
        commit = FIXED
        if world == 'parent_force_exact':
            commit = PARENT
            for name in ('mmq.cu', 'mmq.cuh'):
                shutil.copyfile(ROOT / 'assets' / ('parent-' + name), sd / 'ggml/src/ggml-cuda' / name)
        if world == 'cutoff7' or world in DIAGNOSTICS:
            p = sd / 'ggml/src/ggml-cuda/mmvq.cu'
            p.write_text(selector(p.read_text(), 7 if world == 'cutoff7' else 1))
        if world in DIAGNOSTICS:
            p = sd / 'ggml/src/ggml-cuda/ggml-cuda.cu'
            p.write_text(exact_pool(p.read_text()))
        flags = ['-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF', '-DGGML_BACKEND_DL=OFF',
                 '-DGGML_CCACHE=OFF', '-DGGML_NATIVE=ON', '-DLLAMA_BUILD_TESTS=ON',
                 '-DLLAMA_BUILD_COMMON=ON', '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_EXAMPLES=ON',
                 '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_APP=OFF', '-DLLAMA_BUILD_MTMD=OFF',
                 '-DLLAMA_OPENSSL=OFF', f'-DLLAMA_BUILD_COMMIT={commit}', '-DLLAMA_BUILD_NUMBER=0',
                 f'-DEXPERIMENT_SOURCE={sd}']
        if args.cpu:
            flags += ['-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DGGML_OPENMP=OFF']
        else:
            flags += ['-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=89',
                      '-DGGML_CUDA_NO_VMM=OFF', '-DGGML_CUDA_FA_ALL_QUANTS=OFF',
                      '-DGGML_CUDA_FORCE_MMQ=OFF', '-DGGML_CUDA_FORCE_CUBLAS=OFF',
                      '-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler']
        if world in DIAGNOSTICS:
            flags += ['-DGGML_CUDA_GRAPHS=OFF']
        targets = ['dense-gate'] if args.cpu or world in DIAGNOSTICS else ['dense-gate', 'llama-bench']
        start = time.time()
        call(['cmake', '-S', ROOT, '-B', bd, *flags], wd / 'configure.log')
        call(['cmake', '--build', bd, '--target', *targets, '-j', args.jobs], wd / 'build.log', timeout=5400)
        records[world] = {'commit': commit, 'flags': flags, 'wall_s': time.time() - start,
                         'binaries': {name: sha(binary(world, name)) for name in targets},
                         'sources': {name: sha(sd / 'ggml/src/ggml-cuda' / name)
                                     for name in ('mmvq.cu', 'ggml-cuda.cu', 'mmq.cu', 'mmq.cuh')}}
        save(ROOT / 'work' / ('cpu-build.json' if args.cpu else 'build.json'), records)


def preflight():
    clean_env()
    name = output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).splitlines()
    if name != ['NVIDIA L20']:
        raise RuntimeError(f'Require one exclusive full L20: {name}')
    probe = ROOT / 'work/device_probe'
    probe.parent.mkdir(exist_ok=True)
    if not probe.exists():
        call(['nvcc', ROOT / 'device_probe.cu', '-o', probe])
    device = json.loads(output([probe]))
    if device['cc'] != '8.9' or device['sm_count'] != 92 or device['memory_bytes'] < 44 * 1024**3:
        raise RuntimeError(f'Wrong or partitioned GPU: {device}')
    if output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
        raise RuntimeError('GPU already in use')
    if 'V13.0.88' not in output(['nvcc', '--version']):
        raise RuntimeError('Use CUDA toolkit 13.0.88')
    q = output(['nvidia-smi', '-q'])
    if re.search(r'Virtualization Mode\s*:\s*VGPU', q, re.I):
        raise RuntimeError('vGPU is not valid')
    if any(x in output(['nvidia-smi', f'--query-gpu={FIELDS}', '--format=csv,noheader,nounits']) for x in ('N/A', 'Not Supported')):
        raise RuntimeError('Required telemetry unavailable')
    call(['compute-sanitizer', '--version'])
    try:
        call(['nvidia-smi', '-i', '0', '-lgc', '810,810'])
    finally:
        call(['nvidia-smi', '-i', '0', '-rgc'])
    folder = ROOT / 'host-metadata'
    folder.mkdir(exist_ok=True)
    for name, cmd in {'nvidia-smi.txt': ['nvidia-smi', '-q'], 'nvcc.txt': ['nvcc', '--version'],
                      'sanitizer.txt': ['compute-sanitizer', '--version'], 'gcc.txt': ['gcc', '--version'],
                      'uname.txt': ['uname', '-a'], 'lscpu.txt': ['lscpu']}.items():
        (folder / name).write_text(output(cmd) + '\n')
    components = Path(shutil.which('nvcc')).resolve().parent.parent / 'version.json'
    if components.exists():
        shutil.copy2(components, folder / 'cuda-components.json')
    save(folder / 'device.json', device)
    return device


def check_build():
    records = json.loads((ROOT / 'work/build.json').read_text())
    if set(records) != set(WORLDS + DIAGNOSTICS):
        raise RuntimeError('Incomplete builds')
    for world, record in records.items():
        for name, expected in record['binaries'].items():
            if sha(binary(world, name)) != expected:
                raise RuntimeError(f'Changed binary: {world}/{name}')
    return records


def gate():
    preflight()
    check_build()
    folder = ROOT / 'gates' / str(time.time_ns())
    folder.mkdir(parents=True)
    report = {'status': 'RUNNING', 'build_sha256': sha(ROOT / 'work/build.json'),
              'harness_sha256': sha(ROOT / 'dense_gate.cpp'), 'commands': []}
    try:
        for world in ('parent_force_exact', 'fixed_force_exact', 'selected', 'cutoff7'):
            logfile = folder / (world + '.log')
            cmd = [binary(world, 'dense-gate')]
            if world in DIAGNOSTICS:
                cmd = ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99',
                       '--padding', '65536', '--target-processes', 'all', *cmd]
            result = call(cmd, logfile, timeout=2400, check=False)
            text = logfile.read_text(errors='replace')
            numeric = 'DENSE_GATE_RESULT passed=168 total=168 cpu_smoke=0' in text
            zero = 'ERROR SUMMARY: 0 errors' in text
            record = {'world': world, 'returncode': result.returncode, 'numerical_168_pass': numeric,
                      'memcheck_zero': zero, 'log_sha256': sha(logfile), 'command': list(map(str, cmd))}
            report['commands'].append(record)
            if world == 'parent_force_exact':
                if result.returncode != 99 or 'Invalid __global__ read' not in text or not numeric:
                    raise RuntimeError('Pre-fix positive control did not reproduce a numerical-pass over-read; investigate, do not proceed')
            elif result.returncode != 0 or not numeric or (world in DIAGNOSTICS and not zero):
                raise RuntimeError(f'Admission failed: {world}')
        report['status'] = 'PASSED_FOCUSED_DENSE_GATE'
    except BaseException as e:
        report.update(status='FAILED_OR_INTERRUPTED', error=str(e))
        raise
    finally:
        save(folder / 'gate.json', report)
        save(ROOT / 'work/latest-gate.json', {'folder': str(folder.relative_to(ROOT)), **report})
        print('PRESERVED_GATE', folder, flush=True)


def schedule():
    rows = []
    for r in range(6):
        clocks = ('810', 'default') if r % 2 == 0 else ('default', '810')
        cases = CASES[r % 3:] + CASES[:r % 3]
        worlds = WORLDS[r % 3:] + WORLDS[:r % 3]
        for clock in clocks:
            for model, quant in cases:
                for position, world in enumerate(worlds):
                    for ns in ((8,), (1, 16)):
                        rows.append({'round': r, 'clock': clock, 'model': model, 'quant': quant,
                                     'world': world, 'position': position, 'ns': list(ns)})
    return rows


def bench_args(model, ns, repeats=100):
    return ['-m', model, '-ngl', '999', '-fa', '1', '-n', '0', '-embd', '1', '-t', '4',
            '-b', '2048', '-ub', '512', '-p', ','.join(map(str, ns)), '-r', str(repeats), '-o', 'json']


def validate_rows(rows, ns, repeats, filename):
    if sorted(x.get('n_prompt', -1) for x in rows) != sorted(ns):
        raise RuntimeError('Missing/duplicate/unexpected prompt sizes')
    for row in rows:
        expected = {'n_gen': 0, 'embeddings': True, 'flash_attn': 1, 'n_gpu_layers': 999,
                    'n_ubatch': 512, 'n_batch': 2048, 'n_threads': 4}
        if any(row.get(k) != v for k, v in expected.items()):
            raise RuntimeError('Wrong benchmark settings')
        if not row.get('build_commit') or not FIXED.startswith(row['build_commit']):
            raise RuntimeError('Wrong benchmark source version')
        if Path(row['model_filename']).name != filename:
            raise RuntimeError('Wrong model filename')
        samples = row.get('samples_ts', [])
        if len(samples) != repeats or any(not math.isfinite(x) or x <= 0 for x in samples + [row['avg_ts']]):
            raise RuntimeError('Invalid samples')


def measure(folder, item, model):
    tag = f"r{item['round']}-{item['clock']}-{item['model']}-{item['quant']}-{item['world']}-n{'_'.join(map(str,item['ns']))}"
    telemetry = folder / 'raw' / (tag + '.telemetry.csv')
    with telemetry.open('w') as tf:
        monitor = subprocess.Popen(['nvidia-smi', '-i', '0', f'--query-gpu={FIELDS}',
                                    '--format=csv,noheader,nounits', '-lms', '100'], stdout=tf, stderr=subprocess.PIPE)
        start = time.time()
        try:
            cmd = [binary(item['world'], 'llama-bench'), *bench_args(model, item['ns'])]
            with (folder / 'raw' / (tag + '.json')).open('w') as out, (folder / 'raw' / (tag + '.stderr.log')).open('w') as err:
                subprocess.run(list(map(str, cmd)), stdout=out, stderr=err, check=True, timeout=600)
        finally:
            stopped_early = monitor.poll() is not None
            monitor.terminate()
            try:
                _, monitor_err = monitor.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                _, monitor_err = monitor.communicate()
            save(folder / 'raw' / (tag + '.window.json'), {**item, 'start': start, 'end': time.time()})
        if stopped_early or monitor_err:
            raise RuntimeError('Telemetry failed: ' + monitor_err.decode(errors='replace'))
    rows = json.loads((folder / 'raw' / (tag + '.json')).read_text())
    validate_rows(rows, item['ns'], 100, model.name)
    for row in rows:
        row.update({k: v for k, v in item.items() if k != 'ns'})
        row['n'] = row['n_prompt']
        row['telemetry'] = str(telemetry.relative_to(folder))
    return rows


def gm(xs):
    return math.exp(statistics.fmean(math.log(x) for x in xs))


def estimate(ratios):
    rng = random.Random(0)
    boot = sorted(gm(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    return {'ratio': gm(ratios), 'ci95_low': boot[249], 'ci95_high': boot[9749],
            'wins': sum(x > 1 for x in ratios), 'paired_ratios': ratios}


def reduce_rows(rows):
    expected = {(clock, m, q, r, w, n) for clock in ('810', 'default') for m, q in CASES
                for r in range(6) for w in WORLDS for n in NS}
    values = {}
    for row in rows:
        k = tuple(row[x] for x in ('clock', 'model', 'quant', 'round', 'world', 'n'))
        if k in values or k not in expected or not math.isfinite(row['avg_ts']) or row['avg_ts'] <= 0:
            raise RuntimeError('Duplicate/unexpected/invalid row')
        values[k] = row['avg_ts']
    if set(values) != expected:
        raise RuntimeError(f'Incomplete measurements: {len(values)}/{len(expected)}')
    result = []
    for clock in ('810', 'default'):
        for model, quant in CASES:
            def ratios(w, n):
                return [values[clock, model, quant, r, w, n] / values[clock, model, quant, r, 'selected', n] for r in range(6)]
            floor = max(abs(gm(ratios('null_rebuild', n)) - 1) for n in NS)
            control = max(abs(gm(ratios('cutoff7', n)) - 1) for n in (1, 16))
            effect = estimate(ratios('cutoff7', 8))
            threshold = max(.03, floor, control)
            result.append({'clock': clock, 'model': model, 'quant': quant, **effect,
                           'pilot_null_floor_n1_n8_n16': floor, 'route_control_deviation': control,
                           'control_pass': max(floor, control) <= .02,
                           'material_win': effect['ratio'] >= 1 + threshold and effect['ci95_low'] > 1 + floor and effect['wins'] >= 5,
                           'material_loss': effect['ratio'] <= 1 - threshold and effect['ci95_high'] < 1 - floor and effect['wins'] <= 1})
    return result


def read_clocks(path):
    rows = list(csv.reader(Path(path).read_text().splitlines()))
    parsed = []
    for row in rows:
        if len(row) != 8:
            raise RuntimeError('Invalid telemetry row')
        sm, mem, power, limit, temp, util = map(float, row[1:7])
        if not all(math.isfinite(x) for x in (sm, mem, power, limit, temp, util)) or sm <= 0 or mem <= 0:
            raise RuntimeError('Nonfinite or invalid telemetry')
        if util > 0:
            parsed.append((sm, mem, power, limit, temp, util))
    if len(parsed) < 3:
        raise RuntimeError('Too few busy telemetry samples; do not infer achieved clock')
    return {'busy_samples': len(parsed), 'sm_mean': statistics.fmean(x[0] for x in parsed),
            'mem_mean': statistics.fmean(x[1] for x in parsed),
            'pin_fraction': statistics.fmean(abs(x[0] - 810) <= 15 for x in parsed),
            'temperature_max': max(x[4] for x in parsed)}


def analyze(folder):
    folder = Path(folder)
    rows = [json.loads(x) for x in (folder / 'raw/bench.jsonl').read_text().splitlines()]
    reduced = reduce_rows(rows)
    clock_records = []
    problems = []
    for row in rows:
        if row['n'] != 8:
            continue
        record = {k: row[k] for k in ('clock', 'model', 'quant', 'round', 'world')}
        record.update(read_clocks(folder / row['telemetry']))
        clock_records.append(record)
        if row['clock'] == '810' and record['pin_fraction'] < .90:
            problems.append('Low clock not held: ' + str(record))
        if record['temperature_max'] > 80:
            problems.append('Thermal state requires review: ' + str(record))
    for model, quant in CASES:
        for world in WORLDS:
            groups = {c: [x for x in clock_records if (x['clock'], x['model'], x['quant'], x['world']) == (c, model, quant, world)] for c in ('810', 'default')}
            if any(len(v) != 6 for v in groups.values()):
                raise RuntimeError('Missing clock groups')
            means = {c: statistics.fmean(x['sm_mean'] for x in v) for c, v in groups.items()}
            if means['default'] < 1.5 * means['810']:
                problems.append(f'Insufficient frequency separation: {model}/{quant}/{world}')
            mem = {c: statistics.fmean(x['mem_mean'] for x in v) for c, v in groups.items()}
            if abs(mem['default'] / mem['810'] - 1) > .01:
                problems.append(f'Memory clocks differ: {model}/{quant}/{world}')
    if not all(x['control_pass'] for x in reduced):
        problems.append('Null or unchanged-route controls exceed 2%; do not purchase more GPUs')
    worthwhile = any(x['model'] == '8b' and x['material_win'] for x in reduced)
    decision = 'REVIEW_CANDIDATE_RULE_BEFORE_ANY_SECOND_GPU' if worthwhile else 'STOP_NO_MATERIAL_8B_WIN'
    if problems:
        decision = 'INCONCLUSIVE_FIX_MEASUREMENT_BEFORE_MORE_SPENDING'
    save(folder / 'analysis.json', {'decision': decision, 'problems': problems, 'primary': reduced,
                                   'clocks': clock_records, 'scope': 'Single L20, focused dense gate and three files; no universal rule or release acceptance'})
    return decision


def seal(folder):
    hashes = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json'}
    save(folder / 'SHA256SUMS.json', hashes)
    with tarfile.open(str(folder) + '.tar.gz', 'w:gz') as t:
        t.add(folder, arcname=folder.name)
    print('PRESERVED_RESULT', str(folder) + '.tar.gz', flush=True)


def run():
    device = preflight()
    check_build()
    gate_record = json.loads((ROOT / 'work/latest-gate.json').read_text())
    if (gate_record['status'] != 'PASSED_FOCUSED_DENSE_GATE' or gate_record['build_sha256'] != sha(ROOT / 'work/build.json')
            or gate_record['harness_sha256'] != sha(ROOT / 'dense_gate.cpp')):
        raise RuntimeError('No matching passing gate')
    for command in gate_record['commands']:
        if sha(ROOT / gate_record['folder'] / (command['world'] + '.log')) != command['log_sha256']:
            raise RuntimeError('Gate evidence changed')
    call(['python3', ROOT / 'prepare_models.py'])
    folder = ROOT / 'results' / ('l20-fixed-' + str(time.time_ns()))
    (folder / 'raw').mkdir(parents=True)
    for name in ('protocol.json', 'models.lock.json', 'pilot.py', 'source.lock.json'):
        shutil.copy2(ROOT / name, folder / name)
    shutil.copytree(ROOT / gate_record['folder'], folder / 'gate')
    shutil.copytree(ROOT / 'host-metadata', folder / 'host-metadata')
    save(folder / 'schedule.json', schedule())
    shutil.copy2(ROOT / 'models/verified.json', folder / 'models.verified.json')
    shutil.copy2(ROOT / 'work/build.json', folder / 'build.json')
    for world in WORLDS + DIAGNOSTICS:
        dest = folder / 'build' / world
        dest.mkdir(parents=True)
        for name in ('configure.log', 'build.log'):
            shutil.copy2(ROOT / 'work' / world / name, dest / name)
        shutil.copy2(ROOT / 'work' / world / 'build/CMakeCache.txt', dest / 'CMakeCache.txt')
        for name in ('mmvq.cu', 'ggml-cuda.cu', 'mmq.cu', 'mmq.cuh'):
            shutil.copy2(ROOT / 'work' / world / 'src/ggml/src/ggml-cuda' / name, dest / name)
    state = {'status': 'RUNNING', 'device': device, 'invocations': 0, 'rows': 0, 'source': FIXED}
    lock = json.loads((ROOT / 'models.lock.json').read_text())
    deadline = time.monotonic() + 3600
    try:
        current = None
        with (folder / 'raw/bench.jsonl').open('w') as combined:
            for item in schedule():
                if time.monotonic() > deadline:
                    raise RuntimeError('One-hour timing budget reached; preserve partial evidence')
                phase = (item['round'], item['clock'])
                if phase != current:
                    call(['nvidia-smi', '-i', '0', '-rgc'])
                    if item['clock'] == '810':
                        call(['nvidia-smi', '-i', '0', '-lgc', '810,810'])
                    current = phase
                file = lock[item['model']]['quantized'][item['quant']]['file']
                rows = measure(folder, item, ROOT / 'models' / file)
                for row in rows:
                    combined.write(json.dumps(row) + '\n')
                combined.flush()
                state['invocations'] += 1
                state['rows'] += len(rows)
                save(folder / 'execution.json', state)
                print(f"{state['invocations']}/216 invocations", flush=True)
        state.update(status='COMPLETE_FOCUSED_STUDY_REQUIRES_REVIEW', decision=analyze(folder))
    except BaseException as e:
        state.update(status='FAILED_OR_INTERRUPTED', error=str(e))
        raise
    finally:
        try:
            reset = call(['nvidia-smi', '-i', '0', '-rgc'], timeout=30, check=False)
            state['clock_reset_returncode'] = reset.returncode
        except Exception as e:
            state['clock_reset_error'] = str(e)
        save(folder / 'execution.json', state)
        seal(folder)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['offline-check', 'build', 'preflight', 'gate', 'run', 'analyze', 'seal'])
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--results', type=Path)
    args = p.parse_args()
    if args.jobs < 1:
        raise RuntimeError('jobs must be positive')
    if args.action == 'build':
        build(args)
    elif args.action == 'preflight':
        print(json.dumps(preflight(), indent=2))
    elif args.action == 'gate':
        gate()
    elif args.action == 'run':
        run()
    elif args.action in ('analyze', 'seal'):
        if not args.results:
            p.error('--results is required')
        print(analyze(args.results) if args.action == 'analyze' else seal(args.results))
    else:
        baseline = source()
        selector((baseline / 'ggml/src/ggml-cuda/mmvq.cu').read_text(), 7)
        exact_pool((baseline / 'ggml/src/ggml-cuda/ggml-cuda.cu').read_text())
        print(json.dumps({'status': 'OFFLINE_SOURCE_AND_ANCHORS_PASS', 'invocations': len(schedule()),
                          'expected_rows': 324, 'cuda_execution': 'NOT_RUN'}))


if __name__ == '__main__':
    main()
