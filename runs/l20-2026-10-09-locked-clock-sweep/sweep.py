#!/usr/bin/env python3
"""L20 n=8 clock sweep on llama.cpp fc9ce6b9. No third-party Python dependencies."""
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
ARCHIVE_SHA = '733a52df76205ced1285801a32121d35b9d0a2c16db386034bd134f1689391a5'
WORLDS = ('selected', 'cutoff7', 'null_rebuild')
PRIMARY = (('8b', 'Q4_0'), ('8b', 'Q8_0'))
GRID = (1200, 1500, 1800, 2100)
HIGH_CANDIDATES = (2520, 2460, 2400, 2310, 2250)
PIN_TOLERANCE_MHZ = 15
PROBE_REPEATS = 30
REPEATS = 100
TIMING_BUDGET_S = 150 * 60
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
    return subprocess.check_output(list(map(str, cmd)), text=True, timeout=60).strip()


def clean_env():
    bad = [k for k in os.environ if k.startswith('GGML_CUDA_') or k in ('CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING')]
    if bad:
        raise RuntimeError(f'Remove CUDA overrides: {bad}')


def rotate(seq, k):
    seq = tuple(seq)
    k %= len(seq)
    return seq[k:] + seq[:k]


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
    return dest


def binary(world):
    return ROOT / 'work' / world / 'build/bin/llama-bench'


def build(args):
    clean_env()
    baseline = source()
    if 'V13.0.88' not in output(['nvcc', '--version']):
        raise RuntimeError('Use CUDA toolkit 13.0.88; do not silently change toolchains')
    records = {}
    for world in WORLDS:
        wd = ROOT / 'work' / world
        sd, bd = wd / 'src', wd / 'build'
        if wd.exists():
            raise RuntimeError(f'Preserve existing work and use a fresh bundle: {wd}')
        shutil.copytree(baseline, sd)
        if world == 'cutoff7':
            p = sd / 'ggml/src/ggml-cuda/mmvq.cu'
            p.write_text(selector(p.read_text(), 7))
        flags = ['-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF', '-DGGML_BACKEND_DL=OFF',
                 '-DGGML_CCACHE=OFF', '-DGGML_NATIVE=ON', '-DLLAMA_BUILD_TESTS=ON',
                 '-DLLAMA_BUILD_COMMON=ON', '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_EXAMPLES=ON',
                 '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_APP=OFF', '-DLLAMA_BUILD_MTMD=OFF',
                 '-DLLAMA_OPENSSL=OFF', f'-DLLAMA_BUILD_COMMIT={FIXED}', '-DLLAMA_BUILD_NUMBER=0',
                 '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=89',
                 '-DGGML_CUDA_NO_VMM=OFF', '-DGGML_CUDA_FA_ALL_QUANTS=OFF',
                 '-DGGML_CUDA_FORCE_MMQ=OFF', '-DGGML_CUDA_FORCE_CUBLAS=OFF',
                 '-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler']
        start = time.time()
        call(['cmake', '-S', sd, '-B', bd, *flags], wd / 'configure.log')
        call(['cmake', '--build', bd, '--target', 'llama-bench', '-j', args.jobs], wd / 'build.log', timeout=5400)
        records[world] = {'commit': FIXED, 'flags': flags, 'wall_s': time.time() - start,
                         'binaries': {'llama-bench': sha(binary(world))},
                         'sources': {name: sha(sd / 'ggml/src/ggml-cuda' / name)
                                     for name in ('mmvq.cu', 'ggml-cuda.cu', 'mmq.cu', 'mmq.cuh')}}
        save(ROOT / 'work/build.json', records)


def preflight():
    clean_env()
    name = [x.strip() for x in output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader']).splitlines() if x.strip()]
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
    try:
        call(['nvidia-smi', '-i', '0', '-lgc', '1200,1200'])
    finally:
        call(['nvidia-smi', '-i', '0', '-rgc'])
    folder = ROOT / 'host-metadata'
    folder.mkdir(exist_ok=True)
    for item, cmd in {'nvidia-smi.txt': ['nvidia-smi', '-q'], 'nvcc.txt': ['nvcc', '--version'],
                      'gcc.txt': ['gcc', '--version'], 'uname.txt': ['uname', '-a'], 'lscpu.txt': ['lscpu']}.items():
        (folder / item).write_text(output(cmd) + '\n')
    components = Path(shutil.which('nvcc')).resolve().parent.parent / 'version.json'
    if components.exists():
        shutil.copy2(components, folder / 'cuda-components.json')
    save(folder / 'device.json', device)
    return device


def check_build():
    records = json.loads((ROOT / 'work/build.json').read_text())
    if set(records) != set(WORLDS):
        raise RuntimeError('Incomplete builds')
    for world, record in records.items():
        if sha(binary(world)) != record['binaries']['llama-bench']:
            raise RuntimeError(f'Changed binary: {world}')
    return records


def schedule(clocks):
    clocks = tuple(sorted(int(c) for c in clocks))
    if len(clocks) < 2 or len(set(clocks)) != len(clocks):
        raise RuntimeError(f'Need at least two distinct clocks: {clocks}')
    ends = {clocks[0], clocks[-1]}
    rows = []
    for r in range(6):
        worlds = rotate(WORLDS, r)
        models = rotate(PRIMARY, r)
        for clock in rotate(clocks, r):
            for model, quant in models:
                for position, world in enumerate(worlds):
                    rows.append({'phase': 'primary', 'round': r, 'clock': clock, 'model': model,
                                 'quant': quant, 'world': world, 'position': position, 'ns': [8]})
            if r == 0 and clock in ends:
                for model, quant in PRIMARY:
                    for position, world in enumerate(worlds):
                        rows.append({'phase': 'route_control', 'round': r, 'clock': clock, 'model': model,
                                     'quant': quant, 'world': world, 'position': position, 'ns': [1, 16]})
    return rows


def schedule_1b(clocks):
    clocks = tuple(sorted(set(int(c) for c in clocks)))
    rows = []
    for r in range(6):
        worlds = rotate(WORLDS, r)
        for clock in rotate(clocks, r):
            for position, world in enumerate(worlds):
                rows.append({'phase': 'small_model', 'round': r, 'clock': clock, 'model': '1b',
                             'quant': 'Q8_0', 'world': world, 'position': position, 'ns': [8]})
    return rows


def bench_args(model, ns, repeats):
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


def set_lock(mhz):
    call(['nvidia-smi', '-i', '0', '-rgc'])
    call(['nvidia-smi', '-i', '0', '-lgc', f'{int(mhz)},{int(mhz)}'])
    time.sleep(2)


def read_clocks(path, target):
    parsed = []
    for row in csv.reader(Path(path).read_text().splitlines()):
        if len(row) != 8:
            raise RuntimeError('Invalid telemetry row')
        sm, mem, power, limit, temp, util = map(float, row[1:7])
        if not all(math.isfinite(x) for x in (sm, mem, power, limit, temp, util)) or sm <= 0 or mem <= 0:
            raise RuntimeError('Nonfinite or invalid telemetry')
        if util > 0:
            parsed.append((sm, mem, power, limit, temp))
    if len(parsed) < 3:
        raise RuntimeError('Too few busy telemetry samples; do not infer achieved clock')
    sms = [x[0] for x in parsed]
    return {'busy_samples': len(parsed), 'sm_mean': statistics.fmean(sms),
            'sm_min': min(sms), 'sm_max': max(sms),
            'mem_mean': statistics.fmean(x[1] for x in parsed),
            'held': all(abs(x - target) <= PIN_TOLERANCE_MHZ for x in sms),
            'temperature_max': max(x[4] for x in parsed)}


def measure(folder, item, model, repeats):
    tag = f"r{item['round']}-{item['clock']}-{item['model']}-{item['quant']}-{item['world']}-n{'_'.join(map(str, item['ns']))}"
    if item['phase'] != 'primary':
        tag = item['phase'] + '-' + tag
    telemetry = folder / 'raw' / (tag + '.telemetry.csv')
    with telemetry.open('w') as tf:
        monitor = subprocess.Popen(['nvidia-smi', '-i', '0', f'--query-gpu={FIELDS}',
                                    '--format=csv,noheader,nounits', '-lms', '100'], stdout=tf, stderr=subprocess.PIPE)
        start = time.time()
        try:
            cmd = [binary(item['world']), *bench_args(model, item['ns'], repeats)]
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
    validate_rows(rows, item['ns'], repeats, model.name)
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
            'wins': sum(x > 1 for x in ratios), 'losses': sum(x < 1 for x in ratios),
            'paired_ratios': ratios}


def classify_cell(effect, floor):
    threshold = max(.03, floor)
    control_pass = floor <= .02
    win = control_pass and effect['ratio'] >= 1 + threshold and effect['ci95_low'] > 1 and effect['wins'] == 6
    loss = control_pass and effect['ratio'] <= 1 - threshold and effect['ci95_high'] < 1 and effect['losses'] == 6
    return {'pilot_null_floor': floor, 'control_pass': control_pass,
            'material_win': win, 'material_loss': loss, **effect}


def reduce_primary(rows, clocks):
    clocks = tuple(sorted(set(clocks)))
    expected = {(c, m, q, r, w) for c in clocks for m, q in PRIMARY for r in range(6) for w in WORLDS}
    values = {}
    for row in rows:
        if row.get('phase') != 'primary' or row.get('n') != 8:
            continue
        k = tuple(row[x] for x in ('clock', 'model', 'quant', 'round', 'world'))
        if k in values or k not in expected or not math.isfinite(row['avg_ts']) or row['avg_ts'] <= 0:
            raise RuntimeError('Duplicate/unexpected/invalid primary row')
        values[k] = row['avg_ts']
    if set(values) != expected:
        raise RuntimeError(f'Incomplete primary measurements: {len(values)}/{len(expected)}')
    result = []
    for clock in clocks:
        for model, quant in PRIMARY:
            def ratios(world):
                return [values[clock, model, quant, r, world] / values[clock, model, quant, r, 'selected'] for r in range(6)]
            floor = abs(gm(ratios('null_rebuild')) - 1)
            result.append({'clock': clock, 'model': model, 'quant': quant, **classify_cell(estimate(ratios('cutoff7')), floor)})
    return result


def reduce_controls(rows, clocks):
    ends = {min(clocks), max(clocks)}
    found = {}
    for row in rows:
        if row.get('phase') != 'route_control':
            continue
        k = (row['clock'], row['model'], row['quant'], row['world'], row['n'])
        if row['clock'] not in ends or k in found:
            raise RuntimeError('Unexpected route-control row')
        found[k] = row['avg_ts']
    deviations = []
    for clock in sorted(ends):
        for model, quant in PRIMARY:
            for n in (1, 16):
                for world in ('cutoff7', 'null_rebuild'):
                    ratio = found[clock, model, quant, world, n] / found[clock, model, quant, 'selected', n]
                    deviations.append({'clock': clock, 'model': model, 'quant': quant, 'world': world,
                                       'n': n, 'ratio': ratio, 'deviation': abs(ratio - 1)})
    if not deviations:
        raise RuntimeError('Missing route controls')
    worst = max(x['deviation'] for x in deviations)
    return {'deviations': deviations, 'worst': worst, 'pass': worst <= .02}


def crossover(points):
    points = sorted(points, key=lambda x: x['clock'])
    if len(points) < 2:
        return {'status': 'INSUFFICIENT_VALID_CLOCKS', 'transitions': []}
    transitions = []
    for left, right in zip(points, points[1:]):
        if left['material_win'] and not right['material_win']:
            transitions.append({'last_win_mhz': left['clock'], 'first_nonwin_mhz': right['clock']})
        elif not left['material_win'] and right['material_win']:
            transitions.append({'upward_reversal': True, 'from': left['clock'], 'to': right['clock']})
    if any(item.get('upward_reversal') for item in transitions):
        return {'status': 'NONMONOTONIC', 'transitions': transitions}
    if len(transitions) > 1:
        return {'status': 'MULTIPLE_TRANSITIONS', 'transitions': transitions}
    if len(transitions) == 1:
        return {'status': 'CROSSED', 'transitions': transitions, **transitions[0]}
    if all(item['material_win'] for item in points):
        return {'status': 'STILL_WINNING_AT_HIGHEST', 'highest_mhz': points[-1]['clock'], 'transitions': []}
    return {'status': 'NO_WIN_IN_RANGE', 'lowest_mhz': points[0]['clock'], 'transitions': []}


def boundary_clocks(reports, valid):
    chosen = []
    for report in reports.values():
        if report['status'] == 'CROSSED':
            chosen += [report['last_win_mhz'], report['first_nonwin_mhz']]
        else:
            for item in report['transitions']:
                chosen += [item.get('last_win_mhz'), item.get('first_nonwin_mhz'), item.get('from'), item.get('to')]
            if report['status'] in ('STILL_WINNING_AT_HIGHEST', 'NO_WIN_IN_RANGE'):
                chosen += [valid[0], valid[-1]]
    return sorted({int(x) for x in chosen if x is not None})


def reduce_1b(rows, clocks):
    clocks = tuple(sorted(set(clocks)))
    expected = {(c, r, w) for c in clocks for r in range(6) for w in WORLDS}
    values = {}
    for row in rows:
        if row.get('phase') != 'small_model' or row.get('n') != 8:
            continue
        k = (row['clock'], row['round'], row['world'])
        if k in values or (row['clock'], row['round'], row['world']) not in expected:
            raise RuntimeError('Duplicate/unexpected 1B row')
        values[k] = row['avg_ts']
    if set(values) != expected:
        raise RuntimeError(f'Incomplete 1B measurements: {len(values)}/{len(expected)}')
    result = []
    for clock in clocks:
        ratios = lambda world: [values[clock, r, world] / values[clock, r, 'selected'] for r in range(6)]
        floor = abs(gm(ratios('null_rebuild')) - 1)
        result.append({'clock': clock, 'model': '1b', 'quant': 'Q8_0', **classify_cell(estimate(ratios('cutoff7')), floor)})
    return result


def pin_table(folder, rows, target_for):
    records = []
    problems = []
    for row in rows:
        if row['n'] != 8:
            continue
        record = {k: row[k] for k in ('phase', 'clock', 'model', 'quant', 'round', 'world')}
        record.update(read_clocks(folder / row['telemetry'], target_for(row)))
        records.append(record)
        if not record['held']:
            problems.append('Lock not held: ' + str(record))
        if record['temperature_max'] > 80:
            problems.append('Thermal state requires review: ' + str(record))
    return records, problems


def analyze(folder):
    folder = Path(folder)
    rows = [json.loads(x) for x in (folder / 'raw/bench.jsonl').read_text().splitlines()]
    probed = json.loads((folder / 'probe.json').read_text())['accepted_mhz']
    primary = reduce_primary(rows, probed)
    controls = reduce_controls(rows, probed)
    clocks, clock_problems = pin_table(folder, rows, lambda row: row['clock'])
    valid = []
    for mhz in probed:
        cells = [x for x in primary if x['clock'] == mhz]
        samples = [x for x in clocks if x['phase'] == 'primary' and x['clock'] == mhz]
        if cells and samples and all(x['control_pass'] for x in cells) and all(x['held'] for x in samples):
            valid.append(mhz)
    reports = {}
    for quant in ('Q4_0', 'Q8_0'):
        points = [x for x in primary if x['quant'] == quant and x['clock'] in valid]
        reports[quant] = crossover(points)
    lock_failures = [x for x in clock_problems if x.startswith('Lock not held')]
    problems = [x for x in clock_problems if x.startswith('Thermal')]
    if not controls['pass']:
        problems.append(f"Route control deviation {controls['worst']:.4f} exceeds 2%")
    mem = {}
    for mhz in valid:
        group = [x['mem_mean'] for x in clocks if x['clock'] == mhz and x['phase'] == 'primary']
        if group:
            mem[mhz] = statistics.fmean(group)
    if mem and max(mem.values()) / min(mem.values()) - 1 > .01:
        problems.append(f'Memory clocks differ across locks: {mem}')
    small = []
    small_problems = []
    if any(x.get('phase') == 'small_model' for x in rows):
        planned = sorted({x['clock'] for x in rows if x.get('phase') == 'small_model'})
        small = reduce_1b(rows, planned)
        for cell in small:
            if not cell['control_pass']:
                small_problems.append(f"1B null floor failed at {cell['clock']}")
        for record in clocks:
            if record['phase'] == 'small_model' and not record['held']:
                small_problems.append('1B lock not held: ' + str({k: record[k] for k in ('clock', 'round', 'world', 'sm_min', 'sm_max')}))
    if 1200 not in valid or len(valid) < 3:
        problems.append(f'Need 1200 MHz and at least three valid locks, got {valid}')
    decision = 'INCONCLUSIVE_PIN_OR_CONTROL' if problems else 'CURVE_READY_FOR_REVIEW'
    if decision == 'CURVE_READY_FOR_REVIEW' and small_problems:
        decision = 'CURVE_READY_1B_INCOMPLETE'
    save(folder / 'analysis.json', {
        'decision': decision, 'problems': problems, 'excluded_lock_failures': lock_failures,
        'small_model_problems': small_problems, 'valid_mhz': valid, 'primary': primary,
        'route_controls': controls, 'crossover': reports,
        'boundary_mhz': boundary_clocks(reports, valid) if valid else [],
        'small_model': small, 'clocks': clocks,
        'prior_anchor': {'mhz': 810, 'commit': FIXED, 'remeasured': False,
                         'note': '8B Q4_0/Q8_0 gains at 810 MHz are the 9 Oct result. Do not pool those rounds into this curve.'},
        'scope': 'One L20. n=8 dense Q4_0/Q8_0 clock sweep. No static-table change and no portable rule by itself.'})
    return decision


def probe_lock(folder, mhz, model):
    item_base = {'phase': 'probe', 'round': 0, 'clock': mhz, 'model': '8b', 'quant': 'Q4_0', 'ns': [8]}
    reports = []
    try:
        set_lock(mhz)
        for world in ('selected', 'cutoff7'):
            rows = measure(folder, {**item_base, 'world': world, 'position': 0}, model, PROBE_REPEATS)
            record = read_clocks(folder / rows[0]['telemetry'], mhz)
            record.update(world=world, avg_ts=rows[0]['avg_ts'])
            reports.append(record)
    except Exception as e:
        return {'mhz': mhz, 'accepted': False, 'error': str(e), 'worlds': reports}
    return {'mhz': mhz, 'accepted': all(x['held'] for x in reports), 'worlds': reports}


def choose_clocks(probe_rows):
    accepted = {row['mhz'] for row in probe_rows if row['accepted']}
    if 1200 not in accepted:
        raise RuntimeError('1200 MHz did not hold on both binaries; stop, the lock or the load is not usable')
    chosen = [mhz for mhz in GRID if mhz in accepted]
    high = next((mhz for mhz in HIGH_CANDIDATES if mhz in accepted), None)
    if high and high not in chosen:
        chosen.append(high)
    if len(chosen) < 3:
        raise RuntimeError(f'Fewer than three locks held: {chosen}')
    return sorted(chosen)


def execute(folder, items, repeats, state, deadline):
    lock = json.loads((ROOT / 'models.lock.json').read_text())
    current = None
    with (folder / 'raw/bench.jsonl').open('a') as combined:
        for item in items:
            if time.monotonic() > deadline:
                raise RuntimeError('Timing budget reached; preserve partial evidence and do not stitch a later rerun')
            if item['clock'] != current:
                set_lock(item['clock'])
                current = item['clock']
            file = lock[item['model']]['quantized'][item['quant']]['file']
            for row in measure(folder, item, ROOT / 'models' / file, repeats):
                combined.write(json.dumps(row) + '\n')
            combined.flush()
            state['invocations'] += 1
            save(folder / 'execution.json', state)
            print(f"{state['invocations']}/{state['expected_invocations']} invocations", flush=True)


def run():
    device = preflight()
    check_build()
    call(['python3', ROOT / 'prepare_models.py'])
    folder = ROOT / 'results' / ('l20-sweep-' + str(time.time_ns()))
    (folder / 'raw').mkdir(parents=True)
    for name in ('protocol.json', 'models.lock.json', 'sweep.py', 'source.lock.json'):
        shutil.copy2(ROOT / name, folder / name)
    shutil.copytree(ROOT / 'host-metadata', folder / 'host-metadata')
    shutil.copy2(ROOT / 'models/verified.json', folder / 'models.verified.json')
    shutil.copy2(ROOT / 'work/build.json', folder / 'build.json')
    for world in WORLDS:
        dest = folder / 'build' / world
        dest.mkdir(parents=True)
        for name in ('configure.log', 'build.log'):
            shutil.copy2(ROOT / 'work' / world / name, dest / name)
        shutil.copy2(ROOT / 'work' / world / 'build/CMakeCache.txt', dest / 'CMakeCache.txt')
        for name in ('mmvq.cu', 'ggml-cuda.cu', 'mmq.cu', 'mmq.cuh'):
            shutil.copy2(ROOT / 'work' / world / 'src/ggml/src/ggml-cuda' / name, dest / name)
    model = ROOT / 'models' / json.loads((ROOT / 'models.lock.json').read_text())['8b']['quantized']['Q4_0']['file']
    state = {'status': 'RUNNING', 'device': device, 'invocations': 0, 'source': FIXED}
    deadline = time.monotonic() + TIMING_BUDGET_S
    try:
        probe_rows = []
        for mhz in (*GRID, *HIGH_CANDIDATES):
            if time.monotonic() > deadline:
                raise RuntimeError('Timing budget reached during lock probe')
            print(f'PROBE {mhz}', flush=True)
            probe_rows.append(probe_lock(folder, mhz, model))
            save(folder / 'probe.json', {'rows': probe_rows})
            if mhz == 1200 and not probe_rows[-1]['accepted']:
                raise RuntimeError('1200 MHz did not hold on both selected and cutoff7')
            if mhz not in GRID and probe_rows[-1]['accepted']:
                break
        accepted = choose_clocks(probe_rows)
        save(folder / 'probe.json', {'rows': probe_rows, 'accepted_mhz': accepted,
                                    'rejected_mhz': [r['mhz'] for r in probe_rows if not r['accepted']]})
        primary = schedule(accepted)
        state['expected_invocations'] = len(primary) + len(schedule_1b(accepted))
        save(folder / 'schedule.json', primary)
        execute(folder, primary, REPEATS, state, deadline)
        decision = analyze(folder)
        analysis = json.loads((folder / 'analysis.json').read_text())
        if decision != 'CURVE_READY_FOR_REVIEW':
            state.update(status='COMPLETE_8B_INCONCLUSIVE', decision=decision)
            return
        small_clocks = analysis['boundary_mhz'] or [accepted[0], accepted[-1]]
        small = schedule_1b(small_clocks)
        state['expected_invocations'] = state['invocations'] + len(small)
        save(folder / 'schedule-1b.json', small)
        execute(folder, small, REPEATS, state, deadline)
        state.update(status='COMPLETE_SWEEP_REQUIRES_REVIEW', decision=analyze(folder))
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


def seal(folder):
    folder = Path(folder)
    hashes = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS.json'}
    save(folder / 'SHA256SUMS.json', hashes)
    with tarfile.open(str(folder) + '.tar.gz', 'w:gz') as t:
        t.add(folder, arcname=folder.name)
    print('PRESERVED_RESULT', str(folder) + '.tar.gz', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['offline-check', 'build', 'preflight', 'run', 'analyze', 'seal'])
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--results', type=Path)
    args = p.parse_args()
    if args.jobs < 1:
        raise RuntimeError('jobs must be positive')
    if args.action == 'build':
        build(args)
    elif args.action == 'preflight':
        print(json.dumps(preflight(), indent=2))
    elif args.action == 'run':
        run()
    elif args.action in ('analyze', 'seal'):
        if not args.results:
            p.error('--results is required')
        print(analyze(args.results) if args.action == 'analyze' else seal(args.results))
    else:
        protocol = json.loads((ROOT / 'protocol.json').read_text())
        if tuple(protocol['grid_mhz']) != GRID or tuple(protocol['high_candidates_mhz']) != HIGH_CANDIDATES:
            raise RuntimeError('protocol.json does not match the runner')
        baseline = source()
        selector((baseline / 'ggml/src/ggml-cuda/mmvq.cu').read_text(), 7)
        clocks = (*GRID, HIGH_CANDIDATES[-1])
        print(json.dumps({'status': 'OFFLINE_SOURCE_AND_SCHEDULE_PASS',
                          'primary_invocations_for_five_clocks': len(schedule(clocks)),
                          'cuda_execution': 'NOT_RUN'}))


if __name__ == '__main__':
    main()
