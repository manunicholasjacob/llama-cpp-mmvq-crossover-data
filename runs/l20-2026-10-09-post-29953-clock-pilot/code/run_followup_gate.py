#!/usr/bin/env python3
"""Explicit admission amendment before any performance observations.

Keep the failed v1 gate. A known-bug positive control may abort after memcheck
detects MMQ OOB reads. All fixed-code acceptance requirements stay unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path('/root/issue28090_fixed_pilot_20261009')
sys.path.insert(0, str(ROOT))
import pilot

state = {'started_unix': time.time(), 'status': 'RUNNING', 'stage': 'amended-safety-gate', 'steps': []}

def state_save():
    pilot.save(ROOT / 'followup-execution.json', state)

def run_step(name, cmd, timeout):
    record = {'stage': name, 'command': list(map(str, cmd)), 'started_unix': time.time()}
    state['stage'] = name
    state['steps'].append(record)
    state_save()
    print('START', name, flush=True)
    with (ROOT / ('followup-' + name + '.log')).open('w') as log:
        proc = subprocess.Popen(list(map(str, cmd)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except BaseException:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    record.update(returncode=code, finished_unix=time.time())
    state_save()
    if code:
        raise RuntimeError(f'{name} exited {code}')
    print('DONE', name, flush=True)

try:
    pilot.preflight()
    pilot.check_build()
    assert not (ROOT / 'results').exists(), 'Amendment must precede any performance run'
    assert pilot.sha(ROOT / 'protocol.json') == '13ada6d5682100291d49dd75b4884283727296dbef1583e5a3bc8986195efced'
    old = json.loads((ROOT / 'work/latest-gate.json').read_text())
    assert old['status'] == 'FAILED_OR_INTERRUPTED'
    assert old['build_sha256'] == pilot.sha(ROOT / 'work/build.json')
    assert old['harness_sha256'] == pilot.sha(ROOT / 'dense_gate.cpp')
    assert len(old['commands']) == 1
    parent = old['commands'][0]
    assert parent['world'] == 'parent_force_exact' and parent['returncode'] != 0
    oldlog = ROOT / old['folder'] / 'parent_force_exact.log'
    assert pilot.sha(oldlog) == parent['log_sha256']
    text = oldlog.read_text(errors='replace')
    counts = [int(x) for x in re.findall(r'ERROR SUMMARY: (\d+) errors\b', text)]
    assert counts and max(counts) > 0
    assert 'Invalid __global__ read' in text and 'mul_mat_q<' in text
    folder = ROOT / 'gates' / ('v2-' + str(time.time_ns()))
    folder.mkdir()
    amendment = {
        'schema': 'issue28090-admission-amendment-v2',
        'declared_unix': time.time(),
        'reason': 'v1 positive control detected MMQ OOB reads and then aborted under memcheck; requiring numerical completion of the intentionally faulty control was over-constrained',
        'documentation': 'https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#error-actions',
        'original_protocol_sha256': pilot.sha(ROOT / 'protocol.json'),
        'original_gate_folder': old['folder'],
        'original_gate_json_sha256': pilot.sha(ROOT / old['folder'] / 'gate.json'),
        'positive_control': 'Same-harness immediate parent must report MMQ Invalid __global__ read and nonzero error summary; it may abort or complete',
        'fixed_requirements': 'Unchanged: fixed exact memcheck zero errors and 168/168 numerical; stock selected and cutoff7 each 168/168 numerical',
        'timing_and_analysis': 'Unchanged; no model performance observations exist when this amendment is declared',
        'parent_memcheck_error_count': max(counts),
    }
    pilot.save(folder / 'amendment.json', amendment)
    shutil.copy2(oldlog, folder / 'parent_force_exact.log')
    report = {'status': 'RUNNING', 'protocol_revision': amendment['schema'],
              'build_sha256': pilot.sha(ROOT / 'work/build.json'),
              'harness_sha256': pilot.sha(ROOT / 'dense_gate.cpp'),
              'commands': [parent], 'amendment': amendment}
    try:
        for world in ('fixed_force_exact', 'selected', 'cutoff7'):
            log = folder / (world + '.log')
            cmd = [pilot.binary(world, 'dense-gate')]
            if world == 'fixed_force_exact':
                cmd = ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99',
                       '--padding', '65536', '--target-processes', 'all', *cmd]
            print('CHECK', world, flush=True)
            result = pilot.call(cmd, log, timeout=2400, check=False)
            content = log.read_text(errors='replace')
            numerical = 'DENSE_GATE_RESULT passed=168 total=168 cpu_smoke=0' in content
            zero = 'ERROR SUMMARY: 0 errors' in content
            record = {'world': world, 'returncode': result.returncode, 'numerical_168_pass': numerical,
                      'memcheck_zero': zero, 'log_sha256': pilot.sha(log), 'command': list(map(str, cmd))}
            report['commands'].append(record)
            pilot.save(folder / 'gate.json', report)
            if result.returncode != 0 or not numerical or (world == 'fixed_force_exact' and not zero):
                raise RuntimeError(f'Fixed-code admission failed: {world}')
        report['status'] = 'PASSED_FOCUSED_DENSE_GATE'
    except BaseException as exc:
        report.update(status='FAILED_OR_INTERRUPTED', error=str(exc))
        raise
    finally:
        pilot.save(folder / 'gate.json', report)
        pilot.save(ROOT / 'work/latest-gate.json', {'folder': str(folder.relative_to(ROOT)), **report})
    state['amended_gate'] = str(folder.relative_to(ROOT))
    state_save()
    print('AMENDED_GATE_PASSED', flush=True)
    steps = [
        ('model-download', ['python3', '-u', 'download_models_range.py', '--workers', '8'], 3600),
        ('input-quantizer', ['python3', '-u', 'build_input_quantizer.py'], 1800),
        ('model-prepare', ['python3', '-u', 'prepare_models.py', '--quantizer', 'input-tools/build/bin/llama-quantize'], 1800),
        ('focused-performance', ['python3', '-u', 'pilot.py', 'run'], 4200),
    ]
    for name, cmd, timeout in steps:
        run_step(name, cmd, timeout)
    state['status'] = 'COMPLETE_REQUIRES_INDEPENDENT_REVIEW'
except BaseException as exc:
    state.update(status='FAILED_OR_INTERRUPTED', error=str(exc))
    print('STOP', repr(exc), flush=True)
finally:
    try:
        reset = subprocess.run(['nvidia-smi', '-i', '0', '-rgc'], capture_output=True, text=True, timeout=30)
        state['clock_reset_returncode'] = reset.returncode
        (ROOT / 'followup-clock-reset.log').write_text(reset.stdout + reset.stderr)
    except Exception as exc:
        state['clock_reset_error'] = str(exc)
    state['finished_unix'] = time.time()
    state_save()
    archive = ROOT.parent / 'issue28090-fixed-pilot-v2-evidence.tar.gz'
    files = set(ROOT.glob('server-*')) | set(ROOT.glob('followup-*'))
    for name in ('host-metadata', 'gates', 'results'):
        if (ROOT / name).exists():
            files.add(ROOT / name)
    for pattern in ('work/*.json', 'work/*/*.log', 'work/*/build/CMakeCache.txt', 'models/verified.json', 'input-tools/*.log'):
        files.update(ROOT.glob(pattern))
    for name in ('protocol.json', 'models.lock.json', 'pilot.py', 'dense_gate.cpp', 'CMakeLists.txt', 'source-provenance.json'):
        files.add(ROOT / name)
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(files):
            if path.is_file() or path.is_dir():
                tar.add(path, arcname=str(path.relative_to(ROOT.parent)))
        tar.add(Path(__file__), arcname='run_followup_gate.py')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(archive.suffix + '.sha256').write_text(digest + '  ' + archive.name + '\n')
    print('PRESERVED', archive, digest, flush=True)
