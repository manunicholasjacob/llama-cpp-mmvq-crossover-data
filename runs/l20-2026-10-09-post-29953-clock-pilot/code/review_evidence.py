#!/usr/bin/env python3
"""Independent CPU-only review of the public post-29953 pilot evidence.

SPDX-License-Identifier: MIT
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def gm(values):
    return math.exp(statistics.mean(map(math.log, values)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--out', type=Path, help='Optional JSON output outside the sealed input')
    args = parser.parse_args()
    root = args.folder
    manifest = json.loads((root / 'SHA256SUMS.json').read_text())
    provenance = json.loads((root / 'PUBLIC-PROVENANCE.json').read_text())
    assert sha(root / 'SHA256SUMS.json') == provenance['original_manifest_sha256']
    assert len(manifest) == provenance['original_collected_files'] == 923
    assert len(provenance['redactions']) == 7
    for name, digest in manifest.items():
        redaction = provenance['redactions'].get(name)
        if redaction:
            assert redaction['original_sha256'] == digest
            digest = redaction['public_sha256']
        assert sha(root / name) == digest, name
    shared = {}
    for line in (root / 'SHARE_SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert name not in shared and sha(root / name) == digest, name
        shared[name] = digest
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'SHARE_SHA256SUMS'}
    assert actual == set(shared), 'Public checksum membership differs'
    gate = json.loads((root / 'gate/gate.json').read_text())
    assert gate['status'] == 'PASSED_FOCUSED_DENSE_GATE'
    assert gate['protocol_revision'] == 'issue28090-admission-amendment-v2'
    assert gate['build_sha256'] == sha(root / 'build.json')
    assert gate['harness_sha256'] == '10b101ab5722012af5e4bdac2ca428d3ad9fafd429ea6352e6f1761f0004e0cf'
    for command in gate['commands']:
        log = root / 'gate' / (command['world'] + '.log')
        assert sha(log) == command['log_sha256']
        text = log.read_text()
        if command['world'] == 'parent_force_exact':
            assert command['returncode'] != 0 and 'Invalid __global__ read' in text
            assert 'ERROR SUMMARY: 1842 errors' in text
        else:
            assert command['returncode'] == 0
            assert 'DENSE_GATE_RESULT passed=168 total=168 cpu_smoke=0' in text
            if command['world'] == 'fixed_force_exact':
                assert 'ERROR SUMMARY: 0 errors' in text
    build = json.loads((root / 'build.json').read_text())
    for world, record in build.items():
        for name, digest in record['sources'].items():
            assert sha(root / 'build' / world / name) == digest
    assert build['selected']['sources'] == build['null_rebuild']['sources']
    lock = json.loads((root / 'models.lock.json').read_text())
    model_records = json.loads((root / 'models.verified.json').read_text())
    assert model_records == {entry['file']: {'sha256': entry['sha256'], 'bytes': entry['bytes']}
                             for spec in lock.values() for entry in spec['quantized'].values()}
    rows = [json.loads(line) for line in (root / 'raw/bench.jsonl').read_text().splitlines()]
    cases = [('8b', 'Q4_0'), ('8b', 'Q8_0'), ('1b', 'Q8_0')]
    expected = {(c, m, q, r, w, n) for c, (m, q), r, w, n in itertools.product(
        ['810', 'default'], cases, range(6), ['selected', 'cutoff7', 'null_rebuild'], [1, 8, 16])}
    values = {}
    for row in rows:
        key = tuple(row[k] for k in ('clock', 'model', 'quant', 'round', 'world', 'n'))
        assert key in expected and key not in values
        assert len(row['samples_ts']) == 100
        assert all(math.isfinite(x) and x > 0 for x in row['samples_ts'])
        assert len(row['samples_ns']) == 100 and all(t > 0 for t in row['samples_ns'])
        rates = [1e9 * row['n_prompt'] / t for t in row['samples_ns']]
        # avg_ts uses six decimal places; sample rates use six significant digits.
        assert math.isclose(statistics.mean(rates), row['avg_ts'], abs_tol=1e-6)
        assert all(math.isclose(a, b, rel_tol=5e-6) for a, b in zip(rates, row['samples_ts']))
        assert row['n_gen'] == 0 and row['embeddings'] is True
        assert row['flash_attn'] == 1 and row['n_gpu_layers'] == 999
        assert row['n_threads'] == 4 and row['n_ubatch'] == 512 and row['n_batch'] == 2048
        assert Path(row['model_filename']).name == lock[row['model']]['quantized'][row['quant']]['file']
        assert 'fc9ce6b9d52a8504edcb262abc92737c2289f96c'.startswith(row['build_commit'])
        values[key] = row['avg_ts']
    assert set(values) == expected
    assert len(list((root / 'raw').glob('*.window.json'))) == 216
    assert len(list((root / 'raw').glob('*.telemetry.csv'))) == 216
    schedule = json.loads((root / 'schedule.json').read_text())
    assert len(schedule) == 216
    for item in schedule:
        tag = f"r{item['round']}-{item['clock']}-{item['model']}-{item['quant']}-{item['world']}-n{'_'.join(map(str, item['ns']))}"
        window = json.loads((root / 'raw' / (tag + '.window.json')).read_text())
        assert all(window[k] == v for k, v in item.items()) and window['start'] < window['end']
        source_rows = json.loads((root / 'raw' / (tag + '.json')).read_text())
        assert sorted(x['n_prompt'] for x in source_rows) == sorted(item['ns'])
        for source_row in source_rows:
            combined = next(x for x in rows if all(x[k] == item[k] for k in ('clock', 'model', 'quant', 'round', 'world'))
                            and x['n'] == source_row['n_prompt'])
            assert all(combined[k] == v for k, v in source_row.items())
    amendment = gate['amendment']
    assert sha(root / amendment['original_gate_folder'] / 'gate.json') == amendment['original_gate_json_sha256']
    assert sha(root / amendment['original_gate_folder'] / 'parent_force_exact.log') == sha(root / 'gate/parent_force_exact.log')
    assert sha(root / 'code/dense_gate.cpp') == gate['harness_sha256']
    starts = [json.loads(p.read_text())['start'] for p in (root / 'raw').glob('*.window.json')]
    assert amendment['declared_unix'] < min(starts)
    saved = json.loads((root / 'analysis.json').read_text())
    reductions = []
    for c, (m, q) in itertools.product(['810', 'default'], cases):
        def ratios(w, n):
            return [values[c, m, q, r, w, n] / values[c, m, q, r, 'selected', n] for r in range(6)]
        floor = max(abs(gm(ratios('null_rebuild', n)) - 1) for n in [1, 8, 16])
        control = max(abs(gm(ratios('cutoff7', n)) - 1) for n in [1, 16])
        effects = ratios('cutoff7', 8)
        rng = random.Random(0)
        boot = sorted(gm(rng.choices(effects, k=6)) for _ in range(10000))
        ratio, lo, hi, wins = gm(effects), boot[249], boot[9749], sum(x > 1 for x in effects)
        item = {'clock': c, 'model': m, 'quant': q, 'ratio': ratio, 'ci95_low': lo, 'ci95_high': hi,
                'wins': wins, 'paired_ratios': effects, 'pilot_null_floor_n1_n8_n16': floor,
                'route_control_deviation': control, 'control_pass': max(floor, control) <= .02,
                'material_win': ratio >= 1 + max(.03, floor, control) and lo > 1 + floor and wins >= 5,
                'material_loss': ratio <= 1 - max(.03, floor, control) and hi < 1 - floor and wins <= 1}
        original = next(x for x in saved['primary'] if (x['clock'], x['model'], x['quant']) == (c, m, q))
        for key, value in item.items():
            if isinstance(value, float):
                assert math.isclose(original[key], value, rel_tol=1e-12, abs_tol=1e-12), (c, m, q, key)
            else:
                assert original[key] == value, (c, m, q, key)
        reductions.append(item)
    summary = list(csv.DictReader((root / 'summary.csv').read_text().splitlines()))
    assert len(summary) == len(reductions) == 6
    for published, computed in zip(summary, reductions):
        for key, value in published.items():
            expected = computed[key]
            if isinstance(expected, bool):
                assert value == str(expected)
            elif isinstance(expected, (float, int)):
                assert math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12)
            else:
                assert value == expected
    assert len(saved['clocks']) == 108
    clock_issues = []
    for record in saved['clocks']:
        row = next(x for x in rows if x['n'] == 8 and all(x[k] == record[k] for k in ('clock', 'model', 'quant', 'round', 'world')))
        samples = [[float(x) for x in line[1:7]] for line in csv.reader((root / row['telemetry']).read_text().splitlines())]
        busy = [line for line in samples if line[5] > 0]
        assert len(busy) >= 3 and len(busy) == record['busy_samples']
        checks = {'sm_mean': statistics.mean(x[0] for x in busy), 'mem_mean': statistics.mean(x[1] for x in busy),
                  'temperature_max': max(x[4] for x in busy), 'pin_fraction': statistics.mean(abs(x[0] - 810) <= 15 for x in busy)}
        for key, value in checks.items():
            assert math.isclose(record[key], value, rel_tol=1e-12, abs_tol=1e-12)
        if record['clock'] == '810' and checks['pin_fraction'] < .9:
            clock_issues.append('low_clock_not_held')
        if checks['temperature_max'] > 80:
            clock_issues.append('thermal_review')
    for (m, q), w in itertools.product(cases, ['selected', 'cutoff7', 'null_rebuild']):
        groups = {c: [x for x in saved['clocks'] if (x['clock'], x['model'], x['quant'], x['world']) == (c, m, q, w)]
                  for c in ['810', 'default']}
        assert all(len(group) == 6 for group in groups.values())
        sm = {c: statistics.mean(x['sm_mean'] for x in group) for c, group in groups.items()}
        mem = {c: statistics.mean(x['mem_mean'] for x in group) for c, group in groups.items()}
        if sm['default'] < 1.5 * sm['810']:
            clock_issues.append('insufficient_frequency_separation')
        if abs(mem['default'] / mem['810'] - 1) > .01:
            clock_issues.append('memory_clock_difference')
    if clock_issues or not all(x['control_pass'] for x in reductions):
        decision = 'INCONCLUSIVE_FIX_MEASUREMENT_BEFORE_MORE_SPENDING'
    elif any(x['model'] == '8b' and x['material_win'] for x in reductions):
        decision = 'REVIEW_CANDIDATE_RULE_BEFORE_ANY_SECOND_GPU'
    else:
        decision = 'STOP_NO_MATERIAL_8B_WIN'
    assert decision == saved['decision']
    output = {'status': 'INDEPENDENT_HASH_AND_REDUCTION_REVIEW_PASSED', 'artifact_files': len(manifest),
              'rows': len(rows), 'invocations': 216, 'primary': reductions, 'analysis_decision': saved['decision'],
              'analysis_problems': saved['problems'], 'independent_clock_issues': clock_issues, 'scope': saved['scope']}
    if args.out:
        args.out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
