#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Recompute both GPUs with the collected six-round paired estimator (stdlib only)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[1]
WORLDS = ('selected', 'force_mmq', 'cutoff7', 'null_rebuild')

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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path,
                        help='Output directory; use a temporary directory to preserve committed results')
    args = parser.parse_args()
    if args.out.resolve() == (RUN / 'out').resolve():
        raise SystemExit('Use a separate output directory for verification')
    manifest = json.loads((RUN / 'SHA256SUMS.json').read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((RUN / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Collected artifact hash mismatch: ' + name)
    if clock_report(RUN) != json.loads((RUN / 'clocks.json').read_text()):
        raise RuntimeError('Raw telemetry does not reproduce the collected clock summary')
    l4 = REPO / 'runs/l4-2026-10-07-pinned-810MHz'
    lock = json.loads((RUN / 'models.lock.json').read_text())
    l4_models = json.loads((l4 / 'models/manifest.json').read_text())
    l20_models = json.loads((RUN / 'models.verified.json').read_text())
    for spec in lock.values():
        for entry in spec['quantized'].values():
            for field in ('sha256', 'bytes'):
                if not entry[field] == l4_models[entry['file']][field] == l20_models[entry['file']][field]:
                    raise RuntimeError('Model hash or size mismatch: ' + entry['file'])
    expected_inputs = json.loads((RUN / 'out/comparison.json').read_text())['inputs']
    inputs = {}
    all_rows = []
    for gpu, folder in (('L4', l4), ('L20', RUN)):
        path = folder / 'raw/bench.jsonl'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_inputs[gpu]['sha256']:
            raise RuntimeError('Raw input changed: ' + gpu)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if gpu == 'L4':
            mapping = {'llama3.1-8b': '8b', 'llama3.2-3b': '3b', 'llama3.2-1b': '1b'}
            for row in rows:
                matches = [value for key, value in mapping.items() if key in row['model']]
                if len(matches) != 1:
                    raise RuntimeError('Unknown L4 model: ' + row['model'])
                row['model'] = matches[0]
        all_rows.extend({'gpu': gpu, **row} for row in reduce_rows(rows))
        inputs[gpu] = {'raw_file': path.relative_to(REPO).as_posix(),
                       'sha256': digest, 'raw_rows': len(rows)}
    primary = [row for row in all_rows if row['n'] == 8]
    report = json.loads((RUN / 'out/comparison.json').read_text())
    report.update(inputs=inputs, primary_n8=primary, full_curves=all_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('paired-curves.csv', all_rows), ('paired-n8.csv', primary)):
        with (args.out / name).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'input_rows_per_gpu': {gpu: value['raw_rows'] for gpu, value in inputs.items()},
                      'primary_rows': len(primary), 'curve_rows': len(all_rows),
                      'original_artifact_hashes_verified': len(manifest)}, indent=2))

if __name__ == '__main__':
    main()
