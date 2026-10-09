#!/usr/bin/env python3
"""Read-only independent review of the sealed L20 sweep; writes a sibling report."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import statistics
import tarfile


def review(archive):
    payload = {}
    public = archive.is_dir()
    if public:
        payload = {str(p.relative_to(archive)): p.read_bytes()
                   for p in archive.rglob('*') if p.is_file()}
    else:
        with tarfile.open(archive) as bundle:
            for member in bundle:
                if member.isdir():
                    continue
                path = PurePosixPath(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts
                name = str(PurePosixPath(*path.parts[1:]))
                assert name not in payload
                payload[name] = bundle.extractfile(member).read()

    def read(name):
        return json.loads(payload[name])

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    inventory = read('SHA256SUMS.json')
    if public:
        provenance = read('PUBLIC-PROVENANCE.json')
        redacted = provenance['redactions']
        assert digest(payload['SHA256SUMS.json']) == provenance['original_manifest_sha256']
        for name, value in inventory.items():
            expected = value
            if name in redacted:
                assert redacted[name]['original_sha256'] == value
                expected = redacted[name]['public_sha256']
            assert digest(payload[name]) == expected
        shares = dict(line.split('  ', 1)[::-1]
                      for line in payload['SHARE_SHA256SUMS'].decode().splitlines())
        assert set(shares) == set(payload) - {'SHARE_SHA256SUMS'}
        assert all(digest(payload[name]) == value for name, value in shares.items())
        archive_digest = provenance['private_transfer_archive_sha256']
    else:
        assert set(inventory) == set(payload) - {'SHA256SUMS.json'}
        assert all(digest(payload[name]) == value for name, value in inventory.items())
        archive_digest = digest(archive.read_bytes())
        assert archive.with_suffix(archive.suffix + '.sha256').read_text().split()[0] == archive_digest
    protocol = read('protocol.json')
    builds = read('build.json')
    source_lock = read('source.lock.json')
    for world, record in builds.items():
        assert record['commit'] == protocol['fixed_commit']
        assert record['flags'] == builds['selected']['flags']
        for name, value in record['sources'].items():
            data = payload[f'build/{world}/{name}']
            assert digest(data) == value
            if world != 'cutoff7' or name != 'mmvq.cu':
                assert value == source_lock[f'ggml/src/ggml-cuda/{name}']
    anchor = b'bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {'
    insertion = b'\n    if (cc == GGML_CUDA_CC_ADA_LOVELACE &&\n        (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0)) {\n        return ne11 <= 7;\n    }'
    assert payload['build/cutoff7/mmvq.cu'] == payload['build/selected/mmvq.cu'].replace(anchor, anchor + insertion)
    models = read('models.lock.json')
    verified = read('models.verified.json')
    for model in models.values():
        for spec in model['quantized'].values():
            assert verified[spec['file']] == {k: spec[k] for k in ('sha256', 'bytes')}

    rows = [json.loads(line) for line in payload['raw/bench.jsonl'].splitlines()]
    schedule = read('schedule.json') + read('schedule-1b.json')
    identity = lambda r: tuple(r[k] for k in ('phase', 'round', 'clock', 'model', 'quant', 'world', 'n'))
    expected = [identity({**item, 'n': n}) for item in schedule for n in item['ns']]
    assert [identity(r) for r in rows] == expected and len(set(expected)) == len(expected)
    assert read('execution.json')['invocations'] == len(schedule) == read('execution.json')['expected_invocations']
    windows = []
    sample_count = 0
    for item in schedule:
        matching = [r for r in rows if identity({**item, 'n': r['n']}) == identity(r)]
        telemetry = matching[0]['telemetry']
        prefix = telemetry.removesuffix('.telemetry.csv')
        raw = read(prefix + '.json')
        window = read(prefix + '.window.json')
        assert all(window[k] == v for k, v in item.items())
        assert window['end'] > window['start']
        assert len(raw) == len(matching)
        for original, combined in zip(raw, matching):
            assert all(combined[k] == v for k, v in original.items())
            assert all(combined[k] == v for k, v in item.items() if k != 'ns')
            assert original['n_prompt'] == combined['n'] and original['n_gen'] == 0
            settings = {'embeddings': True, 'flash_attn': 1, 'n_gpu_layers': 999,
                        'n_ubatch': 512, 'n_batch': 2048, 'n_threads': 4}
            assert all(original[k] == v for k, v in settings.items())
            assert original['build_commit'] == protocol['fixed_commit']
            assert Path(original['model_filename']).name == models[item['model']]['quantized'][item['quant']]['file']
            ns, ts = original['samples_ns'], original['samples_ts']
            assert len(ns) == len(ts) == 100 and all(x > 0 for x in ns)
            reconstructed = [combined['n'] * 1e9 / value for value in ns]
            # llama-bench's sample array uses ostream's six significant digits;
            # avg_ts uses std::to_string's six decimal places.
            assert all(abs(a - b) <= .500001 * 10 ** (math.floor(math.log10(b)) - 5)
                       for a, b in zip(ts, reconstructed))
            assert abs(statistics.fmean(reconstructed) - original['avg_ts']) < .000001
            sample_count += len(ns)
        busy = []
        for record in csv.reader(io.StringIO(payload[telemetry].decode())):
            assert len(record) == 8
            sm, mem, power, limit, temp, util = map(float, record[1:7])
            assert all(math.isfinite(x) for x in (sm, mem, power, limit, temp, util))
            if util > 0:
                busy.append((sm, mem, temp))
        assert len(busy) >= 3
        windows.append({**{k: item[k] for k in ('phase', 'clock', 'quant', 'round', 'world')},
                        'held': all(abs(x[0] - item['clock']) <= protocol['pin_tolerance_mhz'] for x in busy),
                        'sm_min': min(x[0] for x in busy), 'sm_max': max(x[0] for x in busy),
                        'memory_clocks': sorted({x[1] for x in busy}),
                        'busy_samples': len(busy), 'temperature_max': max(x[2] for x in busy),
                        'start': window['start'], 'end': window['end']})
    ordered = sorted(windows, key=lambda x: x['start'])
    assert all(a['end'] <= b['start'] for a, b in zip(ordered, ordered[1:]))
    gm = lambda xs: math.exp(statistics.fmean(math.log(x) for x in xs))
    cells = []
    grouped = collections.defaultdict(dict)
    for row in rows:
        if row['n'] == 8:
            grouped[row['phase'], row['clock'], row['model'], row['quant']][row['round'], row['world']] = row['avg_ts']
    recorded = read('analysis.json')
    for (phase, clock, model, quant), values in sorted(grouped.items()):
        ratios = [values[r, 'cutoff7'] / values[r, 'selected'] for r in range(6)]
        null = abs(gm([values[r, 'null_rebuild'] / values[r, 'selected'] for r in range(6)]) - 1)
        rng = random.Random(0)
        boot = sorted(gm(rng.choices(ratios, k=6)) for _ in range(10000))
        effect = {'ratio': gm(ratios), 'ci95_low': boot[249], 'ci95_high': boot[9749],
                  'wins': sum(x > 1 for x in ratios), 'losses': sum(x < 1 for x in ratios),
                  'pilot_null_floor': null}
        saved = next(x for x in recorded['primary' if phase == 'primary' else 'small_model']
                     if (x['clock'], x['quant']) == (clock, quant))
        assert all(math.isclose(effect[k], saved[k], rel_tol=1e-12, abs_tol=1e-12) for k in effect)
        cells.append({'phase': phase, 'clock': clock, 'model': model, 'quant': quant, **effect})
    valid = [clock for clock in read('probe.json')['accepted_mhz']
             if all(x['held'] for x in windows if x['phase'] == 'primary' and x['clock'] == clock)
             and all(x['pilot_null_floor'] <= .02 for x in cells if x['phase'] == 'primary' and x['clock'] == clock)]
    assert valid == recorded['valid_mhz']
    control = {(r['clock'], r['quant'], r['n'], r['world']): r['avg_ts']
               for r in rows if r['phase'] == 'route_control'}
    worst = max(abs(v / control[c, q, n, 'selected'] - 1)
                for (c, q, n, w), v in control.items() if w != 'selected')
    assert math.isclose(worst, recorded['route_controls']['worst'], abs_tol=1e-12)
    return {'archive_sha256': archive_digest, 'sealed_hashes_verified': len(inventory),
            'source_patch_and_recorded_model_hashes_verified': True,
            'invocations_verified': len(schedule), 'rows_verified': len(rows),
            'inner_samples_verified': sample_count, 'valid_mhz': valid,
            'recomputed_cells': cells, 'route_control_worst': worst,
            'failed_lock_windows': [x for x in windows if not x['held']],
            'max_temperature_c': max(x['temperature_max'] for x in windows),
            'memory_clocks': sorted({v for x in windows for v in x['memory_clocks']}),
            'minimum_busy_samples': min(x['busy_samples'] for x in windows),
            'timing_span_minutes': (ordered[-1]['end'] - ordered[0]['start']) / 60,
            'notes': ['810 MHz and sanitizer were not repeated in this archive.',
                      'Model payloads and executable binaries are not included; their recorded hashes were checked.',
                      'No per-operator timing, intermediate boundary frequencies, or held-out routing validation.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    result = review(args.archive)
    dest = args.archive.with_name(args.archive.name.removesuffix('.tar.gz') + '-independent-review.json')
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('recomputed_cells', 'failed_lock_windows')}, indent=2))
    print('failed_lock_windows:', len(result['failed_lock_windows']))
    print('report:', dest)
