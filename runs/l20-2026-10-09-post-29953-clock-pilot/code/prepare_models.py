#!/usr/bin/env python3
"""Restore immutable inputs; fail closed if regenerated pure files differ."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--endpoint', default='https://huggingface.co')
    parser.add_argument('--quantizer', type=Path)
    parser.add_argument('--only', choices=['1b', '8b'])
    args = parser.parse_args()
    lock = json.loads((ROOT / 'models.lock.json').read_text())
    folder = ROOT / 'models'
    folder.mkdir(exist_ok=True)
    report = {}
    for key in ('1b', '8b'):
        if args.only and key != args.only:
            continue
        spec = lock[key]
        base = folder / spec['file']
        if args.download and not base.exists():
            part = base.with_suffix('.gguf.part')
            url = f"{args.endpoint.rstrip('/')}/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
            print(f'Downloading fixed revision: {key}', flush=True)
            subprocess.run(['curl', '--fail', '--location', '--retry', '5', '--retry-delay', '3',
                            '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '120',
                            '--continue-at', '-', '--output', str(part), url], check=True)
            if sha(part) != spec['source_sha256']:
                raise RuntimeError(f'Source SHA256 mismatch: {part}')
            part.rename(base)
        if base.exists() and sha(base) != spec['source_sha256']:
            raise RuntimeError(f'Source SHA256 mismatch: {base}')
        if args.download_only:
            if not base.exists():
                raise RuntimeError(f'Missing source: {base}')
            print(f'Verified source: {key}', flush=True)
            continue
        for quant, entry in spec['quantized'].items():
            target = folder / entry['file']
            if not target.exists() and args.quantizer:
                if not base.exists():
                    raise RuntimeError(f'Missing source: {base}')
                part = target.with_suffix('.gguf.part')
                cmd = [str(args.quantizer.resolve()), '--pure', '--allow-requantize',
                       str(base), str(part), quant, '4']
                with open(folder / f'{key}-{quant}.quantize.log', 'w') as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
                actual = sha(part)
                if actual != entry['sha256'] or part.stat().st_size != entry['bytes']:
                    (folder / f'{key}-{quant}.mismatch.json').write_text(json.dumps({
                        'expected': entry, 'actual_sha256': actual,
                        'actual_bytes': part.stat().st_size, 'command': cmd}, indent=2) + '\n')
                    raise RuntimeError(f'Quantized SHA256 mismatch: {part}; preserved for diagnosis')
                part.rename(target)
            if not target.exists():
                raise RuntimeError(f'Missing pure model: {target}; download and quantize first')
            actual = sha(target)
            if actual != entry['sha256'] or target.stat().st_size != entry['bytes']:
                raise RuntimeError(f'Quantized input mismatch: {target}')
            report[target.name] = {'sha256': actual, 'bytes': target.stat().st_size}
            print(f'Verified: {target.name}', flush=True)
    (folder / ('verified.json' if not args.only else f'verified-{args.only}.json')).write_text(
        json.dumps(report, indent=2) + '\n')

if __name__ == '__main__':
    main()
