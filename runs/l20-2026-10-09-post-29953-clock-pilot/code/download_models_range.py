#!/usr/bin/env python3
"""Bounded resumable range download; validate ranges, sizes and source SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from prepare_models import ROOT, sha

def curl(cmd):
    return subprocess.run(['curl', '--http1.1', '--fail', '--location', '--silent', '--show-error',
        '--retry', '5', '--retry-all-errors', '--connect-timeout', '30', *cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

def download(key, spec, endpoint, workers):
    folder = ROOT / 'models'
    folder.mkdir(exist_ok=True)
    target = folder / spec['file']
    if target.exists():
        if sha(target) != spec['source_sha256']:
            raise RuntimeError('Existing source mismatch')
        return
    pieces = folder / (spec['file'] + '.chunks')
    pieces.mkdir(exist_ok=True)
    url = f"{endpoint.rstrip('/')}/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
    probe = curl(['--range', '0-0', '--max-time', '120', '--output', str(pieces / 'probe'),
                  '--dump-header', str(pieces / 'probe.headers'), '--write-out', '%{json}', url])
    headers = (pieces / 'probe.headers').read_text()
    ranges = re.findall(r'(?im)^content-range:\s*bytes 0-0/(\d+)', headers)
    if not ranges or (pieces / 'probe').stat().st_size != 1:
        raise RuntimeError('Server did not honor the probe range')
    total = int(ranges[-1])
    direct_url = json.loads(probe.stdout)['url_effective']
    chunk_size = 64 * 1024 * 1024
    count = (total + chunk_size - 1) // chunk_size
    start_time = time.time()
    def part(index):
        start = index * chunk_size
        end = min(total - 1, start + chunk_size - 1)
        file = pieces / f'{index:05d}.bin'
        if file.exists() and file.stat().st_size == end - start + 1:
            return file
        temp = file.with_suffix(f'.part.{os.getpid()}')
        header = file.with_suffix(f'.headers.{os.getpid()}')
        curl(['--range', f'{start}-{end}', '--max-time', '1800', '--output', str(temp),
              '--dump-header', str(header), direct_url])
        expected = f'bytes {start}-{end}/{total}'
        actual = re.findall(r'(?im)^content-range:\s*(.*?)\s*$', header.read_text())
        if not actual or actual[-1] != expected or temp.stat().st_size != end - start + 1:
            raise RuntimeError(f'Invalid download range: {index}')
        temp.rename(file)
        return file
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(part, index): index for index in range(count)}
        done = 0
        for future in as_completed(pending):
            future.result()
            done += 1
            print(f'{key}: {done}/{count} chunks, elapsed {time.time()-start_time:.0f}s', flush=True)
    temp = folder / (spec['file'] + f'.assembled.{os.getpid()}')
    with open(temp, 'wb') as out:
        for index in range(count):
            with open(pieces / f'{index:05d}.bin', 'rb') as source:
                shutil.copyfileobj(source, out, 8 * 1024 * 1024)
    if temp.stat().st_size != total or sha(temp) != spec['source_sha256']:
        raise RuntimeError('Assembled source SHA256 mismatch; preserve pieces for diagnosis')
    temp.rename(target)
    print(f'{key}: source SHA256 verified', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default='https://huggingface.co')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--only', choices=['1b', '8b'])
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise RuntimeError('Use 1 to 16 connections')
    lock = json.loads((ROOT / 'models.lock.json').read_text())
    for key in ('1b', '8b'):
        if not args.only or key == args.only:
            download(key, lock[key], args.endpoint, args.workers)

if __name__ == '__main__':
    main()
