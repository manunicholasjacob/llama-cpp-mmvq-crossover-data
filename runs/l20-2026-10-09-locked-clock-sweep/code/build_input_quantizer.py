#!/usr/bin/env python3
"""Build the historical CPU quantizer only to restore byte-matched inputs."""
import hashlib
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
ARCHIVE_SHA = '265bbeada8caa673e2577bdf4fb2236b10f7fdcdacd5b686d5d47132d01b2a39'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    archive = ROOT / 'assets/quantizer-b96806d9.tar.gz'
    if sha(archive) != ARCHIVE_SHA:
        raise RuntimeError('Historical input quantizer source mismatch')
    source = ROOT / 'input-tools/source'
    if source.exists():
        raise RuntimeError('Use a fresh input-tools directory; do not overwrite provenance')
    source.mkdir(parents=True)
    with tarfile.open(archive) as t:
        for item in t:
            parts = Path(item.name).parts[1:]
            if not parts:
                continue
            if Path(item.name).is_absolute() or '..' in parts or not (item.isfile() or item.isdir()):
                raise RuntimeError('Unsafe source member')
            item.name = str(Path(*parts))
            t.extract(item, source)
    build = ROOT / 'input-tools/build'
    flags = ['-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF', '-DGGML_CUDA=OFF',
             '-DGGML_METAL=OFF', '-DLLAMA_OPENSSL=OFF', '-DLLAMA_BUILD_TESTS=OFF',
             '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_APP=OFF']
    subprocess.run(['cmake', '-S', str(source), '-B', str(build), *flags], check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'llama-quantize', '-j', '4'], check=True)
    print(build / 'bin/llama-quantize')


if __name__ == '__main__':
    main()
