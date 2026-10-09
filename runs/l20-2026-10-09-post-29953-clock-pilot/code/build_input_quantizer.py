#!/usr/bin/env python3
"""Build the historical CPU quantizer only to restore byte-matched inputs."""
from pathlib import Path
import tarfile
from pilot import ROOT, sha, call

ARCHIVE_SHA = '265bbeada8caa673e2577bdf4fb2236b10f7fdcdacd5b686d5d47132d01b2a39'
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
call(['cmake', '-S', source, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF',
      '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', '-DLLAMA_OPENSSL=OFF', '-DLLAMA_BUILD_TESTS=OFF',
      '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_APP=OFF'], ROOT / 'input-tools/configure.log')
call(['cmake', '--build', build, '--target', 'llama-quantize', '-j', '4'], ROOT / 'input-tools/build.log')
print(build / 'bin/llama-quantize')
