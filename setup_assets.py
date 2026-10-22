"""Install separately shared weights; needs only the Python standard library."""
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def install(archive):
    manifest = json.loads((ROOT / 'bundle_manifest.json').read_text(encoding='utf-8'))
    expected = {k: v for k, v in manifest['files'].items() if k.startswith('weights/')}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('The archive must contain exactly the weights listed in bundle_manifest.json.')
        # Verify every member before changing any installed weight.
        for name, digest in expected.items():
            h = hashlib.sha256()
            with z.open(name) as src:
                for block in iter(lambda: src.read(4 * 1024**2), b''): h.update(block)
            if h.hexdigest() != digest: raise ValueError('Weight checksum mismatch: ' + name)
        for name in expected:
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.part')
            with z.open(name) as src, temporary.open('wb') as dest:
                for block in iter(lambda: src.read(4 * 1024**2), b''): dest.write(block)
            os.replace(temporary, target)
    print('WEIGHTS_OK:', len(expected), 'verified files installed. Do not install while a run is active.')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True, type=Path)
    install(p.parse_args().archive)
