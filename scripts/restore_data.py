"""Restore large research inputs from lossless, checksum-verified repository parts."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil
import tempfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def restore(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'data/packed/manifest.json').read_text())
    for entry in manifest:
        destination = root / entry['path']
        if destination.is_file():
            if destination.stat().st_size != entry['size'] or digest(destination) != entry['sha256']:
                raise ValueError(f'Existing data differ from the reference: {destination}')
            print(f"Verified {entry['path']}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile() as compressed:
            for part in entry['parts']:
                source = root / part['path']
                if digest(source) != part['sha256']:
                    raise ValueError(f'Packed-data checksum mismatch: {source}')
                with source.open('rb') as stream:
                    shutil.copyfileobj(stream, compressed)
            compressed.seek(0)
            temporary = destination.with_name(destination.name + '.restoring')
            try:
                with gzip.GzipFile(fileobj=compressed) as decoded, temporary.open('wb') as output:
                    shutil.copyfileobj(decoded, output)
                if temporary.stat().st_size != entry['size'] or digest(temporary) != entry['sha256']:
                    raise ValueError(f'Restored-data checksum mismatch: {destination}')
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Restored and verified {entry['path']}")


if __name__ == '__main__':
    restore(Path(__file__).resolve().parents[1])
