#!/usr/bin/env python3
"""Verify and reconstruct the original model checkpoints (no external downloads)."""
import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
manifest = json.loads((BASE / 'manifest.json').read_text())

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify_parts():
    for p in manifest['parts']:
        path=BASE / Path(p['path']).relative_to('checkpoints')
        if not path.is_file() or path.stat().st_size!=p['bytes'] or digest(path)!=p['sha256']:
            raise RuntimeError(f'Missing or corrupt archived checkpoint part: {path}')
    print(f"Verified {len(manifest['parts'])} packed parts.")

def restore():
    verify_parts()
    with tempfile.TemporaryDirectory(prefix='checkpoint-restore-') as temp:
        archive=Path(temp)/'checkpoints.tar.gz'
        with archive.open('wb') as output:
            for p in manifest['parts']:
                with (BASE / Path(p['path']).relative_to('checkpoints')).open('rb') as input_:
                    for block in iter(lambda: input_.read(2 * 1024 * 1024), b''):
                        output.write(block)
        if archive.stat().st_size!=manifest['archive_bytes'] or digest(archive)!=manifest['archive_sha256']:
            raise RuntimeError('Archive size or SHA-256 mismatch')
        records={f['path']: f for f in manifest['files']}
        encountered=set()
        with tarfile.open(archive,'r:gz') as tar:
            for item in tar:
                if item.name not in records or not item.isfile() or item.name in encountered:
                    raise RuntimeError(f'Unexpected tar archive item: {item.name}')
                encountered.add(item.name)
                rec=records[item.name]
                path=BASE / Path(rec['path']).relative_to('checkpoints')
                path.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(item) as input_, path.open('wb') as output:
                    for block in iter(lambda: input_.read(2 * 1024 * 1024), b''):
                        output.write(block)
                if path.stat().st_size!=rec['bytes'] or digest(path)!=rec['sha256']:
                    raise RuntimeError(f'Checkpoint checksum mismatch: {path}')
        if encountered!=set(records): raise RuntimeError('Missing files in checkpoint archive')
        for alias in manifest['aliases']:
            source=BASE / Path(alias['target']).relative_to('checkpoints')
            target=BASE / Path(alias['path']).relative_to('checkpoints')
            target.parent.mkdir(parents=True,exist_ok=True)
            # Alias is duplicated locally only, not stored twice in the Git repository.
            with source.open('rb') as input_,target.open('wb') as output:
                for block in iter(lambda: input_.read(2 * 1024 * 1024), b''):
                    output.write(block)
            if target.stat().st_size!=alias['bytes'] or digest(target)!=alias['sha256']:
                raise RuntimeError(f'Checkpoint alias checksum mismatch: {target}')
    print(f"Restored {len(records)} distinct artifacts and {len(manifest['aliases'])} local path aliases.")
    verify_restored()

def verify_restored():
    missing=[]
    for rec in manifest['files']+manifest['aliases']:
        path=BASE / Path(rec['path']).relative_to('checkpoints')
        if not path.is_file() or path.stat().st_size!=rec['bytes'] or digest(path)!=rec['sha256']:
            missing.append(rec['path'])
    if missing: raise RuntimeError(f'Missing/corrupt unpacked checkpoint artifacts: {missing}')
    print(f"Verified {len(manifest['files'])} distinct restored weights and {len(manifest['aliases'])} compatibility aliases.")

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-packed',action='store_true',help='Check packed parts without expanding')
    parser.add_argument('--verify-restored',action='store_true',help='Check unpacked checkpoints only')
    args=parser.parse_args()
    if args.verify_packed: verify_parts()
    elif args.verify_restored: verify_restored()
    else: restore()
