"""Restore byte-preserved historical documentation into a separate directory."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def restore(output, archive_root=ROOT / 'docs/archive'):
    output, archive_root = Path(output).resolve(), Path(archive_root)
    manifest = json.loads((archive_root / 'retired-documents.json').read_text())
    archive = archive_root / manifest['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise ValueError('Historical archive checksum mismatch')
    records = [r for r in manifest['files'] if r['disposition'] == 'public_history_archive']
    prepared = []
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != {r['path'] for r in records}:
            raise ValueError('Historical archive membership mismatch')
        for record in records:
            destination = (output / record['path']).resolve()
            if not destination.is_relative_to(output) or not record['path'].startswith('docs/'):
                raise ValueError('Historical path escapes output directory')
            if destination.exists() or destination.is_symlink():
                raise FileExistsError('Refusing to overwrite: ' + str(destination))
            payload = z.read(record['path'])
            if hashlib.sha256(payload).hexdigest() != record['sha256']:
                raise ValueError('Historical document checksum mismatch')
            prepared.append((destination, payload))
    # Validate the entire set before creating any restored file.
    for destination, payload in prepared:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return dict(restored=len(prepared), original_bytes_verified=True,
                private_documents_omitted=len(manifest['files'])-len(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.output)))


if __name__ == '__main__':
    main()
