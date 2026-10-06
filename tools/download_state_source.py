"""Acquire a pinned count source with capacity and publisher-checksum checks."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from opal.artifacts import read_json, write_json, sha256
from tools.download import download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/datasets/sciplex-state.json')
    p.add_argument('--raw', type=Path, default=ROOT / 'data/raw/state')
    args = p.parse_args()
    config = read_json(args.config)
    args.raw.mkdir(parents=True, exist_ok=True)
    destination = args.raw / config['filename']
    if not destination.exists():
        if shutil.disk_usage(args.raw).free < config['bytes'] + 1_000_000_000:
            raise OSError('Insufficient capacity for source and preparation reserve')
        result = download(config['url'], destination, config['publisher_md5'])
        if result['status'] != 'downloaded_not_accepted':
            raise RuntimeError(result.get('error'))
    with destination.open('rb') as f:
        md5 = hashlib.file_digest(f, 'md5').hexdigest()
    if md5 != config['publisher_md5'] or destination.stat().st_size != config['bytes']:
        raise ValueError('Source checksum or size mismatch')
    # Preserve the inspected processing script, whose hash identifies its contents.
    script = args.raw / 'publisher_processing.py'
    if not script.exists():
        with urllib.request.urlopen(config['processing_source'], timeout=60) as r:
            script.write_bytes(r.read())
    write_json(args.raw / 'verified.json', {'config_sha256': sha256(args.config),
        'source_sha256': sha256(destination), 'publisher_md5_verified': md5,
        'processing_sha256': sha256(script), 'accepted_for_modeling': False,
        'status': 'downloaded_pending_metadata_and_count_qualification'})


if __name__ == '__main__':
    main()
