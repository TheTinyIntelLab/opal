"""Retrieve the pinned independent source, accepting only registered checksums."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, sha256
from download import download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/datasets/dspin-confirmation-v1.json')
    args = p.parse_args()
    config = read_json(args.config)
    path = Path(config['raw_path'])
    if path.exists():
        if sha256(path) != config['sha256']:
            raise ValueError('Existing independent-source file does not match the registered SHA256')
        print('Verified cached independent source:', path)
        return
    # Avoid a cached, expired signed object-store redirect from the publisher.
    url = config['source'] + '/files/dosage_combination_raw_counts.h5ad?download=1&opal=' + str(int(time.time()))
    result = download(url, path, config['publisher_md5'])
    if result['status'] != 'downloaded_not_accepted' or result.get('sha256') != config['sha256']:
        raise RuntimeError('Independent source download or registered SHA256 verification failed')


if __name__ == '__main__':
    main()
