"""Read-only custody and source-item inventory. No response parsing or network."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'results/frontier-execution-20261026'
MAX_BYTES = 1024**3


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def inventory(output):
    # Only previously qualified model custody roots. No reserved-source directory.
    scopes = [
        ('original_fitness', Path('C:/Users/USER/Desktop/opal-next-data-20261020/fitness-reference-model'), '*.npz'),
        ('block', Path('C:/Users/USER/Desktop/opal-research-next-20261020/block-v2'), '*/fold-*/*.npz'),
        ('budget', Path('C:/Users/USER/Desktop/opal-wave2-20261021/q3b-v3/checkpoints'), '**/*.npz'),
        ('reset_development', ROOT / 'results/research-reset-development', 'rep*-*.npz'),
        ('native_missing', Path('E:/opal-research-next-20261021/native-block-v6'), '*/fold-*/*.npz'),
        ('reagent_missing', Path('E:/opal-wave2-20261021/q1b-v2'), '**/checkpoints/*.npz'),
        ('flexible_missing', Path('E:/opal-wave2-20261021/q2b-v1'), '**/checkpoints/*.npz'),
        ('wave3_missing', Path('E:/opal-wave3-20261021/nuisance-v1'), '**/checkpoints/*.npz'),
    ]
    started = time.perf_counter()
    records, total = [], 0
    for name, base, pattern in scopes:
        row = dict(scope=name, root=str(base), present=base.is_dir(), pattern=pattern, files=[])
        if base.is_dir():
            for path in sorted(base.glob(pattern)):
                if not path.is_file() or path.is_symlink():
                    continue
                if not path.resolve().is_relative_to(base.resolve()):
                    raise ValueError('Custody path escaped its declared root')
                size = path.stat().st_size
                if total + size > MAX_BYTES or time.perf_counter() - started > 120:
                    raise RuntimeError('Custody hash byte/time cap; no partial success')
                row['files'].append(dict(path=str(path.relative_to(base)), bytes=size, sha256=sha(path)))
                total += size
        records.append(row)
    items = []
    for name in ('A_source_requests_v2.json', 'A_source_requests_v3.json',
                 'A_source_requests_v6.json', 'A_source_requests_v8.json'):
        receipt = read(EVIDENCE / name)
        requests = receipt.get('requests', [receipt])
        for request in requests:
            items.append(dict(url=request['url'], evidence=name, receipt_sha256=sha(EVIDENCE / name),
                              source_sha256=request.get('sha256'), status=request.get('status'),
                              role=request.get('outcome'), new_access=False,
                              rights='See item-specific rights ledger; no blanket data grant'))
    annotation = read(ROOT / 'results/research-continuation/colo1-annotation-acquisition-receipt-v2.json')
    items.append(dict(url=annotation['member']['content_url'], role='isolated annotation member',
                      source_sha256=annotation['sha256'], git_blob_sha1=annotation['git_blob_sha1'],
                      bytes=annotation['bytes'], rights='CC BY4.0 exact-member association in annotation results v1',
                      permitted='Local metadata inspection; raw member not redistributed', new_access=False))
    catalog = read(EVIDENCE / 'A_source_requests_v6.json')['requests'][0]['catalog']
    for item in catalog['files']:
        items.append(dict(url=item['download_url'], role='unretrieved mixed mapping archive',
                          bytes=item['size'], catalog_md5_unverified=item['computed_md5'],
                          rights='Version container declares CC BY4.0; unidentified member rights unknown',
                          permitted='Only prior registered range request; closed403, no body read', new_access=False))
    output.mkdir(parents=True, exist_ok=False)
    result = dict(utc=datetime.now(timezone.utc).isoformat(), tool_sha256=sha(Path(__file__)),
                  wall_seconds=time.perf_counter()-started, checkpoint_bytes_hashed=total,
                  checkpoint_deserialization=False, response_reads=0, network_requests=0,
                  historical_files_modified=False, custody=records, source_items=items,
                  scope='Declared qualified model roots only; missing roots remain blockers, not verified custody')
    (output / 'inventory.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(checkpoints=sum(len(r['files']) for r in records), bytes_hashed=total,
                          missing=[r['scope'] for r in records if not r['present']], source_items=len(items))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    inventory(parser.parse_args().output.resolve())
