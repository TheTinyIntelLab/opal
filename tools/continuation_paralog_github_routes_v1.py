"""Finite owner-catalog inspection; never request repository file contents."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / 'results/research-continuation'
BASE = 'https://api.github.com/repos/team113sanger/harle_dgCRISPR_paralogs/'
ROUTES = [('pinned-tree', 'git/trees/fe14bb800f2766235096a655c2fb5da572808a8a?recursive=1'),
          ('current-tree', 'git/trees/HEAD?recursive=1'), ('releases', 'releases?per_page=100')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    freeze_path = RECORDS / 'paralog-github-routes-freeze-v1.json'
    binding = json.loads((RECORDS / 'paralog-github-routes-code-binding-v1.json').read_text())
    freeze = json.loads(freeze_path.read_text())
    checks = {**freeze['files_sha256'], **binding['code_sha256']}
    if sha(freeze_path) != binding['freeze_sha256']:
        raise ValueError('Freeze drift')
    for path, expected in checks.items():
        if sha(ROOT / path) != expected:
            raise ValueError('Input drift: ' + path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Disk reserve')
    opener = urllib.request.build_opener(NoRedirect())
    rows = []
    for name, route in ROUTES:
        started = time.perf_counter()
        row = dict(name=name, url=BASE + route, observed_utc=datetime.now(timezone.utc).isoformat())
        try:
            request = urllib.request.Request(row['url'], headers={'User-Agent': 'Opal-metadata-catalog/1.0', 'Accept': 'application/vnd.github+json'})
            with opener.open(request, timeout=30) as stream:
                row['status'] = stream.status
                if stream.status != 200 or stream.headers.get_content_type() != 'application/json':
                    raise ValueError('Catalog status/MIME refusal')
                raw = stream.read(2 * 1024**2 + 1)
                if len(raw) > 2 * 1024**2:
                    raise ValueError('Catalog byte cap')
            row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            value = json.loads(raw)
            if name.endswith('tree'):
                row.update(tree_sha=value['sha'], truncated=value['truncated'])
                if value['truncated']:
                    raise ValueError('Truncated catalog')
                row['entries'] = [{k: entry[k] for k in ('path', 'type', 'sha', 'size') if k in entry} for entry in value['tree']]
            else:
                row['releases'] = [dict(tag=v['tag_name'], assets=[dict(name=a['name'], size=a['size'], url=a['browser_download_url']) for a in v['assets']]) for v in value]
                row['pagination_not_requested'] = True
        except (urllib.error.URLError, ValueError, TimeoutError) as error:
            row['error'] = type(error).__name__ + ': ' + str(error)
        row['elapsed_seconds'] = time.perf_counter() - started
        rows.append(row)
        if 'error' in row:
            break
    for path, expected in checks.items():
        if sha(ROOT / path) != expected:
            raise ValueError('Post-execution drift: ' + path)
    result = dict(schema='opal-paralog-github-routes-v1', freeze_sha256=sha(freeze_path), requests=rows,
                  source_asset_requests=0, repository_blobs_opened=0, response_tables_or_headers_opened=False,
                  source_qualified=False, fits=0, scores=0)
    with (RECORDS / 'paralog-github-routes-receipt-v1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps([dict(name=r['name'], status=r.get('status'), entries=len(r.get('entries', [])), releases=len(r.get('releases', [])), error=r.get('error')) for r in rows]))


if __name__ == '__main__':
    main()
