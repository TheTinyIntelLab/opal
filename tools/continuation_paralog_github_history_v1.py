"""Inspect a finite set of public owner Git catalogs without opening blobs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research-continuation'
BASE = 'https://api.github.com/repos/team113sanger/harle_dgCRISPR_paralogs/'
KNOWN = 'fe14bb800f2766235096a655c2fb5da572808a8a'
CATALOGS = [('branches', 'branches?per_page=100'),
            ('tags', 'tags?per_page=100'), ('commits', 'commits?per_page=100')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate catalog key')
        value[key] = item
    return value


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def identity(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{40}', value) is None:
        raise ValueError('Invalid Git identity')
    return value


def validate_refs(value, kind):
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError('Invalid ref catalog')
    rows = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError('Invalid ref entry')
        if kind == 'commits':
            row = dict(sha=identity(item.get('sha')))
            key = row['sha']
        else:
            name = item.get('name')
            commit = item.get('commit')
            if not isinstance(name, str) or not name or not isinstance(commit, dict):
                raise ValueError('Invalid named ref')
            row = dict(name=name, sha=identity(commit.get('sha')))
            key = name
        if key in seen:
            raise ValueError('Duplicate ref identity')
        seen.add(key)
        rows.append(row)
    return rows


def validate_tree(value):
    if not isinstance(value, dict) or value.get('truncated') is not False:
        raise ValueError('Truncated or malformed tree')
    tree_sha = identity(value.get('sha'))
    entries = value.get('tree')
    if not isinstance(entries, list):
        raise ValueError('Missing tree entries')
    seen = set()
    rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Malformed tree entry')
        path, kind = entry.get('path'), entry.get('type')
        if not isinstance(path, str) or not path or kind not in {'blob', 'tree', 'commit'}:
            raise ValueError('Invalid tree path/type')
        parsed = PurePosixPath(path)
        if path in seen or parsed.is_absolute() or '..' in parsed.parts or '\\' in path:
            raise ValueError('Unsafe or duplicate tree path')
        seen.add(path)
        row = dict(path=path, type=kind, sha=identity(entry.get('sha')))
        if kind == 'blob':
            size = entry.get('size')
            if type(size) is not int or size < 0:
                raise ValueError('Invalid blob size')
            row['size'] = size
        rows.append(row)
    return dict(tree_sha=tree_sha, entries=rows, truncated=False, entry_count=len(rows))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch(task):
    kind, route = task
    row = dict(kind=kind, url=BASE + route, observed_utc=datetime.now(timezone.utc).isoformat())
    started = time.perf_counter()
    try:
        request = urllib.request.Request(row['url'], headers={
            'User-Agent': 'Opal-response-free-history-catalog/1.0',
            'Accept': 'application/vnd.github+json'})
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as stream:
            row['status'] = stream.status
            if stream.status != 200 or stream.headers.get_content_type() != 'application/json':
                raise ValueError('Unexpected catalog status/MIME')
            row['pagination_advertised'] = bool(stream.headers.get('Link'))
            raw = stream.read(2 * 1024**2 + 1)
        if len(raw) > 2 * 1024**2:
            raise ValueError('Catalog byte limit')
        row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        value = decode(raw)
        if kind == 'tree':
            row.update(validate_tree(value))
        else:
            row['refs'] = validate_refs(value, kind)
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        if isinstance(error, urllib.error.HTTPError):
            row['status'] = error.code
            error.close()
        row['error'] = type(error).__name__ + ': ' + str(error)
    row['elapsed_seconds'] = time.perf_counter() - started
    return row


def main():
    output = OUT / 'paralog-github-history-receipt-v1.json'
    if output.exists():
        raise FileExistsError(output)
    freeze_path = OUT / 'paralog-github-history-freeze-v1.json'
    binding_path = OUT / 'paralog-github-history-code-binding-v1.json'
    freeze, binding = decode(freeze_path.read_bytes()), decode(binding_path.read_bytes())
    if sha(freeze_path) != binding['freeze_sha256']:
        raise ValueError('History freeze drift')
    checks = {**freeze['sha256'], **binding['code_sha256']}
    for path, expected in checks.items():
        if sha(ROOT / path) != expected:
            raise ValueError('History input drift: ' + path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')
    baseline = decode((OUT / 'paralog-github-routes-receipt-v1.json').read_bytes())
    old_paths = {e['path'] for e in baseline['requests'][0]['entries']}
    started = time.perf_counter()
    discovery, trees = [], []
    stopped = None
    for task in CATALOGS:
        row = fetch(task)
        discovery.append(row)
        if 'error' in row:
            stopped = 'Discovery refusal'
            break
    commits = sorted({v['sha'] for row in discovery for v in row.get('refs', [])} - {KNOWN})
    if len(commits) > 40:
        stopped = 'Unique commit ceiling exceeded'
    if stopped is None:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0, len(commits), 4):
                tasks = [('tree', 'git/trees/' + commit + '?recursive=1') for commit in commits[start:start + 4]]
                wave = list(pool.map(fetch, tasks))
                trees.extend(wave)
                if any('error' in row for row in wave):
                    stopped = 'Tree-wave refusal'
                    break
    for row in trees:
        if 'error' not in row:
            paths = {e['path'] for e in row['entries']}
            row['paths_unseen_at_baseline'] = sorted(paths - old_paths)
            row['metadata_named_paths'] = sorted(p for p in paths if 'metadata' in p.lower())
    for path, expected in checks.items():
        if sha(ROOT / path) != expected:
            raise ValueError('Post-history drift: ' + path)
    result = dict(schema='opal-paralog-github-history-v1', freeze_sha256=sha(freeze_path),
                  code_binding_sha256=sha(binding_path), discovery=discovery, trees=trees,
                  new_commit_identities=commits, stopped=stopped,
                  wall_seconds=time.perf_counter() - started,
                  requests=len(discovery) + len(trees),
                  discovery_pagination_incomplete=any(r.get('pagination_advertised', False) for r in discovery),
                  repository_blobs_opened=0, response_tables_or_headers_opened=False,
                  source_qualified=False, fits=0, scores=0)
    with output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(requests=result['requests'], trees=len(trees), stopped=stopped,
                          metadata_named_paths=sorted({p for r in trees for p in r.get('metadata_named_paths', [])}),
                          unseen_paths=sorted({p for r in trees for p in r.get('paths_unseen_at_baseline', [])}))))


if __name__ == '__main__':
    main()
