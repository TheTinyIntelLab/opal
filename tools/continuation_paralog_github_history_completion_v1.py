"""Complete the frozen filename menu; never retry the refused tree URL."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import continuation_paralog_github_history_v1 as history

ROOT, OUT = history.ROOT, history.OUT
FAILED = 'a1f64b8cd3e4517c5268c39f89f568193142f62c'
EMPTY = hashlib.sha1(b'tree 0\0').hexdigest()


def descriptor(value):
    if not isinstance(value, dict) or value.get('sha') != FAILED:
        raise ValueError('Commit descriptor identity mismatch')
    tree = value.get('tree')
    parents = value.get('parents')
    if not isinstance(tree, dict) or not isinstance(parents, list):
        raise ValueError('Malformed commit descriptor')
    tree_sha = history.identity(tree.get('sha'))
    parent_shas = []
    for parent in parents:
        if not isinstance(parent, dict):
            raise ValueError('Malformed commit parent')
        parent_shas.append(history.identity(parent.get('sha')))
    return dict(commit_sha=FAILED, tree_sha=tree_sha, parent_shas=parent_shas,
                empty_tree_by_git_identity=tree_sha == EMPTY)


def fetch_descriptor():
    row = dict(url=history.BASE + 'git/commits/' + FAILED,
               observed_utc=datetime.now(timezone.utc).isoformat())
    started = time.perf_counter()
    try:
        request = urllib.request.Request(row['url'], headers={
            'User-Agent': 'Opal-response-free-history-descriptor/1.0',
            'Accept': 'application/vnd.github+json'})
        with urllib.request.build_opener(history.NoRedirect()).open(request, timeout=30) as stream:
            row['status'] = stream.status
            if stream.status != 200 or stream.headers.get_content_type() != 'application/json':
                raise ValueError('Commit descriptor status/MIME refusal')
            raw = stream.read(2 * 1024**2 + 1)
        if len(raw) > 2 * 1024**2:
            raise ValueError('Commit descriptor byte ceiling')
        row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                   **descriptor(history.decode(raw)))
    except (urllib.error.URLError, ValueError, TimeoutError) as error:
        if isinstance(error, urllib.error.HTTPError):
            row['status'] = error.code
            error.close()
        row['error'] = type(error).__name__ + ': ' + str(error)
    row['elapsed_seconds'] = time.perf_counter() - started
    return row


def main():
    output = OUT / 'paralog-github-history-completion-receipt-v1.json'
    if output.exists():
        raise FileExistsError(output)
    freeze_path = OUT / 'paralog-github-history-completion-freeze-v1.json'
    binding_path = OUT / 'paralog-github-history-completion-code-binding-v1.json'
    freeze, binding = history.decode(freeze_path.read_bytes()), history.decode(binding_path.read_bytes())
    if history.sha(freeze_path) != binding['freeze_sha256']:
        raise ValueError('Completion freeze drift')
    checks = {**freeze['sha256'], **binding['code_sha256']}
    for path, expected in checks.items():
        if history.sha(ROOT / path) != expected:
            raise ValueError('Completion input drift: ' + path)
    if shutil.disk_usage(ROOT).free < 2**30:
        raise RuntimeError('Unchanged disk reserve')
    old = history.decode((OUT / 'paralog-github-history-receipt-v1.json').read_bytes())
    requested = {r['url'].split('/trees/')[1].split('?')[0] for r in old['trees']}
    remaining = sorted(set(old['new_commit_identities']) - requested)
    if len(remaining) != 5 or FAILED in remaining:
        raise ValueError('Unrequested finite list changed')
    started = time.perf_counter()
    commit = fetch_descriptor()
    trees = []
    stopped = None
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(remaining), 4):
            tasks = [('tree', 'git/trees/' + history.identity(s) + '?recursive=1') for s in remaining[start:start + 4]]
            wave = list(pool.map(history.fetch, tasks))
            trees.extend(wave)
            if any('error' in r for r in wave):
                stopped = 'Completion tree-wave refusal'
                break
    for path, expected in checks.items():
        if history.sha(ROOT / path) != expected:
            raise ValueError('Post-completion drift: ' + path)
    result = dict(schema='opal-paralog-github-history-completion-v1',
                  freeze_sha256=history.sha(freeze_path), code_binding_sha256=history.sha(binding_path),
                  descriptor=commit, remaining_fixed_identities=remaining, trees=trees,
                  stopped=stopped, wall_seconds=time.perf_counter() - started,
                  requests=1 + len(trees), failed_tree_retries=0,
                  repository_blobs_opened=0, response_tables_or_headers_opened=False,
                  source_qualified=False, fits=0, scores=0)
    with output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(requests=result['requests'], descriptor_status=commit.get('status'),
                          empty_tree_by_git_identity=commit.get('empty_tree_by_git_identity'),
                          descriptor_error=commit.get('error'), stopped=stopped,
                          metadata_named_paths=sorted({e['path'] for r in trees for e in r.get('entries', []) if 'metadata' in e['path'].lower()}))))


if __name__ == '__main__':
    main()
