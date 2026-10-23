"""Acquire one pinned pure-code object, exposing only registered metadata symbols."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.continuation_colo1_validator_v2 import NoRedirect, sha, unique_object

SYMBOL = re.compile(r'IDs_colo1|\b(?:ID|Note|MyNote|sgRNA1_ID|sgRNA2_ID|Gene1|Gene2)\b')
OUTPUT = re.compile(r'^\s*(?:#\s*)?(?:\[\d+\]|\d+\s+\S|A tibble|>)')
LITERAL_OUTCOME = re.compile(r'(?i)\b(?:lfc|fitness|score|gamma|phenotype)\s*=\s*(?:c\s*\(|[-+]?\d)')


def select_blob(tree, path, expected_size):
    if not isinstance(tree,dict) or tree.get('truncated') is not False:
        raise ValueError('Complete tree required')
    entries = tree.get('tree')
    if not isinstance(entries,list):
        raise ValueError('Tree entries required')
    seen = set()
    for entry in entries:
        if not isinstance(entry,dict) or not isinstance(entry.get('path'),str):
            raise ValueError('Malformed tree entry')
        if entry['path'] in seen:
            raise ValueError('Duplicate tree path')
        seen.add(entry['path'])
    matches = [e for e in entries if e['path']==path]
    if len(matches)!=1:
        raise ValueError('Unique exact code path required')
    entry = matches[0]
    if (entry.get('type')!='blob' or entry.get('mode')!='100644' or
        type(entry.get('size')) is not int or entry['size']!=expected_size or
        not re.fullmatch(r'[0-9a-f]{40}',entry.get('sha',''))):
        raise ValueError('Code object type/size/hash refused')
    return dict(path=path, bytes=entry['size'], git_blob_sha1=entry['sha'])


def verify_code(raw, entry, crc):
    prefix = b'blob '+str(len(raw)).encode()+b'\0'
    result = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                  git_blob_sha1=hashlib.sha1(prefix+raw).hexdigest(), crc32=zlib.crc32(raw)&0xffffffff)
    if (result['bytes']!=entry['bytes'] or result['git_blob_sha1']!=entry['git_blob_sha1'] or
        result['crc32']!=crc):
        raise ValueError('Code byte identity refused before decoding')
    return result


def metadata_view(source):
    selected = []
    for number,line in enumerate(source.splitlines(),1):
        if SYMBOL.search(line):
            if OUTPUT.search(line) or LITERAL_OUTCOME.search(line):
                raise ValueError('Output-like selected source line refused')
            selected.append((number,line))
    if len(selected)>200:
        raise ValueError('Selected metadata source line cap')
    display = '\n'.join(str(n)+': '+line for n,line in selected)+'\n'
    if len(display.encode('utf8'))>24*1024:
        raise ValueError('Selected metadata source byte cap')
    return selected,display


def fetch(url, cap, mime, audit):
    row = dict(url=url, body_bytes_read=0)
    audit.append(row)
    request = urllib.request.Request(url,headers={'User-Agent':'Opal-pinned-annotation-provenance/1.0',
                                                'Accept-Encoding':'identity'})
    started = time.perf_counter()
    try:
        stream = urllib.request.build_opener(NoRedirect()).open(request,timeout=30)
    except urllib.error.HTTPError as error:
        row.update(status=error.code,mime=error.headers.get_content_type())
        error.close()
        raise ValueError('Provenance HTTP status refused before body read') from error
    with stream:
        row.update(status=stream.status,mime=stream.headers.get_content_type())
        if (stream.status!=200 or stream.headers.get_content_type() not in mime or
            stream.headers.get('Content-Encoding','identity').lower()!='identity'):
            raise ValueError('Provenance envelope refused')
        raw=stream.read(cap+1)
        row['body_bytes_read']=len(raw)
    if len(raw)>cap:
        raise ValueError('Provenance body cap')
    row.update(raw_sha256=hashlib.sha256(raw).hexdigest(),elapsed_seconds=time.perf_counter()-started)
    return raw


def main():
    out=ROOT/'results/research-continuation'
    receipt=out/'colo1-provenance-receipt-v1.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    freeze_path=out/'colo1-provenance-freeze-v1.json'
    binding_path=out/'colo1-provenance-code-binding-v1.json'
    freeze=json.loads(freeze_path.read_bytes())
    binding=json.loads(binding_path.read_bytes())
    if sha(freeze_path)!=binding['freeze_sha256']:
        raise ValueError('Provenance freeze drift')
    checks={**freeze['repo_sha256'],**freeze['local_input_sha256'],**binding['code_sha256']}
    for path,expected in checks.items():
        if sha(ROOT/path)!=expected:
            raise ValueError('Provenance prerequisite drift: '+path)
    if shutil.disk_usage(ROOT).free<2**30:
        raise RuntimeError('Unchanged disk reserve')
    local=ROOT/'tmp/colo1-annotation-provenance-v1'
    local.mkdir(exist_ok=False)
    result=dict(schema='opal-colo1-annotation-provenance-receipt-v1',freeze_sha256=sha(freeze_path),
                code_binding_sha256=sha(binding_path),requests=[],source_code_executed=False,
                response_tables_or_headers_opened=False,metadata_values_aggregated=False,
                fits=0,scores=0,source_qualified=False,independent_review=False)
    stage='tree_catalog'
    try:
        url='https://api.github.com/repos/ibarrioh/DualGuide_COLO1/git/trees/'+freeze['commit']+'?recursive=1'
        raw=fetch(url,512*1024,{'application/json'},result['requests'])
        (local/'tree.json').write_bytes(raw)
        entry=select_blob(json.loads(raw,object_pairs_hook=unique_object),freeze['path'],freeze['expected_bytes'])
        result['selected_blob']=entry
        stage='code_identity'
        url='https://raw.githubusercontent.com/ibarrioh/DualGuide_COLO1/'+freeze['commit']+'/'+freeze['path']
        raw=fetch(url,freeze['expected_bytes'],{'text/plain','application/octet-stream'},result['requests'])
        result['verified_code']=verify_code(raw,entry,freeze['expected_crc32'])
        (local/'Rscript_final.R').write_bytes(raw)
        stage='metadata_source_projection'
        selected,display=metadata_view(raw.decode('utf8'))
        (local/'metadata-source-view.txt').write_text(display,encoding='utf8',newline='\n')
        result.update(status='metadata_source_view_ready',selected_line_numbers=[n for n,_ in selected],
                      selected_line_count=len(selected),metadata_view_sha256=sha(local/'metadata-source-view.txt'))
    except Exception as error:
        result.update(status='provenance_refused',refusal_stage=stage,error_type=type(error).__name__)
    for path,expected in checks.items():
        if sha(ROOT/path)!=expected:
            raise ValueError('Post-provenance drift: '+path)
    with receipt.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'],requests=len(result['requests']))))


if __name__=='__main__':
    main()
