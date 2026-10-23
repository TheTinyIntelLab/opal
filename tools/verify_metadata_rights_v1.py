"""Read-only custody audit of metadata rights/access evidence; no requests."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RECORDS=ROOT/'results/research-continuation'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):return json.loads((RECORDS/name).read_text(encoding='utf8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rights_freeze=read('metadata-rights-freeze-v1.json')
    aliases={'plan.md':RECORDS/'metadata-rights-plan-before-v1.md.txt'}
    for name,expected in rights_freeze['files_sha256'].items():
        assert sha(aliases.get(name,ROOT/name))==expected,name
    code=read('metadata-rights-code-binding-v1.json')
    assert code['freeze_sha256']==sha(RECORDS/'metadata-rights-freeze-v1.json')
    assert code['tool_sha256']==sha(ROOT/'tools/continuation_metadata_rights_v1.py')
    rights=read('metadata-rights-receipt-v1.json')
    assert rights['freeze_sha256']==code['freeze_sha256']
    assert len(rights['requests'])==6
    for row in rights['requests']:
        assert row['status']==200 and 'error' not in row
        path=Path(row['path']);assert path.is_relative_to(ROOT/'tmp/metadata-rights-v1')
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    inventory_freeze=read('paralog-inventory-freeze-v1.json')
    for name,expected in inventory_freeze['files_sha256'].items():assert sha(ROOT/name)==expected,name
    inventory_code=read('paralog-inventory-code-binding-v1.json')
    assert inventory_code['freeze_sha256']==sha(RECORDS/'paralog-inventory-freeze-v1.json')
    for name,expected in inventory_code['code_sha256'].items():
        path=RECORDS/'paralog-inventory-tests-failed-v1.py.txt' if name=='tests/test_continuation_paralog_inventory.py' else ROOT/name
        assert sha(path)==expected,name
    inventory=read('paralog-inventory-receipt-v1.json')
    assert inventory['freeze_sha256']==inventory_code['freeze_sha256']
    assert len(inventory['helpers'])==3
    for row in inventory['helpers']:
        raw=Path(row['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==row['sha256']
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert row['identity_verified'] and blob==row['git_blob_sha1']==row['expected_git_blob_sha1']
    asset=inventory['metadata_archive']
    assert asset['error']=='HTTPError: HTTP Error 403: Forbidden'
    assert asset['redirects']==[] and 'path' not in asset and 'inventory' not in asset
    assert not (ROOT/'tmp/paralog-inventory-v1/METADATA.tar.gz').exists()
    assert inventory['member_bodies_opened']==0 and not inventory['response_tables_or_headers_opened']
    assert not read('paralog-inventory-tests-v1.json')['passed']
    assert read('paralog-inventory-tests-v2.json')['passed']
    corrected=read('paralog-inventory-tests-binding-v2.json')
    for name,expected in corrected['code_sha256'].items():assert sha(ROOT/name)==expected,name
    result=dict(passed=True,requests_replayed=0,primary_request_records_verified=10,raw_documentation_objects_verified=9,pinned_helper_blobs_verified=3,source_archive_acquired=False,member_bodies_opened=0,network_access_blocked=True,source_qualified=False,independent_validation=False,executor='root read-only custody audit',historical_aliases={k:str(v.relative_to(ROOT)) for k,v in aliases.items()},biological_fits=0,new_biological_scores=0)
    with args.output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
