"""Frozen five-request documentation and exact DOI catalog acquisition."""
import hashlib
import json
import pathlib
import shutil
import time
import urllib.error
import urllib.request

ROOT=pathlib.Path(__file__).resolve().parents[1]
REG=ROOT/'docs/frontier_execution/A_source_registration_v2.md'
LOCAL=ROOT/'tmp/colo1_frontier_v2'
OUT=ROOT/'results/frontier-execution-20261026/A_source_requests_v2.json'
COMMIT='cfb5eb9a39ff1ec43642fba9485270ae200e81ae'
PATHS=['README.md','Analysis/readme.txt','Analysis/ENCORE/readme.txt','Analysis/ENCORE/input/readme.txt']
INDEX='https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:10.1038/s41467-025-67256-9&format=json'
FIELDS={'id','source','pmid','pmcid','doi','title','authorString','journalTitle','pubYear','isOpenAccess','inEPMC','hasSuppl'}
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None
def main():
    if OUT.exists(): raise RuntimeError('receipt exists; no repeated acquisitions')
    LOCAL.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(ROOT).free<1024**3+640*1024: raise RuntimeError('disk reserve')
    tree=json.loads((ROOT/'tmp/colo1-annotation-provenance-v1/tree.json').read_bytes())['tree']
    records=[]; start=time.monotonic()
    for path in PATHS+[None]:
        url=f'https://raw.githubusercontent.com/ibarrioh/DualGuide_COLO1/{COMMIT}/{path}' if path else INDEX
        rec={'url':url}; then=time.monotonic()
        try:
            request=urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'Opal-source-qualification/1.0'})
            with urllib.request.build_opener(NoRedirect()).open(request,timeout=30) as response:
                rec.update(status=response.status,mime=response.headers.get_content_type())
                expected_mime='text/plain' if path else 'application/json'
                if response.status!=200 or rec['mime']!=expected_mime: raise ValueError('envelope refusal')
                raw=response.read(128*1024+1)
            rec.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            if len(raw)>128*1024: raise ValueError('byte cap')
            if path:
                matches=[x for x in tree if x['path']==path and x['type']=='blob']
                actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
                if len(matches)!=1 or matches[0]['sha']!=actual or matches[0]['size']!=len(raw): raise ValueError('pinned blob mismatch')
                rec.update(git_blob_sha1=actual,path=path,outcome='verified_documentation')
                dest=LOCAL/(path.replace('/','__'))
                dest.write_bytes(raw)
            else:
                doc=json.loads(raw); result=doc['resultList']['result']
                projection=[{k:v for k,v in x.items() if k in FIELDS} for x in result]
                rec.update(outcome='projected_article_catalog',articles=projection)
                (LOCAL/'article_index.json').write_bytes(raw)
        except Exception as exc:
            rec.update(outcome='refused',error=repr(exc))
            if isinstance(exc,urllib.error.HTTPError): rec['status']=exc.code
        rec['wall_seconds']=time.monotonic()-then; records.append(rec)
    receipt={'registration_sha256':hashlib.sha256(REG.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'requests':records,'wall_seconds':time.monotonic()-start,'response_data_opened':False,'upstream_executed':False}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))
if __name__=='__main__': main()
