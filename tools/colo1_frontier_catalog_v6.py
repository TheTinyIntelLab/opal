"""Acquire only newly documented catalog and allowlisted ENA run metadata."""
import csv
import hashlib
import io
import json
import pathlib
import shutil
import time
import urllib.request
import urllib.error
ROOT=pathlib.Path(__file__).resolve().parents[1]
REG=ROOT/'docs/frontier_execution/A_source_registration_v6.md'
OUT=ROOT/'results/frontier-execution-20261026/A_source_requests_v6.json'
LOCAL=ROOT/'tmp/colo1_frontier_v6'
FIELDS='study_accession,sample_accession,experiment_accession,run_accession,sample_alias,experiment_alias,run_alias,library_name,library_strategy,library_source,library_selection,library_layout,instrument_platform,instrument_model'
URLS=['https://api.figshare.com/v2/articles/25533091/versions/1','https://www.ebi.ac.uk/ena/portal/api/filereport?accession=ERP183979&result=read_run&fields='+FIELDS+'&format=tsv']
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None
def main():
    if OUT.exists(): raise RuntimeError('immutable receipt exists')
    if shutil.disk_usage(ROOT).free<1024**3+512*1024: raise RuntimeError('reserve')
    LOCAL.mkdir(parents=True,exist_ok=True); records=[]; start=time.monotonic()
    for i,url in enumerate(URLS):
        then=time.monotonic(); rec={'url':url}
        try:
            req=urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'Opal-source-qualification/1.0'})
            with urllib.request.build_opener(NoRedirect()).open(req,timeout=30) as r:
                rec.update(status=r.status,mime=r.headers.get_content_type())
                if r.status!=200 or rec['mime'] not in ({'application/json'} if i==0 else {'text/plain','text/tab-separated-values'}): raise ValueError('envelope')
                raw=r.read(256*1024+1)
            rec.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            if len(raw)>256*1024: raise ValueError('byte cap')
            (LOCAL/f'item_{i}.raw').write_bytes(raw)
            if i==0:
                doc=json.loads(raw); keys={'id','title','version','doi','url','license'}
                projection={k:v for k,v in doc.items() if k in keys}
                projection['files']=[{k:v for k,v in f.items() if k in {'name','id','size','computed_md5','download_url','is_link_only'}} for f in doc.get('files',[])]
                rec.update(outcome='projected_catalog',catalog=projection)
            else:
                reader=csv.DictReader(io.StringIO(raw.decode()),delimiter='\t')
                if reader.fieldnames!=FIELDS.split(','): raise ValueError('metadata schema')
                rows=list(reader)
                if len(rows)>500: raise ValueError('row cap')
                rec.update(outcome='projected_run_metadata',metadata_rows=rows)
        except Exception as exc:
            rec.update(outcome='refused',error=repr(exc))
            if isinstance(exc,urllib.error.HTTPError): rec['status']=exc.code
        rec['wall_seconds']=time.monotonic()-then; records.append(rec)
    receipt={'registration_sha256':hashlib.sha256(REG.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'requests':records,'wall_seconds':time.monotonic()-start,'response_data_opened':False,'upstream_executed':False}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8'); print(json.dumps(receipt,indent=2))
if __name__=='__main__': main()
