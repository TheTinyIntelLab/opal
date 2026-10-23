"""Six frozen primary documentation/catalog requests; no source assets."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
RECORDS=ROOT/'results/research-continuation'
QUARANTINE=ROOT/'tmp/metadata-rights-v1'
REQUESTS=[
    ('colo1-deposit','https://zenodo.org/api/records/17191951','json'),
    ('colo1-readme','https://raw.githubusercontent.com/ibarrioh/DualGuide_COLO1/cfb5eb9a39ff1ec43642fba9485270ae200e81ae/README.md','text'),
    ('colo1-input-roles','https://raw.githubusercontent.com/ibarrioh/DualGuide_COLO1/cfb5eb9a39ff1ec43642fba9485270ae200e81ae/Analysis/ENCORE/input/readme.txt','text'),
    ('paralog-deposit-v4','https://api.figshare.com/v2/articles/25954027/versions/4','json'),
    ('paralog-readme','https://raw.githubusercontent.com/team113sanger/harle_dgCRISPR_paralogs/fe14bb800f2766235096a655c2fb5da572808a8a/README.md','text'),
    ('paralog-license','https://raw.githubusercontent.com/team113sanger/harle_dgCRISPR_paralogs/fe14bb800f2766235096a655c2fb5da572808a8a/LICENSE','text'),
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):return None


def fetch(spec):
    name,url,kind=spec;started=time.perf_counter()
    row=dict(name=name,url=url,method='GET',purpose='primary rights and input-role documentation only')
    try:
        opener=urllib.request.build_opener(NoRedirect)
        request=urllib.request.Request(url,headers={'User-Agent':'Opal-response-free-metadata-audit/1.0'})
        with opener.open(request,timeout=30) as stream:
            content_type=stream.headers.get_content_type()
            allowed={'application/json'} if kind=='json' else {'text/plain','text/markdown'}
            if content_type not in allowed:raise ValueError('Unapproved documentation MIME: '+content_type)
            chunks=[];size=0
            while True:
                block=stream.read(min(65536,2*1024**2+1-size))
                if not block:break
                chunks.append(block);size+=len(block)
                if size>2*1024**2 or time.perf_counter()-started>30:raise RuntimeError('Registered byte/time cap')
            raw=b''.join(chunks);text=raw.decode('utf8')
            path=QUARANTINE/(name+('.json' if kind=='json' else '.txt'))
            with path.open('xb') as destination:destination.write(raw)
            row.update(status=stream.status,content_type=content_type,path=str(path),bytes=len(raw),sha256=sha(path))
            if kind=='json':
                document=json.loads(text)
                if name=='colo1-deposit':
                    m=document['metadata'];row['selected_catalog']=dict(title=m.get('title'),version=m.get('version'),license=m.get('license'),related_identifiers=m.get('related_identifiers'),description=m.get('description'),files=[{k:f.get(k) for k in ('key','size','checksum','links')} for f in document.get('files',[])])
                else:
                    row['selected_catalog']={k:document.get(k) for k in ('title','version','license','description','references','related_materials','url_public_html')}
                    row['selected_catalog']['files']=[{k:f.get(k) for k in ('id','name','size','download_url','computed_md5')} for f in document.get('files',[])]
            elif name=='paralog-license':
                row['selected_rights']=dict(mit_title=bool(re.search(r'\bMIT License\b',text)),permission_grant=bool('Permission is hereby granted, free of charge' in text),copyright_lines=[line for line in text.splitlines() if line.startswith('Copyright')])
            else:
                row['selected_documentary_lines']=[line for line in text.splitlines() if re.search(r'licen[cs]|copyright|rights|zenodo|figshare|METADATA|IDs_colo1|annotation|normaliz|[Dd]ata availability',line)][:40]
    except (urllib.error.URLError,ValueError,RuntimeError,TimeoutError) as error:
        row['error']=type(error).__name__+': '+str(error)
        if isinstance(error,urllib.error.HTTPError):row['status']=error.code
    row['elapsed_seconds']=time.perf_counter()-started
    return row


def main():
    freeze_path=RECORDS/'metadata-rights-freeze-v1.json'
    binding_path=RECORDS/'metadata-rights-code-binding-v1.json'
    freeze=json.loads(freeze_path.read_text(encoding='utf8'));binding=json.loads(binding_path.read_text(encoding='utf8'))
    if binding['freeze_sha256']!=sha(freeze_path) or binding['tool_sha256']!=sha(__file__):raise ValueError('Code/freeze binding mismatch')
    for name,expected in freeze['files_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen input drift: '+name)
    if shutil.disk_usage(ROOT).free<2**30:raise RuntimeError('Unchanged disk reserve')
    QUARANTINE.mkdir(parents=True,exist_ok=False)
    output=RECORDS/'metadata-rights-receipt-v1.json'
    if output.exists():raise FileExistsError(output)
    started=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,REQUESTS))
    if sum(r.get('bytes',0) for r in rows)>12*1024**2:raise RuntimeError('Aggregate byte cap')
    for name,expected in freeze['files_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Post-execution drift')
    write(output,dict(schema='opal-metadata-rights-receipt-v1',closed_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(freeze_path),tool_sha256=sha(__file__),requests=rows,wall_seconds=time.perf_counter()-started,source_assets_acquired=0,archive_or_rds_objects_opened=0,response_tables_or_headers_opened=False,fits=0,scores=0,timeout_enforcement='30s transport timeout plus cooperative total/stream checks; external120s process-tree watchdog',source_code_executed=False))
    print(json.dumps([dict(name=r['name'],status=r.get('status'),bytes=r.get('bytes'),error=r.get('error')) for r in rows]))


if __name__=='__main__':main()
