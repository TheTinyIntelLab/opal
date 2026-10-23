"""Bound metadata archive acquisition and TAR headers only; no extraction."""
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import continuation_metadata_rights_v1 as documentary

ROOT=Path(__file__).resolve().parents[1]
RECORDS=ROOT/'results/research-continuation'
QUARANTINE=ROOT/'tmp/paralog-inventory-v1'
PREFIX='https://raw.githubusercontent.com/team113sanger/harle_dgCRISPR_paralogs/fe14bb800f2766235096a655c2fb5da572808a8a/combinatorial_crispr_screen_analysis/SCRIPTS/'
HELPERS=[('paralog-dual-helper',PREFIX+'dual_guide/helper.R','4b2a0a1164f70d25f5cbe0eaedbf16d08d29eadb',3491),('paralog-single-helper',PREFIX+'single_guide/helper.R','5d59379e3e32bf355a7063f7095fd785f8940ff4',11960),('paralog-preprocessing-helper',PREFIX+'preprocessing/helper.R','aa02d92e0039959df70a46f8d2804ea54b2d5059',20354)]


class BoundedReader:
    def __init__(self,stream,limit=64*1024**2):self.stream,self.limit,self.count=stream,limit,0
    def read(self,size=-1):
        remaining=self.limit-self.count
        amount=remaining+1 if size<0 else min(size,remaining+1)
        block=self.stream.read(amount);self.count+=len(block)
        if self.count>self.limit:raise ValueError('Decompression cap exceeded')
        return block


def inventory(path):
    rows=[];declared=0
    with gzip.open(path,'rb') as source:
        bounded=BoundedReader(source)
        with tarfile.open(fileobj=bounded,mode='r|') as archive:
            for member in archive:
                name=PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or ':' in member.name or '\\' in member.name:raise ValueError('Unsafe archive member path')
                if not (member.isdir() or member.isfile()):raise ValueError('Link or special archive member')
                if member.size<0 or member.size>16*1024**2:raise ValueError('Archive member size cap')
                if member.isfile():declared+=member.size
                if declared>32*1024**2:raise ValueError('Aggregate declared byte cap')
                rows.append(dict(name=member.name,kind='directory' if member.isdir() else 'regular',bytes=member.size))
                if len(rows)>512:raise ValueError('Archive member count cap')
        return dict(members=rows,declared_regular_bytes=declared,decompressed_bytes_consumed=bounded.count,member_bodies_opened=0,extracted_files=0)


class AllowedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self):self.hops=[]
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        parsed=urllib.parse.urlsplit(newurl);host=(parsed.hostname or '').lower()
        if len(self.hops)>=2 or parsed.scheme!='https' or not (host.endswith('.figshare.com') or host=='figshare.com' or host.endswith('.amazonaws.com')):raise ValueError('Unapproved metadata redirect')
        self.hops.append(dict(status=code,host=host,path=parsed.path,query_redacted=bool(parsed.query)))
        return super().redirect_request(request,fp,code,msg,headers,newurl)


def download():
    started=time.perf_counter();redirect=AllowedRedirect();row=dict(url='https://ndownloader.figshare.com/files/46763992',expected_size=1141797,expected_md5='c6852831c477ca7d0a6d83a56e6c4ad8')
    try:
        opener=urllib.request.build_opener(redirect)
        request=urllib.request.Request(row['url'],headers={'User-Agent':'Opal-response-free-metadata-audit/1.0'})
        with opener.open(request,timeout=30) as stream:
            if stream.status!=200 or stream.headers.get_content_type() not in {'application/gzip','application/x-gzip','application/octet-stream','binary/octet-stream'}:raise ValueError('Unexpected metadata asset status/MIME')
            chunks=[];size=0
            while True:
                block=stream.read(min(65536,row['expected_size']+1-size))
                if not block:break
                chunks.append(block);size+=len(block)
                if size>row['expected_size'] or time.perf_counter()-started>30:raise ValueError('Metadata download byte/time cap')
            raw=b''.join(chunks)
        if len(raw)!=row['expected_size'] or hashlib.md5(raw).hexdigest()!=row['expected_md5']:raise ValueError('Metadata asset identity mismatch')
        path=QUARANTINE/'METADATA.tar.gz'
        with path.open('xb') as destination:destination.write(raw)
        row.update(bytes=len(raw),md5=hashlib.md5(raw).hexdigest(),sha256=hashlib.sha256(raw).hexdigest(),path=str(path),inventory=inventory(path))
    except (urllib.error.URLError,ValueError,RuntimeError,TimeoutError,tarfile.TarError,EOFError) as error:
        row['error']=type(error).__name__+': '+str(error)
    row.update(redirects=redirect.hops,elapsed_seconds=time.perf_counter()-started)
    return row


def main():
    freeze_path=RECORDS/'paralog-inventory-freeze-v1.json';binding_path=RECORDS/'paralog-inventory-code-binding-v1.json'
    freeze=json.loads(freeze_path.read_text(encoding='utf8'));binding=json.loads(binding_path.read_text(encoding='utf8'))
    if binding['freeze_sha256']!=documentary.sha(freeze_path):raise ValueError('Inventory freeze drift')
    for path,expected in binding['code_sha256'].items():
        if documentary.sha(ROOT/path)!=expected:raise ValueError('Inventory code drift')
    for path,expected in freeze['files_sha256'].items():
        if documentary.sha(ROOT/path)!=expected:raise ValueError('Inventory input drift')
    if shutil.disk_usage(ROOT).free<2**30:raise RuntimeError('Unchanged disk reserve')
    QUARANTINE.mkdir(parents=True,exist_ok=False);documentary.QUARANTINE=QUARANTINE
    started=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        helpers=list(pool.map(documentary.fetch,[(name,url,'text') for name,url,_,_ in HELPERS]))
    for row,(_,_,expected,size) in zip(helpers,HELPERS,strict=True):
        if 'path' in row:
            raw=Path(row['path']).read_bytes();blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            row.update(git_blob_sha1=blob,expected_git_blob_sha1=expected,identity_verified=blob==expected and len(raw)==size)
        else:row['identity_verified']=False
    asset=download()
    for path,expected in freeze['files_sha256'].items():
        if documentary.sha(ROOT/path)!=expected:raise ValueError('Post-inventory input drift')
    result=dict(schema='opal-paralog-inventory-receipt-v1',freeze_sha256=documentary.sha(freeze_path),code_binding_sha256=documentary.sha(binding_path),helpers=helpers,metadata_archive=asset,wall_seconds=time.perf_counter()-started,source_qualified=False,response_tables_or_headers_opened=False,member_bodies_opened=0,source_code_executed=False,fits=0,scores=0)
    documentary.write(RECORDS/'paralog-inventory-receipt-v1.json',result)
    print(json.dumps(dict(helper_identities=[r['identity_verified'] for r in helpers],archive_acquired='sha256' in asset,inventory_members=len(asset.get('inventory',{}).get('members',[])),error=asset.get('error'))))


if __name__=='__main__':main()
