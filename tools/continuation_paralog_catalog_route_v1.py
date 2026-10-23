"""One public catalog page; extract metadata-file links, never attachments."""
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research-continuation'
URL='https://figshare.com/articles/dataset/A_compendium_of_synthetic_lethal_gene_pairs_defined_by_extensive_combinatorial_pan-cancer_CRISPR_screening/25954027/4'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CanonicalRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self):self.hops=[]
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        parsed=urllib.parse.urlsplit(newurl)
        if self.hops or parsed.scheme!='https' or parsed.hostname not in {'figshare.com','www.figshare.com'} or '/25954027/4' not in parsed.path:raise ValueError('Unapproved catalog redirect')
        self.hops.append(dict(status=code,host=parsed.hostname,path=parsed.path))
        return super().redirect_request(request,fp,code,msg,headers,newurl)


def main():
    freeze_path=OUT/'paralog-catalog-route-freeze-v1.json';binding_path=OUT/'paralog-catalog-route-code-binding-v1.json'
    freeze=json.loads(freeze_path.read_text(encoding='utf8'));binding=json.loads(binding_path.read_text(encoding='utf8'))
    assert binding['freeze_sha256']==sha(freeze_path) and binding['tool_sha256']==sha(__file__)
    for path,expected in freeze['files_sha256'].items():assert sha(ROOT/path)==expected,path
    assert shutil.disk_usage(ROOT).free>=2**30
    output=OUT/'paralog-catalog-route-receipt-v1.json'
    if output.exists():raise FileExistsError(output)
    started=time.perf_counter();redirect=CanonicalRedirect();result=dict(url=URL,method='GET',source_assets_acquired=0,response_tables_or_headers_opened=False,fits=0,scores=0)
    try:
        request=urllib.request.Request(URL,headers={'User-Agent':'Opal-response-free-metadata-audit/1.0'})
        with urllib.request.build_opener(redirect).open(request,timeout=30) as stream:
            if stream.status!=200 or stream.headers.get_content_type()!='text/html':raise ValueError('Unexpected catalog status/MIME')
            chunks=[];size=0
            while True:
                block=stream.read(min(65536,2*1024**2+1-size))
                if not block:break
                chunks.append(block);size+=len(block)
                if size>2*1024**2 or time.perf_counter()-started>30:raise ValueError('Catalog time/byte cap')
            raw=b''.join(chunks)
        path=ROOT/'tmp/paralog-public-catalog-v1.html'
        with path.open('xb') as destination:destination.write(raw)
        text=html.unescape(raw.decode('utf8')).replace('\\/','/').replace('\\u002F','/')
        candidates=set(re.findall(r'https?://[^\s\"\'<>\\]+',text))
        candidates.update(urllib.parse.urljoin(URL,u) for u in re.findall(r'href=[\"\']([^\"\']+)',text))
        links=[]
        for candidate in sorted(candidates):
            if '46763992' not in candidate and 'METADATA.tar.gz' not in candidate:continue
            parsed=urllib.parse.urlsplit(candidate)
            links.append(dict(host=parsed.hostname,path=parsed.path,query_redacted=bool(parsed.query),same_failed_route=(parsed.hostname=='ndownloader.figshare.com' and parsed.path.rstrip('/')=='/files/46763992')))
        result.update(status=200,bytes=len(raw),path=str(path),sha256=sha(path),metadata_file_links=links,independently_documented_different_route=any(not link['same_failed_route'] for link in links),attachment_requests=0)
    except (urllib.error.URLError,ValueError,RuntimeError,TimeoutError) as error:
        result['error']=type(error).__name__+': '+str(error)
        if isinstance(error,urllib.error.HTTPError):result['status']=error.code
    result.update(freeze_sha256=sha(freeze_path),tool_sha256=sha(__file__),redirects=redirect.hops,elapsed_seconds=time.perf_counter()-started,source_qualified=False)
    with output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','error','metadata_file_links','independently_documented_different_route') if k in result}))


if __name__=='__main__':main()
