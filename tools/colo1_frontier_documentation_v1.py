"""Bounded primary documentation fetch; never execute study code or print responses."""
import hashlib
import json
import pathlib
import shutil
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = 'https://www.nature.com/articles/s41467-025-67256-9'
REG = ROOT / 'docs/frontier_execution/A_source_registration_v1.md'
OUT = ROOT / 'results/frontier-execution-20261026/A_source_requests_v1.json'
LOCAL = ROOT / 'tmp/colo1_frontier_v1'

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None

class Node:
    def __init__(self, tag='', attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []
    def text(self):
        if self.tag in {'script','style','table','figure','noscript','aside'}:
            return ''
        return ' '.join(x.text() if isinstance(x, Node) else x for x in self.children)
    def walk(self):
        yield self
        for x in self.children:
            if isinstance(x, Node):
                yield from x.walk()

class Document(HTMLParser):
    def __init__(self):
        super().__init__(); self.root=Node(); self.stack=[self.root]
    def handle_starttag(self, tag, attrs):
        n=Node(tag,attrs); self.stack[-1].children.append(n)
        if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:
            self.stack.append(n)
    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].tag==tag:
                del self.stack[i:]; break
    def handle_data(self, data):
        self.stack[-1].children.append(data)

def main():
    start=time.monotonic()
    receipt={'url':URL,'registration_sha256':hashlib.sha256(REG.read_bytes()).hexdigest(),
             'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'response_data_opened':False,'upstream_executed':False}
    LOCAL.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(ROOT).free < 1024**3 + 4*1024**2:
        raise RuntimeError('disk reserve')
    try:
        req=urllib.request.Request(URL,headers={'Accept-Encoding':'identity','User-Agent':'Opal-source-qualification/1.0'})
        with urllib.request.build_opener(NoRedirect()).open(req,timeout=30) as r:
            receipt.update(status=r.status,mime=r.headers.get_content_type())
            if r.status!=200 or r.headers.get_content_type()!='text/html':
                raise ValueError('status/MIME refusal before body')
            raw=r.read(4*1024**2+1)
        receipt.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if len(raw)>4*1024**2: raise ValueError('byte cap')
        (LOCAL/'article.html').write_bytes(raw)
        doc=Document(); doc.feed(raw.decode('utf-8'))
        selected=[]; links=[]; headings=[]
        for n in doc.root.walk():
            if n.tag!='section': continue
            direct=[c for c in n.children if isinstance(c,Node) and c.tag=='h2']
            title=' '.join(c.text() for c in direct).strip()
            if title: headings.append({'id':n.attrs.get('id'), 'heading':title})
            if title.lower() in {'methods','data availability','code availability'}:
                selected.append({'section':title,'text':' '.join(n.text().split())})
            if title.lower() in {'supplementary information','supplementary material'}:
                for a in n.walk():
                    if a.tag=='a' and a.attrs.get('href'):
                        links.append({'label':' '.join(a.text().split()),'href':a.attrs['href']})
        if sum(len(x['text'].encode()) for x in selected)>30*1024 or len(links)>80:
            raise ValueError('projection cap')
        view={'sections':selected,'supplementary_links':links}
        (LOCAL/'methods_projection.json').write_text(json.dumps(view,indent=2),encoding='utf-8')
        receipt.update(outcome='projected',section_headings=headings,projection_sha256=hashlib.sha256((LOCAL/'methods_projection.json').read_bytes()).hexdigest(),selected_sections=[x['section'] for x in selected],supplementary_links=links)
    except Exception as e:
        receipt.update(outcome='refused',error=repr(e))
        if isinstance(e,urllib.error.HTTPError): receipt['status']=e.code
    receipt['wall_seconds']=time.monotonic()-start
    receipt['code_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    OUT.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
