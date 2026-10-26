"""Methods-only JATS projection from the newly resolved exact manuscript."""
import hashlib
import json
import pathlib
import shutil
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
ROOT=pathlib.Path(__file__).resolve().parents[1]
REG=ROOT/'docs/frontier_execution/A_source_registration_v3.md'
LOCAL=ROOT/'tmp/colo1_frontier_v3'
OUT=ROOT/'results/frontier-execution-20261026/A_source_requests_v3.json'
URL='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12808759/fullTextXML'
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None
def prose(node):
    if node.tag in {'table-wrap','fig','supplementary-material','table','inline-supplementary-material'}: return ''
    return (node.text or '')+' '+''.join(prose(x)+(x.tail or '')+' ' for x in node)
def main():
    if OUT.exists(): raise RuntimeError('immutable receipt exists')
    LOCAL.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(ROOT).free<1024**3+2*1024**2: raise RuntimeError('reserve')
    start=time.monotonic(); rec={'url':URL,'registration_sha256':hashlib.sha256(REG.read_bytes()).hexdigest(),'response_data_opened':False,'upstream_executed':False,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    try:
        req=urllib.request.Request(URL,headers={'Accept-Encoding':'identity','User-Agent':'Opal-source-qualification/1.0'})
        with urllib.request.build_opener(NoRedirect()).open(req,timeout=30) as r:
            rec.update(status=r.status,mime=r.headers.get_content_type())
            if r.status!=200 or rec['mime'] not in {'application/xml','text/xml'}: raise ValueError('envelope')
            raw=r.read(2*1024**2+1)
        rec.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if len(raw)>2*1024**2: raise ValueError('byte cap')
        (LOCAL/'article.xml').write_bytes(raw)
        doc=ET.fromstring(raw); sections=[]
        for sec in list(doc.findall('./body/sec'))+list(doc.findall('./back/sec')):
            title=' '.join(sec.findtext('title','').split())
            if title.lower() in {'methods','materials and methods','data availability','code availability'}:
                selected=' '.join(prose(sec).split())
                if any(marker in selected for marker in ('[1]','## Warning:','Out[')): raise ValueError('output marker')
                sections.append({'title':title,'text':selected})
        licenses=[{'attributes':x.attrib,'text':' '.join(prose(x).split())} for x in doc.findall('./front/article-meta/permissions/license')]
        links=[]
        for supp in doc.iter('supplementary-material'):
            for n in supp.iter():
                href=n.attrib.get('{http://www.w3.org/1999/xlink}href')
                if href: links.append({'tag':n.tag,'href':href})
        if sum(len(x['text'].encode()) for x in sections)>48*1024 or len(links)>80: raise ValueError('projection cap')
        view={'sections':sections,'licenses':licenses,'supplementary_links':links}
        dest=LOCAL/'methods_projection.json'; dest.write_text(json.dumps(view,indent=2),encoding='utf-8')
        rec.update(outcome='projected',selected_sections=[x['title'] for x in sections],projection_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),licenses=licenses,supplementary_links=links)
    except Exception as exc:
        rec.update(outcome='refused',error=repr(exc))
        if isinstance(exc,urllib.error.HTTPError): rec['status']=exc.code
    rec.update(wall_seconds=time.monotonic()-start,code_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    OUT.write_text(json.dumps(rec,indent=2)+'\n',encoding='utf-8'); print(json.dumps(rec,indent=2))
if __name__=='__main__': main()
