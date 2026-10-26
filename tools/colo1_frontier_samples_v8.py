"""Exact-accession ENA metadata, never read files or numerical outputs."""
import hashlib,json,pathlib,shutil,time,urllib.request,urllib.error,xml.etree.ElementTree as ET
ROOT=pathlib.Path(__file__).resolve().parents[1]
REG=ROOT/'docs/frontier_execution/A_source_registration_v8.md'
OUT=ROOT/'results/frontier-execution-20261026/A_source_requests_v8.json'
LOCAL=ROOT/'tmp/colo1_frontier_v8'
IDS=['SAMEA120519615','SAMEA120519616','SAMEA120519617','SAMEA120519624','ERX15302731','ERX15302404']
TAGS={'cell line','cell_line','cell type','sample name','time point','timepoint','replicate','biological replicate','technical replicate','transduction','culture','experimental condition','library','passage'}
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): return None
def main():
    if OUT.exists(): raise RuntimeError('immutable receipt')
    if shutil.disk_usage(ROOT).free<1024**3+384*1024: raise RuntimeError('reserve')
    LOCAL.mkdir(parents=True,exist_ok=True); records=[]; start=time.monotonic()
    for accession in IDS:
        url=f'https://www.ebi.ac.uk/ena/browser/api/xml/{accession}?includeLinks=false'
        then=time.monotonic(); rec={'accession':accession,'url':url}
        try:
            req=urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'Opal-source-qualification/1.0'})
            with urllib.request.build_opener(NoRedirect()).open(req,timeout=20) as r:
                rec.update(status=r.status,mime=r.headers.get_content_type())
                if r.status!=200 or rec['mime'] not in {'application/xml','text/xml'}: raise ValueError('envelope')
                raw=r.read(64*1024+1)
            rec.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            if len(raw)>64*1024: raise ValueError('byte cap')
            (LOCAL/f'{accession}.xml').write_bytes(raw); doc=ET.fromstring(raw)
            view={}; allowed={'TITLE','DESCRIPTION','DESIGN_DESCRIPTION','LIBRARY_CONSTRUCTION_PROTOCOL','SAMPLE_DESCRIPTOR','LIBRARY_NAME','LIBRARY_STRATEGY','LIBRARY_SOURCE','LIBRARY_SELECTION','LIBRARY_LAYOUT'}
            for n in doc.iter():
                if n.tag in {'SAMPLE','EXPERIMENT'}: view[n.tag+'_identity']=n.attrib
                if n.tag in allowed: view.setdefault(n.tag,[]).append({'attributes':n.attrib,'text':' '.join(' '.join(n.itertext()).split())})
            attrs=[]
            for n in doc.iter('SAMPLE_ATTRIBUTE'):
                tag=n.findtext('TAG',''); a={'tag':tag}
                if tag.lower() in TAGS: a['value']=n.findtext('VALUE','')
                attrs.append(a)
            view['sample_attributes']=attrs
            if len(json.dumps(view).encode())>8192: raise ValueError('projection cap')
            rec.update(outcome='projected_metadata',metadata=view)
        except Exception as exc:
            rec.update(outcome='refused',error=repr(exc))
            if isinstance(exc,urllib.error.HTTPError): rec['status']=exc.code
        rec['wall_seconds']=time.monotonic()-then; records.append(rec)
    out={'registration_sha256':hashlib.sha256(REG.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'requests':records,'wall_seconds':time.monotonic()-start,'response_data_opened':False,'upstream_executed':False}
    OUT.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
