"""Read primary methods and selected supplements, never GEO expression matrices."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
from urllib.request import urlopen,Request
import xml.etree.ElementTree as ET
import zipfile
from urllib.error import URLError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.replicated.contracts import sha,write_json

ARTICLE='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9343464/fullTextXML'
SUPPLEMENTS='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9343464/supplementaryFiles'
CDN='https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-32197-0/MediaObjects/'
SELECTED={'41467_2022_32197_MOESM1_ESM.pdf','41467_2022_32197_MOESM2_ESM.pdf',
          '41467_2022_32197_MOESM4_ESM.xlsx'}


def download(url,path,limit):
    if path.exists():return
    partial=path.with_suffix(path.suffix+'.partial')
    total=0
    with urlopen(Request(url,headers={'User-Agent':'Opal primary-methods audit'}),timeout=60) as stream,partial.open('wb') as dest:
        while chunk:=stream.read(65536):
            total+=len(chunk)
            if total>limit:raise ValueError('Source download exceeds declared bound')
            dest.write(chunk)
    partial.replace(path)


def workbook_cells(path):
    """Read metadata cells without executing formulas, macros or workbook code."""
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        shared=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared=[''.join(e.itertext()) for e in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        result={}
        for name in sorted(archive.namelist()):
            if not name.startswith('xl/worksheets/sheet') or not name.endswith('.xml'):continue
            rows=[]
            for row in ET.fromstring(archive.read(name)).findall('.//m:row',ns):
                cells={}
                for cell in row.findall('m:c',ns):
                    value=cell.find('m:v',ns)
                    text=value.text if value is not None else ''.join(cell.find('m:is',ns).itertext()) if cell.find('m:is',ns) is not None else ''
                    if cell.get('t')=='s':text=shared[int(text)]
                    cells[cell.get('r')]=text
                rows.append(cells)
            result[name]=rows
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fetch',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior source audits')
    args.snapshots.mkdir(parents=True,exist_ok=True)
    article=args.snapshots/'combi_article.xml';supplements=args.snapshots/'combi_supplements.zip'
    retrieval_failures=[]
    if args.fetch:
        download(ARTICLE,article,2_000_000)
        for name in sorted(SELECTED):
            try:download(CDN+name,args.snapshots/name,8_000_000)
            except (URLError,TimeoutError,OSError) as exc:
                retrieval_failures.append({'url':CDN+name,'error_type':type(exc).__name__,
                                           'consequence':'Selected supplementary metadata unavailable'})
    root=ET.parse(article).getroot()
    license_element=root.find('.//license')
    links=[e.get('{http://www.w3.org/1999/xlink}href') for e in license_element.iter('ext-link')] if license_element is not None else []
    files={'article':{'url':ARTICLE,'sha256':sha(article),'bytes':article.stat().st_size}}
    supplement_inventory=[];cells={}
    if supplements.exists():
        with zipfile.ZipFile(supplements) as archive:
            for info in archive.infolist():
                name=Path(info.filename).name
                supplement_inventory.append({'name':name,'bytes':info.file_size,'opened':name in SELECTED})
                if name not in SELECTED:continue
                if info.file_size>8_000_000:raise ValueError('Selected supplementary metadata is too large')
                dest=args.snapshots/name
                if not dest.exists():dest.write_bytes(archive.read(info))
                files[name]={'url':SUPPLEMENTS,'member':info.filename,'sha256':sha(dest),'bytes':dest.stat().st_size}
                if name.endswith('.xlsx'):cells=workbook_cells(dest)
    for name in sorted(SELECTED):
        dest=args.snapshots/name
        if not dest.exists():continue
        files[name]={'url':CDN+name,'sha256':sha(dest),'bytes':dest.stat().st_size}
        if name.endswith('.xlsx'):cells=workbook_cells(dest)
    record={'retrieval_utc':datetime.now(timezone.utc).isoformat(),'source':'Combi-seq',
        'source_files':files,'supplement_inventory':supplement_inventory,'retrieval_failures':retrieval_failures,
        'article_license_links':links,'expression_matrices_downloaded_or_opened':False,
        'published_methods_and_conclusions_inspected':True,
        'replication_evidence':'Primary methods describe three repetitions of the complete screen workflow.',
        'published_biological_replicates_reported':3 if '41467_2022_32197_MOESM2_ESM.pdf' in files else None,
        'reporting_summary_review':'Page 2 reports three biological replicates for each small/large-screen combination. It reports no randomization; drug allocation was based on GR35.' if '41467_2022_32197_MOESM2_ESM.pdf' in files else None,
        'independent_culture_initiation_verified':False,
        'independence_limit':'Separate initiation, passage and culture-to-run records are not established by a repeated-workflow statement.',
        'dose_evidence':'Methods identify single-drug GR35 calibration; supplementary assignments require explicit mapping before eligibility.',
        'analysis_warning':'Published analysis includes z-scoring and ComBat across replicates; that corrected representation must not be used as independent-culture raw measurement evidence.',
        'data_derivative_terms':'Article/supplement license does not automatically establish GEO count or fitted-export redistribution terms.',
        'analysis_code_license':'GPL-3.0 in the primary repository; no upstream code copied',
        'supplement_metadata_cells':cells,'accepted_for_confirmation':False}
    mapping={'EPIRUBICIN':'Epirubicin','Fludarabine':'Fludarabine Phosphate','H7':'H-7 dihydrochloride',
        'Hymenialdisine':'10Z-Hymenialdisine','Iodotubercidine':'5-Iodotubercidine','Methotrexate':'Methotrexat',
        'Mitomycin':'Mitomycin C','NMS1286937':'NMS1286937 (Onvansertib)','Olomucin':'Olomoucine',
        'PF562271':'PF-562271','SB747651':'SB747651A','SF1126':'SF-1126','Sunitinib':'Sunitinib Malate',
        'Tabloid':'Tabloid (Thioguanine)'}
    annotations={}
    for sheet,rows in cells.items():
        for row in rows:
            lookup={''.join(c for c in key if c.isalpha()):value for key,value in row.items()}
            if not lookup.get('A') or not lookup.get('B'):continue
            try:gr35=float(lookup['B'])
            except ValueError:continue
            annotations[lookup['A']]={'reported_GR35':gr35,'GR35_units_verified':False,
                'ChEMBL_ID':lookup.get('F'),'provider':lookup.get('G'),'worksheet':sheet,
                'worksheet_cell':next(key for key in row if key.startswith('B'))}
    import csv,gzip
    metadata=args.snapshots/'combi_treatments.csv.gz'
    if metadata.exists():
        treatments=list(csv.DictReader(gzip.decompress(metadata.read_bytes()).decode().splitlines()))
        components=sorted({row[k] for row in treatments for k in ('Drug1','Drug2')}-{ 'DMSO'})
        record['component_annotation_mapping']={component:{'supplement_name':mapping.get(component,component),
            'mapping_status':'proposed identity mapping; dose/lot assignment not verified',
            **annotations.get(mapping.get(component,component),{})} for component in components}
        record['components_with_reported_GR35']=sum('reported_GR35' in v for v in record['component_annotation_mapping'].values())
    record['qualification_role']='replicated diagnostic lead; exact-dose and independent-initiation confirmation remain unqualified'
    record['disjoint_culture_allocation_limit']='Only three published runs exist; they cannot supply both three development and three independent confirmation culture units.'
    write_json(args.output/'primary_methods.json',record)
    print({'output':str(args.output),'selected_source_files':len(files),'expression_outcomes_opened':False})


if __name__=='__main__':main()
