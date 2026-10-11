"""Small pinned-source follow-up and strict PRESAGE output fixture."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from audit_information_dependencies import retrieve
from prepare_information import OUT,LOCAL,resource_check
from opal.artifacts import read_json,write_json,sha256


def run():
    resource_check(2000000)
    cards=OUT/'comparators'
    revision=read_json(cards/'PertSpectra-pin-audit.json')['revision']
    notebook=retrieve(f'https://raw.githubusercontent.com/insitro/PertSpectra/{revision}/data_preprocessing/inhouse_prior_graph_preprocessing.ipynb')
    utils=retrieve(f'https://raw.githubusercontent.com/insitro/PertSpectra/{revision}/utils.py')
    sources=LOCAL/'source-contract-snapshots';sources.mkdir(exist_ok=True)
    records=[]
    for name,result in [('inhouse-graph',notebook),('utils',utils)]:
        if 'body' in result:
            path=sources/(name+'.txt');path.write_text(result['body'])
            lines=result['body'].splitlines()
            if name=='inhouse-graph':
                lines=[line for cell in json.loads(result['body'])['cells'] if cell['cell_type']=='code' for line in cell['source']]
            records.append({'name':name,'url':result['url'],'sha256':sha256(path),'references':[l.strip() for l in lines if any(t in l.lower() for t in ['s3://','string','gene_network','read_csv','read_h5ad','filter_noisy_genes','filter_genes'])]})
        else: records.append(result)
    write_json(cards/'PertSpectra-graph-followup.json',{'records':records,'large_graph_assets_downloaded':False,'outcomes_accessed':False,'fully_eligible':False})
    # Sanitize the first API record, keeping original public response locally only.
    path=cards/'NAIAD-audit.json';record=read_json(path)
    if 'body' in record['metadata']:
        write_json(sources/'naiad-original-audit.json',record)
        source=record['model_source'];body=source.pop('body','')
        source['sha256']=hashlib.sha256(body.encode()).hexdigest()
        source['scalar_head_evidence']=[l.strip() for l in body.splitlines() if 'nn.Linear' in l and ', 1)' in l]
        record['metadata']={'url':record['metadata']['url'],'revision':record['revision']}
        record.update(status='complete_verified',eligibility_failure='Pinned native heads predict one scalar phenotype; transcriptomic output requires an adaptation.')
        write_json(path,record)
    print('Pinned graph recipe and NAIAD scalar-output contract inspected')


if __name__=='__main__':run()
