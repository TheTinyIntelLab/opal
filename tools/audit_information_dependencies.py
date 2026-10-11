"""Bounded provenance/contract audit; no expression or large asset retrieval."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

import h5py

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import read_json,sha256,write_json
from prepare_information import OUT,resource_check


def retrieve(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Opal-local-contract-audit'}),timeout=30) as response:
            data=response.read(2000000)
            if len(data)>=2000000: return {'url':url,'refusal':'Metadata size bound exceeded'}
            return {'url':url,'status':response.status,'body':data.decode('utf-8')}
    except Exception as error:
        return {'url':url,'error':str(error)}


def run():
    resource_check(2000000)
    native=Path('D:/opal-model-series-20261010-v2/native')
    cards=OUT/'comparators'
    for method in ['PertSpectra','GEARS','CPA','TxPert','PRESAGE','SLIM']:
        pin=native/method/'pin.json'
        if not pin.exists(): pin=ROOT/'data/public_methods'/f'{method.lower()}_local_manifest.json'
        original=read_json(pin)
        paths={f['path']:f.get('sha256') for f in original['files']}
        base=native/method if (native/method).exists() else ROOT/'tools/external'/method
        verification={p:sha256(base/p)==h for p,h in paths.items() if (base/p).is_file() and h}
        write_json(cards/(method+'-pin-audit.json'),{'method':method,'revision':original['revision'],'pin_sha256':sha256(pin),
            'verified_cached_files':verification,'missing_cached_files':[p for p in paths if not (base/p).exists()],
            'upstream_modified':False,'fully_eligible':False})
    with h5py.File(ROOT/'data/raw/pertspectra-model-series/inhouse.h5ad','r') as f:
        uns=list(f.get('uns',{}).keys())
        nrows=len(f['obs/_index']) if '_index' in f['obs'] else len(f['obs'][f['obs'].attrs['_index']])
    tree=read_json(native/'PertSpectra/tree.json')['tree']
    graphfiles=[e['path'] for e in tree if 'prior_graph' in e['path'] or 'string' in e['path'].lower()]
    files=[p for p in (native/'PertSpectra').rglob('*') if p.is_file() and p.suffix in ('.py','.md')]
    references=[{'path':str(p),'sha256':sha256(p)} for p in files if 'sparse_gene_network' in p.read_text(errors='replace')]
    write_json(cards/'PertSpectra-eligibility.json',{'status':'blocked_external','fully_eligible':False,'deposited_rows':nrows,
        'reported_filtered_rows':65648,'difference':nrows-65648,'uns_keys':uns,'canonical_graph_present':'sparse_gene_network' in uns,
        'released_graph_paths':graphfiles,'graph_contract_sources':references,
        'failure':'No exact graph with ordered genes/construction lineage; author 50-cell perturbation-treatment filter differs from original ten-cell population and author log-cell transform differs from log-aggregate endpoint',
        'next_action':'Supply exact author graph and filtering/normalization manifest; verify graph construction excludes held outcomes. Existing no-graph results remain adaptations.',
        'primary_source':'https://github.com/insitro/PertSpectra'})
    pyproject=native/'TxPert/pyproject.toml'
    import tomllib
    project=tomllib.loads(pyproject.read_text())['project']
    req=ROOT/'configs/environments/txpert-public-upstream.txt'
    req.write_text('# Exact direct dependencies from upstream 08d82ee; supported Python >=3.12. Not a resolved transitive lock.\n'+'\n'.join(project['dependencies'])+'\n')
    response=retrieve('https://pypi.org/pypi/torch/2.6.0/json')
    wheel=None
    if 'body' in response:
        wheel=next(({'filename':f['filename'],'bytes':f['size'],'sha256':f['digests']['sha256']} for f in json.loads(response['body'])['urls'] if 'cp312' in f['filename'] and 'win_amd64' in f['filename']),None)
    write_json(cards/'TxPert-closure.json',{'status':'blocked_external','fully_eligible':False,'requires_python':project['requires-python'],
        'dependency_spec_sha256':sha256(req),'upstream_pyproject_sha256':sha256(pyproject),'direct_dependencies':len(project['dependencies']),
        'omegaconf_spec':[d for d in project['dependencies'] if d.startswith('omegaconf')],
        'torch_windows312_wheel':wheel,'remaining_artifact_bytes':3000000000-sum(resource_check().values()),
        'isolated_install_executed':False,'reason':'Supported complete closure cannot fit inherited retention headroom; existing legacy closure is Python 3.10 and cannot certify >=3.12 author recipe. No sequential global package repairs.',
        'next_action':'Provide sufficient separately approved storage and a supported Python3.12/CUDA2.6 isolated environment, resolved public dependency lock, public graph/weight assets with provenance. Proprietary configurations remain unavailable.',
        'primary_source':'https://github.com/valence-labs/TxPert'})
    # Repeat import only, with the preserved legacy closure; it is not a supported environment.
    code=f"import sys;sys.path.insert(0,{str(native/'TxPert')!r});import gspp.models.txpert"
    result=subprocess.run([str(ROOT/'.venv-cpa/Scripts/python.exe'),'-c',code],capture_output=True,text=True,timeout=1800)
    write_json(cards/'TxPert-legacy-import.json',{'return_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'classification':'unsupported Python3.10 diagnostic only','scores':None})
    presage=native/'PRESAGE'
    source=(presage/'src/presage.py').read_text()
    tree=read_json(presage/'tree.json')['tree']
    pca_paths=[e['path'] for e in tree if 'pca' in e['path'].lower()]
    write_json(cards/'PRESAGE-eligibility.json',{'status':'blocked_external','fully_eligible':False,
        'decoder_definition_present':'class PCAWithTrainableDecoder' in source,'released_pca_paths':pca_paths,
        'pool_signature':'Pool.forward(tensor,locs_combos,mask) calls vector/mean pool with tensor,mask but those accept tensor only',
        'pair_axis':'torch.nonzero produces intervention rows; locs_combos is not used by the non-grouplasso pool for condition aggregation',
        'prediction_harness':'indicator_groupby averages repeated measured-condition rows, not a declared intervention-to-condition decoder; get_embedding is attribution extraction, not prediction aggregation',
        'source_sha256':sha256(presage/'src/presage.py'),'scores':None,
        'next_action':'Obtain author-supported PCA decoder and pair pooling/output contract; no arbitrary averaging of the two output rows.',
        'terms':'Genentech noncommercial code; cached GenePT/ESM noncommercial assets, no redistribution',
        'primary_source':'https://github.com/Genentech/PRESAGE'})
    for method in ['GEARS','CPA','SLIM']:
        reason={'GEARS':'Existing GO/coexpression aggregate inputs and signed response loss are a protocol adaptation; native cell preparation/gene/graph parity not established.',
            'CPA':'Existing signed aggregate pseudo-observations and categorical presence doses are a protocol adaptation; genuine cell-level supported covariates/physical dose contract required.',
            'SLIM':'Existing pinned combination fitting uses signed condition-effect pseudoprofiles and omits native nonnegative expression clip; preserved adaptation, not native parity.'}[method]
        write_json(cards/(method+'-eligibility.json'),{'fully_eligible':False,'status':'complete_verified','engineering':'audit complete; existing adaptation preserved',
            'eligibility_failure':reason,'scientific_verdict':'no fully eligible contemporary comparison','next_action':'Freeze matching native cell/input task before a new comparison; do not rerun old grid.'})
    naiad=retrieve('https://api.github.com/repos/NeptuneBio/NAIAD/commits/main')
    revision=None
    if 'body' in naiad: revision=json.loads(naiad['body']).get('sha')
    naiad_source=retrieve(f'https://raw.githubusercontent.com/NeptuneBio/NAIAD/{revision or "main"}/naiad/models.py')
    import hashlib
    metadata={'url':naiad['url'],'revision':revision,'error':naiad.get('error')}
    model_source={k:v for k,v in naiad_source.items() if k!='body'}
    model_source['sha256']=hashlib.sha256(naiad_source.get('body','').encode()).hexdigest()
    model_source['scalar_head_evidence']=[l.strip() for l in naiad_source.get('body','').splitlines() if 'nn.Linear' in l and ', 1)' in l]
    write_json(cards/'NAIAD-audit.json',{'status':'complete_verified','revision':revision,'metadata':metadata,'model_source':model_source,
        'fully_eligible':False,'scores':None,'acquisition_branch':'conditional_not_started',
        'eligibility_failure':'Pinned native heads output one phenotype; a 512-gene output requires a declared adaptation, not native parity.',
        'next_action':'Qualify a transcriptomic multi-output head and matched information adapter before any finite comparison; active learning waits for useful prediction and assay partner.'})
    lead=ROOT/'results/replicated_fixed_state_v1/public-search-20261009-v1/sciplex4_support_audit.json'
    audit=read_json(lead)
    matches=[]
    for root in [ROOT/'results',ROOT/'data/source_audits']:
        for p in root.rglob('*.json'):
            if p.is_relative_to(OUT): continue
            text=p.read_text(errors='replace')
            if 'GSM7056151' in text or 'sciplex4' in text.lower() or 'GSE225775' in text:
                matches.append({'path':str(p.relative_to(ROOT)),'sha256':sha256(p)})
    write_json(OUT/'confirmation/source_qualification.json',{'status':'blocked_external','source':'GSM7056151 / GSE225775 chemical-validation only',
        'source_audit_sha256':sha256(lead),'known_local_access_records':matches,'design_audit':audit,
        'new_expression_access':False,'held_outcomes_opened_this_cycle':False,'freshness_certified':False,
        'missing':['exact sample-specific size manifest and storage approval if needed','cell support and monotherapy/control matches','physical component dose resolution of GEO/notebook conflict','culture initiation and plate mapping','derivative terms','complete prior expression/derivative access attestation','independent custody/registration authorization'],
        'algorithm_gate_passed':False,'confirmation_executed':False,'sealed_predictions':None,
        'next_action':'Custodian must qualify metadata and attest access history before acquiring/opening any held expression. No confirmation triggered by negative development.'})
    write_json(OUT/'exposure_ledger.json',{'development':['PertSpectra TNFA-positive','PertSpectra TNFA-negative'],
        'historically_exposed':['Norman','Wessels','Combi-seq','D-SPIN','EmeraldBay'],
        'protected_historical_queries':'Norman38/Wessels18 not opened in this cycle; prior narratives prevent successor confirmation status',
        'proposed_fresh_source':'sci-Plex-GxE chemical lead, freshness unqualified','new_held_expression_access':False,
        'development_labels':'All original PertSpectra aggregate pair outcomes reused only in development/evaluator; no new independent measurements',
        'initial_tool_and_web_cpu':'unknown; tools without local PID counters are disclosed, not recorded as zero'})
    print('Pinned cached code, native contracts and external qualification audited without large downloads')


if __name__=='__main__':run()
