"""Bounded metadata-only Combi-seq and bulk-RNA source qualification."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import sys
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.replicated.contracts import sha,write_json

URLS={
 'combi_geo.soft.gz':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE174nnn/GSE174695/soft/GSE174695_family.soft.gz',
 'combi_treatments.csv.gz':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE174nnn/GSE174695/suppl/GSE174695_large_scale_meta.csv.gz',
 'backup_geo.soft.gz':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149428/soft/GSE149428_family.soft.gz',
 'combi_repo.json':'https://api.github.com/repos/bence-szalai/Combi-Seq-analysis/contents'}


def analyze_combi(path):
    rows=list(csv.DictReader(gzip.decompress(path.read_bytes()).decode().splitlines()))
    runs={};matched={};all_pairs=set();degree=Counter()
    for run in sorted({r['Run'] for r in rows}):
        rr=[r for r in rows if r['Run']==run]
        singles={c for r in rr if (r['Drug1']=='DMSO')!=(r['Drug2']=='DMSO') for c in (r['Drug1'],r['Drug2']) if c!='DMSO'}
        pairs={tuple(sorted((r['Drug1'],r['Drug2']))) for r in rr if 'DMSO' not in (r['Drug1'],r['Drug2']) and r['Drug1']!=r['Drug2']}
        all_pairs|=pairs
        runs[run]={'treatment_rows':len(rr),'distinct_nonrepeat_pairs':len(pairs),'distinct_single_identities':len(singles),
                   'identity_matched_pairs':sum(set(p)<=singles for p in pairs),
                   'repeated_target_rows':sum(r['Drug1']==r['Drug2']!='DMSO' for r in rr),
                   'control_rows':sum(r['Drug1']==r['Drug2']=='DMSO' for r in rr),
                   'ordered_nonrepeat_rows':sum('DMSO' not in (r['Drug1'],r['Drug2']) and r['Drug1']!=r['Drug2'] for r in rr),
                   'monotherapy_rows':sum((r['Drug1']=='DMSO')!=(r['Drug2']=='DMSO') for r in rr)}
        controls=[r[''] for r in rr if r['Drug1']==r['Drug2']=='DMSO']
        for r in rr:
            pair=tuple(sorted((r['Drug1'],r['Drug2'])))
            if pair not in pairs:continue
            matched[r['']]={'run':run,'canonical_pair':list(pair),'controls':controls,
               'single_a':[s[''] for s in rr if set((s['Drug1'],s['Drug2']))=={pair[0],'DMSO'}],
               'single_b':[s[''] for s in rr if set((s['Drug1'],s['Drug2']))=={pair[1],'DMSO'}],
               'exact_dose_match_verified':False,'culture_match_verified':False}
    for pair in all_pairs:degree.update(pair)
    import numpy as np
    return {'source':'Combi-seq','accession':'GSE174695','superseries':'GSE174696',
            'readout':'aggregate treatment-barcode aligned RNA reads','organism':'Homo sapiens','state':'K562',
            'treatment_time':'12h from series/sample protocols','run_counts':runs,
            'distinct_pair_identities':len(all_pairs),'distinct_components':len(degree),
            'partner_degree':dict(degree),'degree_summary':{'minimum':min(degree.values()),'median':float(np.median(list(degree.values()))),'maximum':max(degree.values())},
            'fraction_possible_pairs':len(all_pairs)/(len(degree)*(len(degree)-1)/2),
            'potential_reference_matches':matched,'eligible_exact_dose_culture_pairs':None,
            'independent_culture_initiation':'unknown; run labels are not culture evidence',
            'gene_universe':'not inspected; source matrix removes all-zero genes',
            'expression_outcomes_opened':False,'prior_owner_exposure':'unknown',
            'terms':{'counts':'unresolved','metadata':'primary GEO metadata, attribution retained',
                     'prepared_banks':'unresolved, do not redistribute','fitted_exports':'unresolved, do not redistribute',
                     'analysis_code':'repository contains LICENSE; no upstream code copied'},
            'role':'metadata_qualified_lead_only','accepted_for_confirmation':False,
            'blocked_on':['component-specific doses and vehicle/ordered-variant contract','independent initiation provenance',
                          'owner exposure declaration','data derivative terms','empirical endpoint and precision qualification']}


def analyze_backup(path):
    text=gzip.decompress(path.read_bytes()).decode()
    samples=[]
    for chunk in text.split('^SAMPLE = ')[1:]:
        title=next(l.split(' = ',1)[1] for l in chunk.splitlines() if l.startswith('!Sample_title'))
        chars=[l.split(' = ',1)[1] for l in chunk.splitlines() if l.startswith('!Sample_characteristics')]
        samples.append({'sample_id':chunk.splitlines()[0],'title':title,'characteristics':chars})
    return {'source':'El Diaz bulk RNA','accession':'GSE149428','sample_count':len(samples),
            'samples':samples,'expression_outcomes_opened':False,'role':'measurement_diagnostic_backup',
            'accepted_for_main_superiority':False,'reason':'small pair panel; assay and biological initiation need separate qualification',
            'independent_cultures_verified':False,'terms':'unresolved for prepared derivatives','prior_owner_exposure':'unknown'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshots',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fetch',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior source qualification records')
    a.snapshots.mkdir(parents=True,exist_ok=True)
    if a.fetch:
        for name,url in URLS.items():
            dest=a.snapshots/name
            if dest.exists():continue
            with urlopen(Request(url,headers={'User-Agent':'Opal metadata qualification'}),timeout=30) as stream:
                payload=stream.read(2000001)
            if len(payload)>2000000:raise ValueError('Metadata download exceeds bounded size')
            dest.write_bytes(payload)
    manifest={'retrieval_utc':datetime.now(timezone.utc).isoformat(),'metadata_only':True,'expression_outcomes_opened':False,
              'files':{name:{'url':url,'sha256':sha(a.snapshots/name),'bytes':(a.snapshots/name).stat().st_size} for name,url in URLS.items()}}
    write_json(a.output/'source_manifest.json',manifest)
    combi=analyze_combi(a.snapshots/'combi_treatments.csv.gz')
    combi['source_manifest_sha256']=sha(a.output/'source_manifest.json')
    write_json(a.output/'qualification.json',combi)
    write_json(a.output/'bulk_backup.json',analyze_backup(a.snapshots/'backup_geo.soft.gz'))
    write_json(a.output/'candidate_inventory.json',{'Combi-seq':'metadata-qualified lead, confirmation blocked',
        'GSE149428':'small-panel bulk measurement diagnostic, main superiority rejected',
        'CoMPoSE':'deposit and replicate provenance unresolved, do not fit',
        'Arc challenge':'matched pair-response panel unestablished, defer',
        'OP3 GSE279945':'donor context lead; matched combination panel unestablished, defer',
        'Norman/Wessels/EmeraldBay/sci-Plex/D-SPIN/Adamson':'previously exposed, rehearsal only'})
    print(json.dumps({'source':'Combi-seq','distinct_pairs':combi['distinct_pair_identities'],'confirmation_accepted':False},indent=2))


if __name__=='__main__':main()
