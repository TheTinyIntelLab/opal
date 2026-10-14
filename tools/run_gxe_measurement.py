"""Technical-half public measurement pilot, with no independent-culture claim."""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import psutil
from opal.artifacts import read_json,write_json,sha256
from opal.gxe_measurement import assignment,rt_context,matrix_blocks,key,reference_variance


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=False);config=read_json(ROOT/'configs/experiments/opal_gxe_measurement_v1.json');start=time.process_time();wall=time.monotonic();peak=0
    def check():
        nonlocal peak
        peak=max(peak,psutil.Process().memory_info().rss)
        if time.process_time()-start>config['budgets']['cpu_seconds'] or peak>config['budgets']['ram_bytes']:
            raise RuntimeError('Accepted CPU/RAM measurement stop')
        if shutil.disk_usage(args.output).free<config['budgets']['reserve_bytes']:
            raise RuntimeError('Writable reserve measurement stop')
        if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>config['budgets']['output_bytes']:
            raise RuntimeError('Retained output measurement stop')
    try:
        receipts=read_json(args.input/'source_receipts.json')
        for record in receipts['files']:
            if sha256(args.input/record['path'])!=record['sha256']:raise ValueError('Official source identity changed')
        write_json(args.output/'registration.json',dict(config=config,source_receipts=receipts,
            config_sha256=sha256(ROOT/'configs/experiments/opal_gxe_measurement_v1.json'),
            code_sha256=sha256(__file__),helper_sha256=sha256(ROOT/'src/gxe_measurement.py'),
            owner_execution_authorized=True,scientific_fitting=False,independent_culture_initiation_verified=False))
        with gzip.open(args.input/'cells.txt.gz','rt') as f:cells=[line.split('\t')[0].strip() for line in f]
        with gzip.open(args.input/'genes.txt.gz','rt') as f:genes=[line.split('\t')[0].strip() for line in f]
        if len(set(cells))!=len(cells) or len(set(genes))!=len(genes):raise ValueError('Literal assay identifiers must be unique')
        lookup={c:i for i,c in enumerate(cells)};contexts=[]
        for cell in cells:
            try:contexts.append(rt_context(cell))
            except ValueError:contexts.append('')
        contexts=np.array(contexts)
        with gzip.open(args.input/'sample_sheet.txt.gz','rt') as f:
            labels=sorted({line.split('\t')[0].strip() for line in f if line.strip()})
        records={label:assignment(label) for label in labels}
        fixed={label:r for label,r in records.items() if (r['drug']=='vehicle' and r['trametinib_dose'] in (0.,1.)) or
               (r['drug']!='vehicle' and r['drug_dose']==1. and r['trametinib_dose'] in (0.,1.))}
        group_keys=sorted({key(c,r) for c in config['contexts'] for r in fixed.values()});group_map={k:i for i,k in enumerate(group_keys)}
        calls=defaultdict(lambda:defaultdict(int))
        for chunk in pd.read_csv(args.input/'hashes.txt.gz',sep='\t',header=None,chunksize=100000):
            if not (chunk[3]==1).all():raise ValueError('Sample-sheet hash family schema mismatch')
            for cell,label,umi in chunk.iloc[:,[1,2,4]].itertuples(index=False,name=None):
                if cell in lookup:calls[cell][str(label)]+=int(umi)
            check()
        mapping=np.full(len(cells),-1,dtype=np.int64);excluded=defaultdict(int)
        for cell,i in lookup.items():
            if not contexts[i]:excluded['outside_author_rt_allocation']+=1;continue
            ranked=sorted(calls.get(cell,{}).items(),key=lambda r:(-r[1],r[0]))
            if not ranked or ranked[0][1]<config['minimum_hash_umis']:
                excluded['hash_support']+=1;continue
            if len(ranked)>1 and ranked[0][1]<config['minimum_hash_ratio']*ranked[1][1]:
                excluded['ambiguous_hash']+=1;continue
            label=ranked[0][0]
            if label not in fixed:excluded['other_dose_or_unknown_assignment']+=1;continue
            mapping[i]=group_map[key(contexts[i],fixed[label])]
        del calls,lookup
        counts=np.bincount(mapping[mapping>=0],minlength=len(group_keys))
        halves=np.arange(len(cells))%2
        support=np.zeros((len(group_keys),2),int)
        np.add.at(support,(mapping[mapping>=0],halves[mapping>=0]),1)
        drugs=sorted({r['drug'] for r in fixed.values() if r['drug']!='vehicle' and r['trametinib_dose']==1.})
        write_json(args.output/'qualification.json',dict(status='chemical_star_technical_pilot_only',
            cells=len(cells),genes=len(genes),canonical_pair_identities=len(drugs),canonical_pair_ids=[[d,'Trametinib'] for d in drugs],
            contexts=config['contexts'],assignment_groups=len(group_keys),excluded_cells=dict(excluded),
            source_36_pair_gate_passed=len(drugs)>=36,genetic_tnf_contract_compatible=False,
            independent_culture_initiation_verified=False,context_assignment_source='author RT barcode allocation; no expression-derived context labels',
            group_keys=[list(k) for k in group_keys],partition_support=support.tolist()))
        total=np.zeros(len(cells));blocks=matrix_blocks(args.input/'counts.mtx.gz',check,expected_shape=(len(genes),len(cells)));shape=next(blocks)
        if shape[:2]!=(len(genes),len(cells)):raise ValueError('Count/annotation axis mismatch')
        for row,col,value in blocks:np.add.at(total,col,value)
        if np.any(total[mapping>=0]<=0):raise ValueError('Zero full-library support in retained cells')
        ref=np.array([k[3]=='vehicle' or k[5]==0 for k in group_keys])
        sums=np.zeros((len(group_keys),2,len(genes)))
        blocks=matrix_blocks(args.input/'counts.mtx.gz',check,expected_shape=(len(genes),len(cells)));next(blocks)
        for row,col,value in blocks:
            active=(mapping[col]>=0);active[active]&=ref[mapping[col[active]]]
            rc,cc=row[active],col[active]
            np.add.at(sums,(mapping[cc],halves[cc],rc),value[active]*10000/total[cc])
        mean=sums/np.maximum(support[:,:,None],1);logged=np.log2(1+mean)
        variance=np.zeros(len(genes));supported_reference=[]
        for context in config['contexts']:
            effects=defaultdict(list)
            for i,k in enumerate(group_keys):
                if k[0]!=context or not ref[i] or (k[3]=='vehicle' and k[5]==0) or min(support[i])<10:continue
                control=(context,k[1],k[2],'vehicle',0.,0.)
                ci=group_map.get(control)
                if ci is None or min(support[ci])<10:continue
                identity=('Trametinib',k[5]) if k[3]=='vehicle' else (k[3],k[4])
                effects[identity].append((logged[i]-logged[ci]).mean(0));supported_reference.append(i)
            variance+=reference_variance(effects)/len(config['contexts'])
        if len(genes)<config['minimum_genes']:raise ValueError('Original gene minimum preserved')
        columns=np.array(sorted(range(len(genes)),key=lambda i:(-variance[i],genes[i]))[:512])
        write_json(args.output/'panel.json',dict(genes=[genes[i] for i in columns],selection=config['panel_selection'],pair_outcomes_used=False))
        sums=sums[:,:,columns].copy();del mean,logged
        blocks=matrix_blocks(args.input/'counts.mtx.gz',check,expected_shape=(len(genes),len(cells)));next(blocks)
        reverse=np.full(len(genes),-1);reverse[columns]=np.arange(len(columns))
        for row,col,value in blocks:
            active=(mapping[col]>=0)&(reverse[row]>=0);active[active]&=~ref[mapping[col[active]]]
            rc,cc=row[active],col[active]
            np.add.at(sums,(mapping[cc],halves[cc],reverse[rc]),value[active]*10000/total[cc])
        logged=np.log2(1+sums/np.maximum(support[:,:,None],1));results=[]
        for context in config['contexts']:
            for drug in drugs:
                rows=[];reasons=[]
                for i,k in enumerate(group_keys):
                    if k[0]!=context or k[3]!=drug or k[4:]!=(1.,1.):continue
                    refs=[(context,k[1],k[2],'vehicle',0.,0.),(context,k[1],k[2],drug,1.,0.),(context,k[1],k[2],'vehicle',0.,1.)]
                    indices=[group_map.get(r) for r in refs]
                    if min(support[i])<10 or any(j is None or min(support[j])<10 for j in indices):
                        reasons.append(dict(group=list(k),reason='matched partition support below fixed minimum'));continue
                    ctrl,single,tram=indices
                    response=logged[i]-logged[ctrl];residual=logged[i]-logged[single]-logged[tram]+logged[ctrl]
                    rows.append((response,residual))
                if rows and not reasons:
                    response=np.mean([r[0] for r in rows],axis=0);residual=np.mean([r[1] for r in rows],axis=0)
                    results.append(dict(context=context,drug=drug,status='technical_support_qualified',strata=len(rows),
                        response_half_mse=float(np.mean((response[0]-response[1])**2)),
                        residual_half_mse=float(np.mean((residual[0]-residual[1])**2)),
                        zero_response_mse=float(np.mean(response[1]**2)),zero_residual_mse=float(np.mean(residual[1]**2))))
                else:results.append(dict(context=context,drug=drug,status='unsupported',reasons=reasons or ['no measured pair stratum']))
        write_json(args.output/'metrics.json',dict(rows=results,declared_population=len(drugs)*3,
            supported_population=sum(r['status']=='technical_support_qualified' for r in results),
            uncertainty='technical cell partition agreement only; no culture covariance or biological interval'))
        write_json(args.output/'decision.json',dict(status='completed_technical_measurement_pilot',
            all_declared_rows_accounted=True,independent_confirmation=False,frontier_claim=False,
            phi18_nomination=False,source_36_pair_gate_passed=len(drugs)>=36,
            counts_acquired=True,scientific_models_fit=0))
        print('technical pilot completed',len(drugs),'pair identities',len(results),'accounted rows',flush=True)
    except Exception as error:
        write_json(args.output/'failure.json',dict(status='failed_partial_measurement',reason=str(error),partial_assets_retained=True));raise
    finally:
        write_json(args.output/'resource_ledger.json',dict(cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall,
            peak_sampled_rss_bytes=peak,numerical_threads=1,installs=0,gpu_seconds=0,source_copied=False,
            retained_bytes=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()),scope='measurement process only; acquisition separately accounted'))


if __name__=='__main__':main()
