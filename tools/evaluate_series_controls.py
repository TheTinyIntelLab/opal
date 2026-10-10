"""Evaluator-only metric controls and stricter metadata-support sensitivities."""
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import read_json,write_json,sha256
from run_model_series_bilinear import source_banks,metrics
from prepare_pertspectra_series import column,label,csc_columns


def run():
    output=ROOT/'results/opal_model_series_v2/evaluator'
    if shutil.disk_usage('C:/').free<1073741824: raise RuntimeError('C reserve reached')
    output.mkdir(exist_ok=True)
    scope={'role':'sealed evaluator-only; no prediction or selection input','seed':20261017,
           'primary_endpoint_unchanged':True,'technical_split':'original cell row index modulo two, shared once per control/single/pair group',
           'thresholds':[25,50],'threshold_information':'source cell-count metadata only; reuse frozen full-budget fits',
           'guide_subset':'unavailable from gene-aggregated saved banks; do not infer guide equivalence',
           'technical_halves':'disjoint technical cells from same deposited sample, never independent cultures or a biological noise ceiling'}
    write_json(output/'scope_before_evaluation.json',scope)
    support=read_json(ROOT/'results/opal_model_series_v1/pertspectra/access-v4/support_and_exclusions.json')
    counts={(str(r['sample_index']),r['context'],tuple(r['targets'])):r['cells'] for r in support}
    for source,(bank_path,_) in source_banks().items():
        with np.load(bank_path,allow_pickle=False) as z:
            bank={k:z[k] for k in ['responses','singles','pair_ids','culture_ids']}
        rows=read_json(ROOT/'results/opal_model_series_v2/primary'/source/'conditions.json')
        selected=[r for r in rows if r['budget']==1 and r['order']=='uniform' and r['acquisition_seed']==11 and r['training_seed']==20261017]
        controls=[]; thresholds=[]
        for row in selected:
            work=Path(row['checkpoints'][0]['path']).parent.parent
            contract=read_json(work/'contract.json')
            test=(bank['culture_ids']==row['stratum'])&np.isin(bank['pair_ids'],contract['held_pair_ids'])
            truth=bank['responses'][test]; rng=np.random.default_rng(scope['seed'])
            predictions={'exact':truth,'constant':np.repeat(truth.mean(0,keepdims=True),len(truth),axis=0),
                         'identity-shuffled':truth[rng.permutation(len(truth))],'no-change':np.zeros_like(truth)}
            controls.append({'stratum':row['stratum'],'outer':row['outer'],'held_rows':len(truth),
                             'outcome_access':'All held outcomes, scoring only; constant control is outcome-derived, never a fitted baseline',
                             'metrics':{k:metrics(p,truth,np.zeros_like(truth)) for k,p in predictions.items()}})
            if source=='combi-seq': continue
            context='TNFA+' if source=='TNFA-positive' else 'TNFA-'
            pairids=bank['pair_ids'][test]
            for minimum in scope['thresholds']:
                keep=[]
                for p in pairids:
                    a,b=json.loads(p); keys=[(),(a,),(b,),tuple(sorted([a,b]))]
                    keep.append(all(counts.get((row['stratum'],context,k),0)>=minimum for k in keys))
                keep=np.asarray(keep)
                record={'stratum':row['stratum'],'outer':row['outer'],'minimum_cells':minimum,'retained_rows':int(keep.sum()),
                        'original_rows':len(truth),'additional_pair_labels':0,'fitting_unchanged':True,'metrics':{}}
                for checkpoint in row['checkpoints']:
                    with np.load(Path(checkpoint['path']).parent/'freeze/predictions.npz',allow_pickle=False) as z: p=z['predictions']
                    record['metrics'][checkpoint['model']]=metrics(p[keep],truth[keep],np.zeros_like(truth[keep])) if keep.any() else {'mse':None,'reason':'No population retained'}
                thresholds.append(record)
        write_json(output/(source+'.json'),{'bank_sha256':sha256(bank_path),'controls':controls,'support_sensitivity':thresholds,'primary_scores_replaced':False})
    # Technical response halves retain the original panel, source labels and
    # guide-length exclusions. Shared-reference and separate-reference routes
    # make correlated control reuse explicit.
    source=ROOT/'data/raw/pertspectra-model-series/inhouse.h5ad'
    prepared=read_json(ROOT/'results/opal_model_series_v1/pertspectra/access-v4/prepared.json')
    with h5py.File(source,'r') as f:
        ids,treatment,targets,guides=[column(f['obs'][k]) for k in ['SampleIndex','Treatment','target_gene_name','feature_call']]
        assigned=[label(t)[0] if len(str(t).split('|'))==len(str(g).split('|')) else None for t,g in zip(targets,guides)]
        n=len(ids); matrix=f['layers/counts']; names=column(f['var'][f['var'].attrs['_index']])
        library=np.zeros(n)
        for start in range(0,len(names),128): library+=np.asarray(csc_columns(matrix,start,min(start+128,len(names)),n).sum(1)).ravel()
        keys=sorted({(str(ids[i]),str(treatment[i]),p,i%2) for i,p in enumerate(assigned) if p is not None and library[i]>0})
        lookup={k:i for i,k in enumerate(keys)}; rows=[]; cols=[]; weights=[]; cellcounts=np.zeros(len(keys),int)
        for i,p in enumerate(assigned):
            if p is None or library[i]<=0: continue
            k=lookup[(str(ids[i]),str(treatment[i]),p,i%2)]; rows.append(k); cols.append(i); weights.append(10000/library[i]); cellcounts[k]+=1
        member=sparse.csr_matrix((weights,(rows,cols)),shape=(len(keys),n))
        technical=[]
        for context,info in prepared['contexts'].items():
            panel=np.asarray([int(np.flatnonzero(names==gene)[0]) for gene in info['genes']])
            total=np.zeros((len(keys),len(panel)))
            for j,gene in enumerate(panel): total[:,j]=(member@csc_columns(matrix,int(gene),int(gene)+1,n)).toarray().ravel()
            means=total/np.maximum(cellcounts[:,None],1)
            source_name='TNFA-positive' if context=='TNFA+' else 'TNFA-negative'
            with np.load(Path(info['local'])/'canonical_bank.npz',allow_pickle=False) as z: cultures,pairs=z['culture_ids'],z['pair_ids']
            for stratum in sorted(set(cultures)):
                truth=[]; prediction=[]; shared_truth=[]; shared_prediction=[]; excluded=[]
                for p in pairs[cultures==stratum]:
                    targets=tuple(json.loads(p)); required=[(stratum,context,targets,h) for h in [0,1]]+[(stratum,context,(),h) for h in [0,1]]
                    if any(k not in lookup or cellcounts[lookup[k]]<2 for k in required): excluded.append(p); continue
                    a,b,ca,cb=[means[lookup[k]] for k in required]
                    truth.append(np.log2(1+b)-np.log2(1+cb)); prediction.append(np.log2(1+a)-np.log2(1+ca))
                    control=(ca+cb)/2
                    shared_truth.append(np.log2(1+b)-np.log2(1+control)); shared_prediction.append(np.log2(1+a)-np.log2(1+control))
                y,p=np.asarray(truth),np.asarray(prediction)
                technical.append({'source':source_name,'stratum':stratum,'pairs':len(y),'excluded':excluded,
                    'independent_reference_halves':metrics(p,y,np.zeros_like(y)),
                    'shared_reference':metrics(np.asarray(shared_prediction),np.asarray(shared_truth),np.zeros_like(y)),
                    'extra_pair_cell_outcomes':'all eligible deposited pair cells, evaluator-only','biological_noise_ceiling':False})
        write_json(output/'technical_halves.json',{'records':technical,'split':scope['technical_split'],'full_source_sha256':prepared['full_file_sha256'], 'fitting_access':False})


if __name__=='__main__': run()
