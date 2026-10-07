"""Registered source-selected panel/count sensitivity; never opens old evaluators."""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from opal.artifacts import read_json, write_json, sha256, environment
from opal.splits import partner_splits
from opal.state.contracts import error_decomposition
from tools.run_stage1 import load_data, select_kernel, kernel_prediction, interval

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiments/panel-scale-20261007.json'


def panels(data, train):
    """Source pair labels may select reporting panels; query labels cannot."""
    train = np.asarray(train, dtype=int)
    y, a, b = [np.asarray(data[k])[train] for k in ['y', 'a', 'b']]
    if not len(train) or any(not np.isfinite(v).all() for v in [y, a, b]):
        raise ValueError('Finite nonempty source responses required')
    genes = y.shape[1]; count = max(1, int(np.ceil(.1 * genes)))
    rank = lambda values, indices: indices[np.lexsort((indices, -values[indices]))[:count]]
    magnitude = (np.abs(a) + np.abs(b)).mean(0) / 2
    eligible = np.flatnonzero(magnitude <= np.median(magnitude))
    return dict(all_genes=np.arange(genes), strong=rank(np.abs(y).mean(0), np.arange(genes)),
                residual_proxy=rank(np.abs(y-a-b).mean(0), eligible))


def partner_contract(pairs, saved, seed):
    splits = [f for f in partner_splits(pairs.astype(str), seed, 3, minimum_development=3, minimum_test=1) if f['feasible']]
    if len(splits) != len(saved):
        raise ValueError('Saved partner fold count changed')
    queries, records = [], []
    for split, fold in zip(splits, saved):
        if split['development'] != fold['train'] or split['test'] != fold['query']:
            raise ValueError('Saved partner identities differ from original protocol')
        held = set(split['held_targets']); query = np.array(split['test'])
        counts = np.array([sum(str(t) in held for t in pair) for pair in pairs[query]])
        if not np.all((counts == 1) | (counts == 2)) or any(str(t) in held for t in pairs[fold['train']].ravel()):
            raise ValueError('Held incident edge leaked into source')
        queries.append(set(query.tolist())); records.append(dict(fold=fold['fold'], held_targets=sorted(held), query=query.tolist(), held_count=counts.tolist()))
    coverage = np.bincount(np.concatenate([np.array(sorted(q)) for q in queries]), minlength=len(pairs))
    return dict(folds=records, intersections=[[len(a & b) for b in queries] for a in queries],
                occurrences=int(coverage.sum()), unique_conditions=int((coverage>0).sum()),
                coverage_histogram={str(n):int((coverage==n).sum()) for n in np.unique(coverage)})


def qualify(data_root, cycle, output):
    sources = {}; identities = ['targets','pairs','genes','pair_indices','query_indices']
    for organism in ['norman','wessels']:
        paths = {s:data_root/'prepared'/f'{organism}-{s}' for s in ['count','log2']}
        metadata = {s:read_json(p/'source_manifest.json') for s,p in paths.items()}
        prep = {s:read_json(p/'preprocessing.json') for s,p in paths.items()}
        with np.load(paths['count']/'profiles.npz',allow_pickle=False) as count, np.load(paths['log2']/'profiles.npz',allow_pickle=False) as log:
            if any(not np.array_equal(count[k],log[k]) for k in identities):
                raise ValueError('Count/log development identity alignment failed')
            pairs = count['pairs'].copy(); dimensions = dict(pairs=len(pairs), genes=len(count['genes']), targets=len(count['targets']))
        if metadata['count']['observed_sha256'] != metadata['log2']['observed_sha256'] or any(prep['count'][k]!=prep['log2'][k] for k in ['normalizer','qc','gene_selection_resources','stratum_weight']):
            raise ValueError('Count/log provenance or preprocessing populations differ')
        source = dict(dimensions=dimensions, matched_identity_fields=identities, exposed=True, independent_cultures=False,
            raw_source_sha256=metadata['count']['observed_sha256'], license=metadata['count']['reuse_terms'],
            scales={s:dict(scale=prep[s]['scale'],normalizer=prep[s]['normalizer'],qc=prep[s]['qc'],
                hashes={name:sha256(p/name) for name in ['profiles.npz','preprocessing.json','source_manifest.json','profile_uncertainty.npz']}) for s,p in paths.items()}, folds={})
        for kind in ['random','partner']:
            saved=read_json(cycle/'run-v1'/f'{organism}-log2'/kind/'folds.json')
            source['folds'][kind]=[dict(fold=f['fold'],train=f['train'],query=f['query'],inner=f['inner']) for f in saved]
            if kind=='partner':source['partner_contract']=partner_contract(pairs,saved,20261007)
        sources[organism]=source
    fingerprints={str(p.relative_to(cycle)):sha256(p) for p in (cycle/'run-v1').glob('*/*/*') if p.name in ['folds.json','predictions.npz'] or p.name.endswith('-predictions.npz')}
    write_json(output/'qualification.json',dict(status='qualified_exposed_matched_count_and_saved_panels',sources=sources,
        saved_prediction_fingerprints=fingerprints,protocol_sha256=sha256(CONFIG),outcome_access=False,
        missing=['Independent cultures','Biologically validated emergent gene panels','Sealed joint outcomes']))


def checked_budget(start, cap):
    if time.perf_counter()-start > cap or psutil.Process().memory_info().rss > read_json(CONFIG)['budget']['rss_bytes']:
        raise RuntimeError('Registered budget exceeded')


def fit_count(data_root, organism, spec, config, output):
    output.mkdir(); start=time.perf_counter(); data=load_data(data_root,f'{organism}-count'); menu=config['count_menu']; summaries={}
    for kind,folds in spec['folds'].items():
        dest=output/kind;dest.mkdir(); selections=[]; all_rows=[];all_predictions={}
        for fold in folds:
            train,query=np.array(fold['train']),np.array(fold['query'])
            # Original inner rows are absolute, not outer-source-relative.
            locations={int(row):i for i,row in enumerate(train)}
            inner=[(np.array([locations[i] for i in f['train']]),np.array([locations[i] for i in f['query']])) for f in fold['inner']]
            pred=dict(additive=(data['a']+data['b'])[query]); selected={}; searches={}
            for family in menu['families']:
                checked_budget(start,config['budget']['seconds_per_count_source'])
                choice,search=select_kernel(data,train,inner,family,menu)
                selected[family]=choice; searches[family]=search
                pred[family]=pred['additive']+kernel_prediction(data,family,menu,train,query,choice['config'])
            simple=min(['mean_residual','global_correction','single_sum'],key=lambda k:selected[k]['mse']);pred['selected_simple']=pred[simple]
            np.savez_compressed(dest/f"fold-{fold['fold']}-predictions.npz",rows=query,**pred)
            selections.append(dict(**fold,selection=selected,search=searches,selected_simple_family=simple))
            write_json(dest/'folds.json',selections)
            all_rows.extend(query.tolist())
            for k,v in pred.items():all_predictions.setdefault(k,[]).append(v)
            print(organism,'count',kind,'fold',fold['fold'],flush=True)
        np.savez_compressed(dest/'predictions.npz',rows=np.array(all_rows),**{k:np.concatenate(v) for k,v in all_predictions.items()})
        summaries[kind]=dict(occurrences=len(all_rows),unique_conditions=len(set(all_rows)))
    write_json(output/'execution.json',dict(status='completed_count_fits',elapsed_seconds=time.perf_counter()-start,
        completion_rss_bytes=psutil.Process().memory_info().rss,splits=summaries))


def diagnostics(data, folds, directory, study, config, output):
    errors={}; rows=[]; counts=[];panel_records=[];technical=[];menu=config['count_menu']
    contract=partner_contract(data['pairs'],folds,config['seeds']['split']) if study.endswith('/partner') else None
    for j,fold in enumerate(folds):
        train,query=np.array(fold['train']),np.array(fold['query']); chosen_panels=panels(data,train)
        with np.load(directory/f"fold-{fold['fold']}-predictions.npz",allow_pickle=False) as saved:
            if not np.array_equal(query,saved['rows']):raise ValueError('Saved prediction row order changed')
            for panel,genes in chosen_panels.items():
                for model in saved.files:
                    if model in ['rows','baseline_only_mean_plus_offset']:continue
                    error=np.mean((saved[model][:,genes]-data['y'][query][:,genes])**2,axis=1)
                    errors.setdefault((panel,model),[]).append(error)
                panel_records.append(dict(fold=fold['fold'],panel=panel,gene_indices=genes.tolist(),source_rows=train.tolist()))
        rows.extend(query.tolist());counts.extend(contract['folds'][j]['held_count'] if contract else [0]*len(query))
        if data.get('halves') is not None:
            p=data['pairs']; halves=data['halves']; choice=fold['selection']['global_correction']['config']
            for h in [0,1]:
                singles,y=halves[f'singles_{h}'],halves[f'doubles_{h}'];main=singles[p].sum(1)
                other_main=halves[f'singles_{1-h}'][p].sum(1);other_y=halves[f'doubles_{1-h}']
                temp={**data,'a':singles[p[:,0]],'b':singles[p[:,1]],'residual':y-main}
                correction=kernel_prediction(temp,'global_correction',menu,train,query,choice)
                other_residual=(other_y-other_main)[query];left=(y-main)[query]
                for panel,genes in chosen_panels.items():
                    prediction=(main[query]+correction)[:,genes]
                    accounting=error_decomposition(prediction,main[query][:,genes],other_main[query][:,genes],other_y[query][:,genes])
                    cosine=np.sum(left[:,genes]*other_residual[:,genes],axis=1)/np.maximum(np.linalg.norm(left[:,genes],axis=1)*np.linalg.norm(other_residual[:,genes],axis=1),1e-12)
                    technical.append(dict(study=study,fold=fold['fold'],fit_half=h,panel=panel,genes=len(genes),
                        independently_estimated_residual_mse=float(np.mean((correction[:,genes]-other_residual[:,genes])**2)),
                        mean_residual_direction_cosine=float(cosine.mean()),**accounting))
    rows=np.array(rows);counts=np.array(counts); metrics=[]
    units=data['groups'][rows] if data['contextual'] else data['pairs'][rows]
    for (panel,model),parts in errors.items():
        error=np.concatenate(parts);reference=np.concatenate(errors[(panel,'selected_simple')])
        for subset in ['all_occurrences',*(['one_held_target','two_held_targets'] if contract else [])]:
            mask=np.ones(len(rows),bool) if subset=='all_occurrences' else counts==(1 if subset=='one_held_target' else 2)
            if not mask.any():
                metrics.append(dict(study=study,panel=panel,subset=subset,model=model,occurrences=0,mse=None,relative_gain=None,lower=None,upper=None,primary_gate=False));continue
            ci=interval(error[mask],reference[mask],units[mask],seed=config['seeds']['bootstrap'])
            mse=float(error[mask].mean());gain=float(1-mse/reference[mask].mean()) if reference[mask].mean()>0 else None
            lo,hi=ci['diagnostic_95_interval']
            metrics.append(dict(study=study,panel=panel,subset=subset,model=model,occurrences=int(mask.sum()),mse=mse,
                relative_gain=gain,lower=lo,upper=hi,primary_gate=bool(panel=='all_genes' and subset=='all_occurrences' and gain is not None and gain>=.05 and lo>0)))
    write_json(output/f"{study.replace('/','-')}-panels.json",panel_records)
    return metrics,technical,contract


def run(data_root,cycle,qpath,output):
    start=time.perf_counter();config=read_json(CONFIG);q=read_json(qpath)
    if q['protocol_sha256']!=sha256(CONFIG):raise ValueError('Qualified protocol changed')
    for relative,fingerprint in q['saved_prediction_fingerprints'].items():
        if sha256(cycle/relative)!=fingerprint:raise ValueError('Saved predictions changed')
    for organism,spec in q['sources'].items():
        for scale,record in spec['scales'].items():
            for name,checksum in record['hashes'].items():
                if sha256(data_root/'prepared'/f'{organism}-{scale}'/name)!=checksum:raise ValueError('Qualified bank changed')
        fit_count(data_root,organism,spec,config,output/f'{organism}-count')
    metrics=[]; technical=[]; partners={}
    tracks=[(f'{organism}-{scale}',kind) for organism in ['norman','wessels'] for scale in ['log2','count'] for kind in ['random','partner']]+[('emeraldbay-pilot-v1','context')]
    for name,kind in tracks:
        checked_budget(start,config['budget']['seconds_total'])
        directory=output/name/kind if name.endswith('-count') else cycle/'run-v1'/name/kind
        data=load_data(data_root,name);folds=read_json(directory/'folds.json');study=f'{name}/{kind}'
        m,t,contract=diagnostics(data,folds,directory,study,config,output);metrics.extend(m);technical.extend(t)
        if contract:partners[study]=contract
        print('scored panels',study,flush=True)
        write_json(output/'partial-status.json',dict(completed_studies=[r['study'] for r in metrics if r['model']=='selected_simple' and r['panel']=='all_genes' and r['subset']=='all_occurrences']))
    with (output/'comparisons.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    write_json(output/'partner-coverage.json',partners)
    write_json(output/'technical-panels.json',dict(records=technical,independent_cultures=False,
        caveat='Disjoint cells only; shared whole reference matrices, frozen QC and full-reference selection. Residual-enriched panel is a source-selected operational proxy, not validated emergent biology.'))
    write_json(output/'results.json',dict(status='completed_exposed_panel_and_count_sensitivity',comparisons=metrics,
        elapsed_seconds=time.perf_counter()-start,completion_rss_bytes=psutil.Process().memory_info().rss,
        protocol_sha256=sha256(CONFIG),qualification_sha256=sha256(qpath),code_sha256=sha256(__file__),environment=environment(),
        original_evaluator_access=False,neural_refits=0,new_neural_candidates=0,independent_confirmation=False,
        primary_count_pair_gates=[r for r in metrics if '-count/' in r['study'] and r['model'] in ['pair_product','mixed'] and r['primary_gate']]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--cycle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--qualify-only',action='store_true');p.add_argument('--qualification',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'execution.json',dict(status='started',protocol_sha256=sha256(CONFIG)))
    try:
        with threadpool_limits(limits=1):
            if args.qualify_only:qualify(args.data_root,args.cycle,args.output)
            elif args.qualification:run(args.data_root,args.cycle,args.qualification,args.output)
            else:raise ValueError('Qualification required')
        write_json(args.output/'execution.json',dict(status='completed',protocol_sha256=sha256(CONFIG),code_sha256=sha256(__file__)))
    except Exception as error:
        write_json(args.output/'execution.json',dict(status='failed',error=repr(error)));raise


if __name__=='__main__':main()
