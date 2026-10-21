"""Hash-registered guide/arm nuisance discriminator on exposed original blocks."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import load_npz, read_json, write_json, sha256, environment
from opal.processed_gamma import library, rows
from opal.scalar_screen import block_assignments, split_report, residual_permutations, metrics, paired_gene_uncertainty
from opal.block_completion import ScalarCompletion
from tools.study_block_completion import Guard

CODE = ('tools/wave3_nuisance.py', 'tests/test_wave3_nuisance.py',
        'docs/research_reset/wave3_nuisance_protocol.md', 'src/processed_gamma.py',
        'src/scalar_screen.py', 'src/block_completion.py', 'src/models/fitness.py',
        'src/artifacts.py', 'tools/study_block_completion.py')
CONFIG = dict(wall_seconds=1800, rss_bytes=1024**3, artifact_bytes=100*1024**2,
              reserve_bytes=1024**3, alphas=[1., 10., 100., 1000.],
              rank=8, penalty=.01, steps=500, seed=20261022,
              control_seeds=[20261023, 20261024, 20261025], draws=2000, gate=.05)
CORE = ['sum', 'abs_difference', 'product', 'minimum', 'sum_squares']
EXTRA = ['arm_A_minus_B', 'arm_A_minus_B_squared', 'arm_A_squared_minus_B_squared',
         'absolute_arm_shift_sum', 'arm_shift_sum', 'arm_shift_product',
         'guide_single_variance_sum', 'minimum_squared', 'construct_count', 'orientation_imbalance']
FAMILIES = ('guide_calibration', 'guide_nodes', 'nuisance_nodes')

def now(): return datetime.now(timezone.utc).isoformat()
def signature(): return {k: sha256(ROOT/k) for k in CODE}
def check_files(files):
    for k, v in files.items():
        if sha256(k) != v: raise ValueError('Frozen input drift: '+k)

def construct_features(x, y, other_x, other_y):
    """Only exact arm-matched control singles, never double-target values."""
    delta_x, delta_y = x-other_x, y-other_y
    return np.stack((x+y, abs(x-y), x*y, np.minimum(x,y), x*x+y*y,
        x-y, (x-y)**2, x*x-y*y, abs(delta_x)+abs(delta_y),
        delta_x+delta_y, delta_x*delta_y, np.zeros_like(x), np.minimum(x,y)**2), axis=-1)

def prepare(reg, check):
    q = read_json(Path(reg['q1b'])/'registration.json')
    bank = load_npz(q['paths'][2]); prepared = load_npz(Path(reg['q1b'])/'prepared.npz')
    side = load_npz(Path(reg['sidecar'])/'partners.npz')
    guides = library(q['paths'][1]); singles = prepared['guide_single']
    if not np.array_equal(side['guide_names'], prepared['guide_names']): raise ValueError('Guide sidecar alignment')
    if not np.array_equal(side['counts'].transpose(0,1,3,2), prepared['guide_single_counts']): raise ValueError('Sidecar support')
    bits = side['partner_bits']; active = prepared['guide_single_counts'].transpose(0,1,3,2)>0
    partner_patterns = np.unique(bits[active]).tolist()
    lookup = {tuple(bank['genes'][p]): i for i,p in enumerate(bank['pairs'])}
    sums = np.zeros((2,2,len(lookup),13)); counts = np.zeros(sums.shape[:3], dtype=np.int32)
    for a,b,v in rows(q['paths'][0],guides,check):
        if a['control'] or b['control'] or a['gene']==b['gene']: continue
        i = lookup.get(tuple(sorted((a['gene'],b['gene']))))
        if i is None: continue
        x, y = singles[:,:,a['index'],0], singles[:,:,b['index'],1]
        f = construct_features(x,y,singles[:,:,a['index'],1],singles[:,:,b['index'],0])
        good = np.isfinite(v) & np.isfinite(f).all(-1)
        sums[:,:,i] += np.where(good[:,:,None],f,0); counts[:,:,i] += good
    if not np.array_equal(counts, prepared['counts']): raise ValueError('No population changes allowed')
    if not np.all(counts>=2): raise ValueError('Insufficient construct support')
    means=sums/counts[:,:,:,None]
    # Across-construct single variance, using only measured control singles.
    means[:,:,:,11] = means[:,:,:,4] - prepared['means'][:,:,:,1]**2 - prepared['means'][:,:,:,2]**2
    fraction = prepared['orientation_counts'][...,0]/counts
    features=np.concatenate((means,counts[:,:,:,None],abs(2*fraction-1)[:,:,:,None]),axis=-1).mean(1)
    if not np.isfinite(features).all(): raise ValueError('Finite features required')
    if not np.allclose(prepared['means'][...,0],bank['replicates'],rtol=0,atol=1e-12): raise ValueError('Q1b target equivalence')
    return bank, features, dict(control_partner_patterns=partner_patterns,
        distinct_active_control_partner_patterns=len(partner_patterns),
        no_constructs_lost=True, feature_names=CORE+EXTRA,
        feature_std=features.std(1).tolist(), counts_min=int(counts.min()),counts_max=int(counts.max()))

def design(features, pairs, ids, nodes, n_genes):
    x=features[ids]
    if not nodes: return x
    z=np.zeros((len(ids),n_genes)); z[np.arange(len(ids))[:,None],pairs[ids]]=1
    return np.column_stack((x,z))

def ridge_grid(features,pairs,y,train,val,nodes,n_genes,check):
    """Training-only centered/scaled statistics in bounded chunks."""
    width=features.shape[1]+(n_genes if nodes else 0)
    total=np.zeros(width); square=np.zeros(width); gram=np.zeros((width,width)); cross=np.zeros(width)
    mean=float(y[train].mean())
    for start in range(0,len(train),2048):
        check(); ids=train[start:start+2048]; x=design(features,pairs,ids,nodes,n_genes)
        total+=x.sum(0); square+=(x*x).sum(0); gram+=x.T@x; cross+=x.T@(y[ids]-mean)
    center=total/len(train); scale=np.maximum(np.sqrt(np.maximum(square/len(train)-center**2,0)),1e-8)
    gram=(gram-len(train)*np.outer(center,center))/np.outer(scale,scale); cross=cross/scale
    models=[]; losses=[]
    for alpha in CONFIG['alphas']:
        check(); coefficient=np.linalg.solve(gram+alpha*np.eye(width),cross)
        model=dict(center=center,scale=scale,coefficient=coefficient,mean=np.array(mean),nodes=np.array(nodes),alpha=np.array(alpha))
        models.append(model); prediction=predict(model,features,pairs,val,n_genes)
        losses.append(float(np.mean((prediction-y[val])**2)))
    chosen=int(np.argmin(losses))
    return models[chosen],dict(validation_mse=losses,chosen=chosen,alpha=CONFIG['alphas'][chosen])

def predict(model,features,pairs,ids,n_genes):
    result=np.empty(len(ids))
    for start in range(0,len(ids),2048):
        sub=ids[start:start+2048]; x=design(features,pairs,sub,bool(model['nodes']),n_genes)
        result[start:start+len(sub)] = float(model['mean'])+(x-model['center'])/model['scale']@model['coefficient']
    return result

def registered(out):
    r=read_json(out/'registration.json')
    if r['code']!=signature() or r['config']!=CONFIG or r['environment']!=environment(): raise ValueError('Registration drift')
    check_files(r['inputs']); return r

def freeze(a):
    if a.output.exists(): raise FileExistsError('Preserve attempts')
    q=read_json(a.q1b/'registration.json'); qr=read_json(a.q1b/'replay.json'); sr=read_json(a.sidecar/'replay_receipt.json')
    for folder,receipt,array in ((a.q1b,qr,'prepared.npz'),(a.sidecar,sr,'partners.npz')):
        if not receipt['passed'] or receipt['arrays_sha256']!=sha256(folder/array) or receipt['results_sha256']!=sha256(folder/'results.json'): raise ValueError('Passing upstream replay required')
    paths=[Path(x) for x in q['paths']]
    for folder,names in ((a.q1b,('registration.json','prepared.npz','results.json','replay.json')),
                         (a.sidecar,('registration.json','partners.npz','results.json','replay_receipt.json')),
                         (a.block,('protocol.json','assignments.npz','selection.json','results.json','replay.json','horlbeck_K562_predictions.npz','horlbeck_Jurkat_predictions.npz'))):
        paths.extend(folder/n for n in names)
    check_files(q['inputs'])
    old=read_json(a.block/'results.json')
    for c in ('K562','Jurkat'):
        if sha256(a.block/('horlbeck_'+c+'_predictions.npz'))!=old['prediction_hashes']['horlbeck_'+c]: raise ValueError('Historical prediction hash')
    a.output.mkdir(parents=True)
    write_json(a.output/'registration.json',dict(created_utc=now(),code=signature(),config=CONFIG,environment=environment(),
        inputs={str(p.resolve()):sha256(p) for p in paths},q1b=str(a.q1b.resolve()),sidecar=str(a.sidecar.resolve()),block=str(a.block.resolve()),
        eligibility='all 81466 original canonical pairs; unchanged five source block tasks; no dropped constructs',
        selection='four historical penalties per simple family, then lowest validation family MSE; ties family order; all choices saved before test scoring',
        evidence='exposed Horlbeck development only; conditional gene sensitivity; no biological interaction identification'))

def prerequisites(out):
    r=registered(out); t=read_json(out/'prerequisites.json')
    if not t['passed'] or t['registration_sha256']!=sha256(out/'registration.json'): raise ValueError('Passing postfreeze tests required')
    if (out/'fit_failure.json').exists(): raise ValueError('Preserve failed fit')
    return r

def fit(out):
    r=prerequisites(out)
    if (out/'fit_started.json').exists(): raise FileExistsError('Preserve started fit')
    write_json(out/'fit_started.json',dict(started=now())); guard=Guard(out,CONFIG)
    try:
        bank,features,audit=prepare(r,guard.check); np.savez_compressed(out/'features.npz',features=features)
        write_json(out/'covariate_audit.json',audit); assignments=load_npz(Path(r['block'])/'assignments.npz'); records=[]
        for c,context in enumerate(('K562','Jurkat')):
            task='horlbeck_'+context; pairs=bank['pairs']; y=bank['responses'][c]; n=len(bank['genes'])
            screen=dict(genes=bank['genes'],pairs=pairs,accession='GSE116198',singles=bank['singles'][c].mean(0),
                counts=bank['counts'][c].T,control_counts=bank['counts'].min(0).T,control_singles_summary=bank['singles'].mean((0,1)))
            for fold,expected in enumerate(block_assignments(screen)):
                guard.check(); split=assignments[f'{task}__{fold}']
                if not np.array_equal(split,expected) or not split_report(screen,split)['eligible']: raise ValueError('Unchanged eligible block required')
                tr,va=np.flatnonzero(split==0),np.flatnonzero(split==1); dest=out/task/f'fold-{fold}'; dest.mkdir(parents=True)
                menus={}; chosen_models={}
                for family in FAMILIES:
                    f=features[c,:,:5] if family=='guide_calibration' else features[c] if family=='nuisance_nodes' else features[c,:,:5]
                    model,menu=ridge_grid(f,pairs,y,tr,va,family!='guide_calibration',n,guard.check)
                    np.savez_compressed(dest/(family+'.npz'),**model); menus[family]=menu; chosen_models[family]=(model,f)
                winner=min(FAMILIES,key=lambda k:menus[k]['validation_mse'][menus[k]['chosen']])
                model,f=chosen_models[winner]; base_train=predict(model,f,pairs,tr,n); base_val=predict(model,f,pairs,va,n)
                residual=y[tr]-base_train; permutations=residual_permutations(screen,split)
                control_records=[]
                for name,order in [('aligned',None)]+[(f'shuffled-{p["seed"]}',p['local_permutation']) for p in permutations]:
                    guard.check(); target=residual if order is None else residual[order]
                    comp=ScalarCompletion(rank=CONFIG['rank'],penalty=CONFIG['penalty'],steps=CONFIG['steps'],seed=CONFIG['seed']).fit(
                        pairs[tr],target,n,float(y[tr].std()),(pairs[va],y[va]-base_val),check=guard.check)
                    comp.save(dest/(name+'.npz'))
                    control_records.append(dict(name=name,best_step=comp.best_step,history=comp.history))
                files={p.name:sha256(p) for p in dest.glob('*.npz')}
                records.append(dict(task=task,fold=fold,menus=menus,winner=winner,files=files,completion=control_records,
                    controls=[{k:p[k] for k in ('seed','permutable_fraction','moved_fraction','sufficient')} for p in permutations],split=split_report(screen,split)))
                write_json(out/'fit_progress.json',dict(records=records,resources=guard.receipt()))
        guard.check(); write_json(out/'selection.json',dict(completed_utc=now(),records=records,features_sha256=sha256(out/'features.npz'),resources=guard.receipt()))
    except BaseException as e:
        write_json(out/'fit_failure.json',dict(error=repr(e),resources=guard.receipt())); raise

def comparison(y,a,b,pairs):
    loss=(a-y)**2-(b-y)**2
    return dict(relative_gain=1-float(np.mean((a-y)**2))/float(np.mean((b-y)**2)),
                uncertainty=paired_gene_uncertainty(loss,pairs,draws=CONFIG['draws']))

def score(out,replay=False):
    r=prerequisites(out); selection=read_json(out/'selection.json'); prefix='replay' if replay else 'score'
    if (out/(prefix+'_started.json')).exists(): raise FileExistsError('Preserve scoring attempts')
    if len(selection['records'])!=10 or {(x['task'],x['fold']) for x in selection['records']}!={(f'horlbeck_{c}',f) for c in ('K562','Jurkat') for f in range(5)}: raise ValueError('Complete choices before scoring')
    if selection['features_sha256']!=sha256(out/'features.npz'): raise ValueError('Feature drift')
    for rec in selection['records']:
        dest=out/rec['task']/f'fold-{rec["fold"]}'; check_files({str(dest/k):v for k,v in rec['files'].items()})
        winner=min(FAMILIES,key=lambda k:rec['menus'][k]['validation_mse'][rec['menus'][k]['chosen']])
        if winner!=rec['winner']: raise ValueError('Selection drift')
    if replay and read_json(out/'score_started.json')['selection_sha256']!=sha256(out/'selection.json'): raise ValueError('Frozen scored selection drift')
    write_json(out/(prefix+'_started.json'),dict(started=now(),selection_sha256=sha256(out/'selection.json')))
    previous=read_json(out/'score_resources.json') if replay else selection['resources']; guard=Guard(out,CONFIG,consumed=previous)
    try:
        q=read_json(Path(r['q1b'])/'registration.json'); bank=load_npz(q['paths'][2]); features=load_npz(out/'features.npz')['features']
        splits=load_npz(Path(r['block'])/'assignments.npz'); results={}
        for c,context in enumerate(('K562','Jurkat')):
            task='horlbeck_'+context; pairs=bank['pairs']; n=len(bank['genes']); gathered=[]
            for rec in [x for x in selection['records'] if x['task']==task]:
                ids=np.flatnonzero(splits[f'{task}__{rec["fold"]}']==2); dest=out/task/f'fold-{rec["fold"]}'
                pred={}
                for family in FAMILIES:
                    f=features[c] if family=='nuisance_nodes' else features[c,:,:5]
                    pred[family]=predict(load_npz(dest/(family+'.npz')),f,pairs,ids,n)
                pred['selected']=pred[rec['winner']]
                for name in ['aligned']+[f'shuffled-{s}' for s in CONFIG['control_seeds']]:
                    pred[name]=pred['selected']+ScalarCompletion.load(dest/(name+'.npz')).predict(pairs[ids])
                gathered.append(dict(ids=ids,pred=pred))
            ids=np.concatenate([g['ids'] for g in gathered]); predictions={k:np.concatenate([g['pred'][k] for g in gathered]) for k in gathered[0]['pred']}
            old=load_npz(Path(r['block'])/(task+'_predictions.npz'))
            if not np.array_equal(ids,old['test_indices']) or len(np.unique(ids))!=len(ids): raise ValueError('Identical paired evaluation required')
            predictions.update(existing_completion=old['candidate'],existing_calibrated_main=old['baseline'],
                historical_minimum=bank['singles'][c].mean(0)[pairs[ids]].min(1),guide_minimum=features[c,ids,3],zero=np.zeros(len(ids)))
            y=bank['responses'][c,ids]; reps=bank['replicates'][c,:,ids]
            stats={k:metrics(y,v) for k,v in predictions.items()}
            compares={k:comparison(y,predictions['aligned'],predictions[k],pairs[ids]) for k in ['selected','existing_completion','nuisance_nodes']+[f'shuffled-{s}' for s in CONFIG['control_seeds']]}
            infection=[{k:metrics(reps[:,i],v) for k,v in predictions.items()} for i in range(2)]
            informative=all(p['sufficient'] for rec in selection['records'] if rec['task']==task for p in rec['controls'])
            gate=(compares['selected']['relative_gain']>=CONFIG['gate'] and compares['selected']['uncertainty']['gene_conditional_95'][1]<0
                and all(x['aligned']['mse']<=x['selected']['mse'] for x in infection) and informative
                and all(stats['aligned']['mse']<stats[f'shuffled-{s}']['mse'] for s in CONFIG['control_seeds'])
                and all(stats['aligned']['mse']<=stats[k]['mse'] for k in FAMILIES+('existing_calibrated_main','historical_minimum','guide_minimum','zero')))
            results[task]=dict(metrics=stats,comparisons=compares,infections=infection,completion_beyond_nuisance_gate=bool(gate),controls_informative=informative)
            arrays=dict(test_indices=ids,**predictions)
            path=out/(task+'_predictions.npz')
            if replay:
                saved=load_npz(path)
                if saved.keys()!=arrays.keys() or any(not np.array_equal(v,saved[k]) for k,v in arrays.items()): raise ValueError('Prediction replay differs')
            else: np.savez_compressed(path,**arrays)
            guard.check()
        science=dict(tasks=results,all_context_gate=all(v['completion_beyond_nuisance_gate'] for v in results.values()),
            registration_sha256=sha256(out/'registration.json'),selection_sha256=sha256(out/'selection.json'),
            label='exposed original-source blocks; conditional uncertainty; technical canonical-pair effects unidentifiable')
        if replay:
            if science!=read_json(out/'results.json'): raise ValueError('Scientific JSON replay differs')
            write_json(out/'replay.json',dict(passed=True,results_sha256=sha256(out/'results.json'),resources=guard.receipt()))
        else:
            write_json(out/'results.json',science); write_json(out/'score_resources.json',guard.receipt())
    except BaseException as e:
        write_json(out/(prefix+'_failure.json'),dict(error=repr(e),resources=guard.receipt())); raise

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('freeze','verify','fit','score','replay'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--q1b',type=Path); p.add_argument('--sidecar',type=Path); p.add_argument('--block',type=Path); a=p.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    with threadpool_limits(1):
        if a.command=='freeze': freeze(a)
        elif a.command=='verify':
            registered(a.output)
            if (a.output/'prerequisites.json').exists(): raise FileExistsError('Preserve prerequisites')
            started=now(); result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_wave3_nuisance.py'],cwd=ROOT,capture_output=True,text=True,timeout=120)
            write_json(a.output/'prerequisites.json',dict(started=started,completed=now(),passed=result.returncode==0,registration_sha256=sha256(a.output/'registration.json'),stdout=result.stdout,stderr=result.stderr))
            if result.returncode: raise RuntimeError(result.stdout+result.stderr)
        elif a.command=='fit': fit(a.output)
        else: score(a.output,a.command=='replay')

if __name__=='__main__': main()
