"""Freeze, fit, score and replay the synthetic-only frontier tournament."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from opal.frontier_tournament import (CONTRACT,WORLDS,SEEDS,Predictor,make_world,budget_rows,
    fit_models,select_records,gene_sensitivity)
from opal.evaluation_precision import metrics_v3

torch.set_num_threads(1)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(obj,stream,indent=2,allow_nan=False)
        stream.write('\n')


def save_arrays(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,**arrays)


def arrays_digest(arrays):
    result = hashlib.sha256()
    for k,v in sorted(arrays.items()):
        result.update(k.encode());result.update(str(v.dtype).encode())
        result.update(str(v.shape).encode());result.update(v.tobytes())
    return result.hexdigest()


class Budget:
    def __init__(self, root):
        self.root = root
        self.protocol = read(root/'protocol.json')
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.prior_wall = self.prior_cpu = 0.
        for path in root.glob('*-resource.json'):
            record = read(path)
            self.prior_wall += record['wall_seconds'];self.prior_cpu += record['cpu_seconds']

    def check(self):
        wall = self.prior_wall+time.perf_counter()-self.started
        cpu = self.prior_cpu+time.process_time()-self.cpu_started
        if wall > self.protocol['cap_seconds'] or cpu > self.protocol['cap_seconds']:
            raise RuntimeError('Cumulative scientific wall/CPU cap')
        if shutil.disk_usage(self.root).free < 1024**3+32*1024**2:
            raise RuntimeError('Disk reserve plus working allowance')

    def finish(self, stage):
        self.check()
        size = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        if size > self.protocol['artifact_cap_bytes']: raise RuntimeError('Artifact cap')
        write(self.root/(stage+'-resource.json'),dict(wall_seconds=time.perf_counter()-self.started,
              cpu_seconds=time.process_time()-self.cpu_started,artifact_bytes=size,
              threads=torch.get_num_threads(),cumulative_prior_wall=self.prior_wall,
              cumulative_prior_cpu=self.prior_cpu))


def freeze(root, pilot):
    root.mkdir(parents=True,exist_ok=False)
    required = ['A_proposal.md','C_proposal.md','C_critique.md','A_revision.md',
                'B_proposal.md','D_proposal.md','D_critique.md','B_revision.md','experiment_matrix.md']
    for name in required:
        if not (ROOT/'docs/frontier_execution'/name).is_file():
            raise ValueError('Missing prefreeze independent document: '+name)
    if pilot:
        cells = [dict(world=w,seed=261001,noise=.1,mask='complete') for w in ('lowrank','partner_curve')]
    else:
        cells = [dict(world=w,seed=s,noise=n,mask=m) for s in SEEDS for w in WORLDS
                 for n in (0.,.1) for m in (('complete',) if n == 0 else ('complete','independent','input_quality'))]
    protocol = dict(contract=CONTRACT,synthetic_only=True,biological_access=False,pilot=pilot,
                    cells=cells,budgets=[.25,1.],seed_fit=261401,bootstrap_seed=261301,
                    cap_seconds=600 if pilot else 7200,
                    artifact_cap_bytes=(64 if pilot else 256)*1024**2,
                    primary='observed equal-construct then equal-canonical-pair MSE',
                    fit_information='selected initiation-one TRAIN/VALIDATION only; identities and measured singles',
                    support='two distinct observed TRAIN partners; fixed measured single sum fallback',
                    uncertainty='VALIDATION absolute residual 90% empirical order statistic, diagnostic only',
                    boundary='all predictions/checkpoints sealed before any TEST scoring',
                    utc=datetime.now(timezone.utc).isoformat())
    write(root/'protocol.json',protocol)
    files = sorted((ROOT/'src').rglob('*.py'))+[Path(__file__).resolve(),
        ROOT/'tests/test_frontier_tournament.py',ROOT/'docs/frontier_execution/experiment_matrix.md']
    files += [ROOT/'docs/frontier_execution'/name for name in required[:-1]]
    if (ROOT/'tests/test_frontier_tournament_independent.py').exists():
        files.append(ROOT/'tests/test_frontier_tournament_independent.py')
    write(root/'code-binding.json',dict(protocol_sha256=digest(root/'protocol.json'),
          files={p.relative_to(ROOT).as_posix():digest(p) for p in files},
          environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
                           threads=torch.get_num_threads()),
          utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(frozen=True,pilot=pilot,cells=len(cells),output=str(root))))


def verify_binding(root):
    binding = read(root/'code-binding.json')
    if digest(root/'protocol.json') != binding['protocol_sha256']: raise ValueError('Protocol drift')
    for name,expected in binding['files'].items():
        if digest(ROOT/name) != expected: raise ValueError('Code binding drift: '+name)
    if binding['environment']['numpy'] != np.__version__ or binding['environment']['torch'] != torch.__version__:
        raise ValueError('Numerical environment drift')
    return read(root/'protocol.json')


def predictions(model, pairs, singles):
    supported = model.supported(pairs)
    result = singles[pairs].sum(1)
    result[supported] = model.predict(pairs[supported])
    return result,supported


def fit(root, qualification=None):
    protocol = verify_binding(root)
    if not protocol['pilot']:
        if qualification is None: raise ValueError('Full run requires measured pilot forecast')
        forecast = read(qualification)
        if not forecast['qualified']: raise ValueError('Failed resource forecast')
    budget = Budget(root)
    directory = root/'models'
    directory.mkdir(exist_ok=False)
    manifest = dict(cells=[],files={},synthetic_only=True)
    for index,cell in enumerate(protocol['cells']):
        budget.check()
        cellroot = directory/f'cell-{index:03d}'
        cellroot.mkdir()
        data = make_world(**cell)
        # Reserved target file is physically separate from fitting/query files.
        save_arrays(cellroot/'sealed-targets.npz',observed=data['observed'][:,data['test']],
                    truth=data['truth'][:,data['test']],biological_truth=data['biological_truth'][:,data['test']])
        save_arrays(cellroot/'construct-audit.npz',constructs=data['constructs'],observed=data['observed'])
        cellrecord = dict(cell=cell,array_identity=arrays_digest(data),procedures=[])
        for fraction in protocol['budgets']:
            budget.check()
            runroot = cellroot/f'budget-{fraction:g}'
            runroot.mkdir()
            train,val,accounting = budget_rows(data,fraction)
            pairs, singles = data['pairs'],data['singles']
            records = fit_models(singles,data['features'],pairs[train],data['observed'][0,train],
                                 pairs[val],data['observed'][0,val],protocol['seed_fit'])
            selected,simple,policy = select_records(records)
            # Retain every simple configuration and each selected interaction family.
            kept = [r for r in records if r['family'].startswith('B')]
            kept += [r for family,r in selected.items() if family.startswith('I')]
            label = lambda r: r['family']+'--'+r['config']
            vp,qp = pairs[val],pairs[data['test']]
            targets_validation = data['observed'][0,val]
            result = dict(accounting=accounting,selected_simple=label(simple),selected_policy=label(policy),
                          family_selections={k:label(v) for k,v in selected.items()},
                          configurations=[{k:v for k,v in r.items() if k != 'model'} for r in records],
                          models=[],validation_rows=len(val),test_rows=len(qp))
            result['shuffled_policies'] = {
                str(i):label(min((r for family,r in selected.items() if family.endswith(f'-shuffle{i}')),
                                key=lambda r:r['validation_mse'])) for i in range(3)}
            outputs = {}
            for i,r in enumerate(kept):
                name = label(r)
                checkpoint = runroot/f'model-{i:02d}.npz'
                r['model'].save(checkpoint)
                prediction,supported = predictions(r['model'],qp,singles)
                val_prediction,_ = predictions(r['model'],vp,singles)
                errors = np.sort(np.abs(targets_validation-val_prediction))
                rank = min(len(errors)-1,int(np.ceil(.9*(len(errors)+1)))-1)
                radius = float(errors[rank])
                outputs[f'prediction_{i}'] = prediction
                result['models'].append(dict(name=name,checkpoint=checkpoint.name,
                    checkpoint_sha256=digest(checkpoint),prediction_key=f'prediction_{i}',
                    radius=radius,family=r['family']))
            degree = np.bincount(pairs[train].ravel(),minlength=len(singles))
            save_arrays(runroot/'queries.npz',pairs=qp,singles=singles,supported=supported,degree=degree)
            save_arrays(runroot/'predictions.npz',**outputs)
            write(runroot/'selection.json',result)
            cellrecord['procedures'].append(runroot.relative_to(root).as_posix())
        manifest['cells'].append(cellrecord)
        budget.check()
        print(json.dumps(dict(fitted_cell=index,world=cell['world'],seed=cell['seed'],mask=cell['mask'])),flush=True)
    for path in directory.rglob('*'):
        if path.is_file(): manifest['files'][path.relative_to(root).as_posix()] = digest(path)
    write(root/'all-predictions-sealed.json',manifest)
    budget.finish('fit')


def verify_seal(root):
    manifest = read(root/'all-predictions-sealed.json')
    protocol = verify_binding(root)
    if len(manifest['cells']) != len(protocol['cells']): raise ValueError('Partial roster')
    for name,expected in manifest['files'].items():
        if digest(root/name) != expected: raise ValueError('Sealed artifact drift: '+name)
    expected_files = set(manifest['files'])
    actual_files = {p.relative_to(root).as_posix() for p in (root/'models').rglob('*') if p.is_file()}
    if actual_files != expected_files: raise ValueError('Unsealed roster artifact')
    return protocol,manifest


def score(root):
    protocol,manifest = verify_seal(root)
    if (root/'results.json').exists(): raise FileExistsError('Preserve scoring')
    budget = Budget(root)
    write(root/'scoring-start.json',dict(seal_sha256=digest(root/'all-predictions-sealed.json'),
          utc=datetime.now(timezone.utc).isoformat()))
    results = dict(synthetic_only=True,biological_fits=0,procedures=[])
    for cellrecord in manifest['cells']:
        for relative in cellrecord['procedures']:
            budget.check()
            run = root/relative
            choice = read(run/'selection.json')
            with np.load(run/'queries.npz',allow_pickle=False) as q:
                pairs,support,degree,singles = q['pairs'],q['supported'],q['degree'],q['singles']
            with np.load(run.parent/'sealed-targets.npz',allow_pickle=False) as y:
                targets = {k:y[k].copy() for k in y.files}
            with np.load(run/'predictions.npz',allow_pickle=False) as p:
                predictions_by_name = {r['name']:p[r['prediction_key']].copy() for r in choice['models']}
            simple = predictions_by_name[choice['selected_simple']]
            models = []
            for r in choice['models']:
                pred = predictions_by_name[r['name']]
                metrics = {kind:[metrics_v3(target[i],pred) for i in (0,1)] for kind,target in targets.items()}
                diagnostics = dict(coverage=[float(np.mean(np.abs(targets['observed'][i]-pred) <= r['radius'])) for i in (0,1)],
                                   interval_width=2*r['radius'],support_fraction=float(support.mean()),
                                   supported_mse=[metrics_v3(targets['observed'][i,support],pred[support])['mse'] if support.any() else None for i in (0,1)])
                models.append(dict(name=r['name'],family=r['family'],metrics=metrics,**diagnostics))
            policy = predictions_by_name[choice['selected_policy']]
            sensitivity = [gene_sensitivity(pairs,targets['observed'][i],policy,simple) for i in (0,1)]
            squared = (targets['observed']-policy)**2
            baseline_squared = (targets['observed']-simple)**2
            gene_errors = {str(g):dict(pairs=int((pairs == g).any(1).sum()),
                policy_mse=squared[:,(pairs == g).any(1)].mean(1).tolist(),
                simple_mse=baseline_squared[:,(pairs == g).any(1)].mean(1).tolist()) for g in np.unique(pairs)}
            minimum_degree = degree[pairs].min(1)
            groups = dict(degree_0_1=minimum_degree < 2,degree_2_4=(minimum_degree >= 2)&(minimum_degree < 5),
                          degree_5_9=(minimum_degree >= 5)&(minimum_degree < 10),degree_10_plus=minimum_degree >= 10,
                          small_single=np.abs(singles[pairs]).max(1) < .1,
                          large_single=np.abs(singles[pairs]).max(1) >= .3)
            strata = {name:dict(pairs=int(mask.sum()),policy_mse=squared[:,mask].mean(1).tolist(),
                      simple_mse=baseline_squared[:,mask].mean(1).tolist()) for name,mask in groups.items() if mask.any()}
            results['procedures'].append(dict(cell=cellrecord['cell'],path=relative,
                accounting=choice['accounting'],selected_simple=choice['selected_simple'],
                selected_policy=choice['selected_policy'],family_selections=choice['family_selections'],
                shuffled_policies=choice['shuffled_policies'],models=models,
                policy_vs_simple_sensitivity=sensitivity,per_gene=gene_errors,strata=strata))
    # Alias assertions are scientific safeguards, not duplicate independent evidence.
    aliases = 0
    if not protocol['pilot']:
        for index,c in enumerate(manifest['cells']):
            if c['cell']['world'] != 'lowrank': continue
            alias = next(x for x in manifest['cells'] if x['cell'] == dict(c['cell'],world='technical_persistent'))
            if c['array_identity'] != alias['array_identity']: raise ValueError('Causal alias changed inputs')
            for one,two in zip(c['procedures'],alias['procedures'],strict=True):
                a,b = root/one,root/two
                with np.load(a/'predictions.npz') as x,np.load(b/'predictions.npz') as y:
                    if x.files != y.files or any(not np.array_equal(x[k],y[k]) for k in x.files):
                        raise ValueError('Causal alias changed predictions')
            aliases += 1
    results['exact_causal_alias_cells'] = aliases
    write(root/'results.json',results)
    budget.finish('score')
    print(json.dumps(dict(scored=True,procedures=len(results['procedures']),exact_alias_cells=aliases)))


def replay(root,installed_root=None):
    protocol,manifest = verify_seal(root)
    budget = Budget(root)
    write(root/'replay-start.json',dict(results_sha256=digest(root/'results.json')))
    module = importlib.import_module('opal.frontier_tournament')
    if installed_root and not Path(module.__file__).resolve().is_relative_to(installed_root.resolve()):
        raise ValueError('Replay imported checkout rather than installed package')
    counts = dict(checkpoints=0,metrics=0,construct_aggregations=0)
    results = read(root/'results.json')
    for cellrecord in manifest['cells']:
        for relative in cellrecord['procedures']:
            budget.check()
            run = root/relative
            choice = read(run/'selection.json')
            result = next(x for x in results['procedures'] if x['path'] == relative)
            with np.load(run/'queries.npz',allow_pickle=False) as q:
                pairs,singles = q['pairs'],q['singles']
            with np.load(run.parent/'sealed-targets.npz',allow_pickle=False) as y:
                targets = {k:y[k].copy() for k in y.files}
            with np.load(run/'predictions.npz',allow_pickle=False) as p:
                for r in choice['models']:
                    model = Predictor.load(run/r['checkpoint'],r['checkpoint_sha256'])
                    prediction,_ = predictions(model,pairs,singles)
                    if not np.array_equal(prediction,p[r['prediction_key']]): raise ValueError('Prediction replay mismatch')
                    scored = next(x for x in result['models'] if x['name'] == r['name'])
                    for kind,target in targets.items():
                        for i in (0,1):
                            # Independent sum-of-squares recomputation on binary64 arrays.
                            mse = float(np.square(target[i].astype(float)-prediction).sum(dtype=np.float64)/len(prediction))
                            if mse != scored['metrics'][kind][i]['mse']: raise ValueError('Metric replay mismatch')
                            counts['metrics'] += 1
                    counts['checkpoints'] += 1
        with np.load((root/cellrecord['procedures'][0]).parent/'construct-audit.npz') as x:
            if not np.array_equal(x['constructs'].mean(2),x['observed']): raise ValueError('Construct aggregation changed')
            counts['construct_aggregations'] += 1
    write(root/'replay.json',dict(passed=True,refits=0,counts=counts,installed=installed_root is not None,
          imported_module=module.__file__,results_sha256=digest(root/'results.json')))
    budget.finish('replay')
    print(json.dumps(dict(replayed=True,**counts)))


def forecast(root):
    if not read(root/'protocol.json')['pilot']: raise ValueError('Forecast requires pilot')
    resources = [read(root/(s+'-resource.json')) for s in ('fit','score','replay')]
    # Two pilot cells * two budgets. Use their average with a conservative 2x
    # world/serialization allowance and the required additional 25% margin.
    wall = sum(r['wall_seconds'] for r in resources)*48*2*1.25
    cpu = sum(r['cpu_seconds'] for r in resources)*48*2*1.25
    artifacts = max(r['artifact_bytes'] for r in resources)*48*1.25
    result = dict(qualified=wall <= 7200 and cpu <= 7200 and artifacts <= 256*1024**2,
        forecast_wall_seconds=wall,forecast_cpu_seconds=cpu,forecast_artifact_bytes=artifacts,
        factor=48,extra_time_allowance=2.,margin=1.25,pilot_resource_sha256={s:digest(root/(s+'-resource.json')) for s in ('fit','score','replay')})
    write(root/'forecast.json',result)
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','fit','score','replay','forecast'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--qualification',type=Path)
    p.add_argument('--installed-root',type=Path)
    a = p.parse_args();root = a.output.resolve()
    wall,cpu = time.perf_counter(),time.process_time()
    try:
        if a.stage == 'freeze': freeze(root,a.pilot)
        elif a.stage == 'fit': fit(root,a.qualification)
        elif a.stage == 'score': score(root)
        elif a.stage == 'replay': replay(root,a.installed_root)
        else: forecast(root)
    except Exception as error:
        if root.is_dir():
            failure = root/(a.stage+'-failure.json')
            if not failure.exists():
                write(failure,dict(error=type(error).__name__,message=str(error),
                      wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
                      partial_outputs_preserved=True))
        raise


if __name__ == '__main__': main()
