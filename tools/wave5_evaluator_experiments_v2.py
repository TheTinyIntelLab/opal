"""Prospective synthetic evaluator probes. No biological inputs or response loaders."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = 510023


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def audit_module():
    spec = importlib.util.spec_from_file_location('wave3_strict_audit', ROOT/'tools/wave3_evaluator_v3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import psutil
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    from opal.scalar_screen import metrics, pair_assignments, block_assignments, paired_gene_uncertainty
    from opal.block_completion import ScalarCompletion, ScalarReference
    from opal.splits import random_pair_split, partner_splits

    freeze = ROOT/'results/research-wave5/freeze.json'
    slot = ROOT/'docs/research_reset/wave5/root_synthetic_slot.json'
    binding = Path(str(args.output)+'.started.json')
    if args.output.exists() or binding.exists():
        raise FileExistsError('Preserve prior experiment')
    authorization = json.loads(slot.read_text())
    if authorization['freeze_sha256'] != digest(freeze) or not authorization['authorized']:
        raise ValueError('Synthetic slot not authorized')
    code_names = ['tools/wave5_evaluator_experiments_v2.py', 'tools/wave3_evaluator_v3.py',
             'src/scalar_screen.py', 'src/block_completion.py', 'src/flexible_singles.py',
             'src/models/fitness.py', 'src/splits/__init__.py', 'src/processed_gamma.py',
             'tests/test_wave3_evaluator_v3.py']
    hashes = {name: digest(ROOT/name) for name in code_names}
    write(binding, dict(freeze_sha256=digest(freeze), code_sha256=hashes, seed=SEED,
                       command=sys.argv, cwd=str(Path.cwd()), synthetic_only=True,
                       input_spec='root_synthetic_proposal.md E1-E8, no biological files',
                       python=sys.version, numpy=np.__version__, torch=torch.__version__,
                       dependencies={name: importlib.metadata.version(name) for name in
                                     ('numpy', 'scipy', 'torch', 'pytest', 'psutil', 'threadpoolctl')},
                       threadpools=threadpool_info()))
    start, cpu = time.perf_counter(), time.process_time()
    proc = psutil.Process()
    torch.set_num_threads(1)
    result = {}

    def usage():
        info = proc.memory_info()
        return dict(wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu,
                    lifetime_peak_bytes=max(info.rss, getattr(info, 'peak_wset', 0)))

    def check():
        u = usage()
        if u['wall_seconds'] > 1200 or u['lifetime_peak_bytes'] > 2*1024**3:
            raise RuntimeError('Synthetic resource cap')

    try:
        with threadpool_limits(1), tempfile.TemporaryDirectory(prefix='opal-wave5-synthetic-') as tmp:
            tmp = Path(tmp)
            # E1: hand arithmetic, dtype wrap and unrepresentable reductions.
            metric_cases = [
                ('ordinary', np.array([1., 2., 4.]), np.array([2., 0., 5.])),
                ('unsigned', np.array([255, 0], np.uint8), np.array([0, 255], np.uint8)),
                ('signed', np.array([2**62, -2**62], np.int64), np.array([-2**62, 2**62], np.int64)),
                ('finite_extreme', np.array([1e308, -1e308]), np.array([-1e308, 1e308])),
                ('constant', np.ones(3), np.full(3, 2.)),
            ]
            result['E1_metrics'] = []
            for name, y, pred in metric_cases:
                with np.errstate(all='ignore'):
                    observed = metrics(y, pred)
                    expected = np.mean((pred.astype(float)-y.astype(float))**2)
                finite = all(v is None or np.isfinite(v) for v in observed.values())
                result['E1_metrics'].append(dict(case=name,
                    observed={k: (v if v is None or np.isfinite(v) else str(v)) for k, v in observed.items()},
                    float64_mse_oracle=float(expected) if np.isfinite(expected) else 'unrepresentable',
                    matches_oracle=bool(np.isfinite(expected) and observed['mse'] == expected),
                    returns_finite=finite))
            y, pred = np.array([1., 2., 4.]), np.array([2., 0., 5.])
            result['E1_scaling'] = {str(s): dict(mse=metrics(y*s+7, pred*s+7)['mse'],
                expected=metrics(y, pred)['mse']*s*s) for s in (.01, 1., 100.)}

            # E2/E3: exactly fixed synthetic menu, no tuning or biological fit.
            rng = np.random.default_rng(SEED)
            pairs = np.array([(a, b) for a in range(30) for b in range(a+1, 30)])
            singles = rng.normal(size=30)*.2
            factors = rng.normal(size=(30, 2))
            interaction = (factors[pairs[:, 0]]*factors[pairs[:, 1]]*np.array([1., -1.])).sum(1)
            structure = np.where(np.arange(30) % 3 == 0, 1., -.5)
            structured = structure[pairs].prod(1)
            additive = singles[pairs].sum(1)
            screen = dict(genes=np.array([f'G{i:03}' for i in range(30)]), pairs=pairs,
                          singles=singles, accession='wave5-synthetic-only')
            split = pair_assignments(screen)
            tr, va, te = [np.flatnonzero(split == k) for k in range(3)]
            write(Path(str(args.output)+'.synthetic-inputs.json'), dict(seed=SEED,
                arrays={name:dict(shape=list(value.shape),dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest()) for name,value in
                    dict(pairs=pairs,singles=singles,interaction=interaction,structured=structured,
                         additive=additive,split=split).items()},
                timing='generated from frozen seed and bound before any model fit or synthetic test score'))
            result['E2_recovery'] = {}
            for name, signal in [('additive_only', np.zeros(len(pairs))),
                                 ('signed_low_rank', interaction), ('structured', structured)]:
                check()
                y = additive+signal
                base = ScalarReference('calibration', 10).fit(singles, pairs[tr], y[tr])
                residual = y[tr]-base.predict(pairs[tr])
                models = {}
                for label, target in [('aligned', residual),
                                      ('shuffled', residual[np.random.default_rng(SEED).permutation(len(tr))])]:
                    model = ScalarCompletion(rank=2, penalty=.0001, steps=400, seed=SEED).fit(
                        pairs[tr], target, 30, max(float(y[tr].std()), 1e-6),
                        (pairs[va], y[va]-base.predict(pairs[va])))
                    model.save(tmp/f'{name}-{label}.npz')
                    models[label] = model
                # Persist selection and state digests before any synthetic test scoring.
                selection = dict(reference_alpha=10, checkpoint_steps={k:m.best_step for k,m in models.items()},
                                 hashes={p.name:digest(p) for p in tmp.glob(name+'-*.npz')})
                write(tmp/(name+'-selection.json'), selection)
                record = dict(selection=selection, reference_mse=metrics(y[te], base.predict(pairs[te]))['mse'],
                    additive_truth_mse=metrics(y[te], additive[te])['mse'], candidates={}, exact_replays={})
                for label, model in models.items():
                    prediction = model.predict(pairs[te])
                    record['candidates'][label] = metrics(y[te], base.predict(pairs[te])+prediction)['mse']
                    restored = ScalarCompletion.load(tmp/f'{name}-{label}.npz')
                    record['exact_replays'][label] = bool(np.array_equal(prediction, restored.predict(pairs[te])))
                result['E2_recovery'][name] = record
            result['E3_identifiability'] = dict(
                biological_and_technical_observables_exactly_equal=bool(np.array_equal(additive+interaction, additive+interaction)),
                construction='world1 biological interaction=signal, technical=0; world2 biological=0, technical=same signal',
                guide_arm_alias='fixed guide/arm per gene permits the same gene/pair-indexed nuisance function',
                biological_validation=False)

            # E4: group identity and source/context isolation are different contracts.
            names = screen['genes'][pairs]
            repeated = np.repeat(names, 3, axis=0)
            repeated[1::3] = repeated[1::3, ::-1]
            assignments = random_pair_split(repeated)
            role = np.full(len(repeated), -1)
            role[assignments['development']], role[assignments['test']] = 0, 1
            cold_ok = all(not set(c['held_targets']).intersection(repeated[i])
                          for c in partner_splits(repeated) for i in c['development'])
            blocks = block_assignments(screen)
            result['E4_splits'] = dict(repeated_orientation_groups_together=bool(np.all(role.reshape(-1, 3)==role.reshape(-1, 3)[:, :1])),
                cold_gene_incident_exclusion=cold_ok,
                block_test_disjoint_from_train_validation=all(not np.intersect1d(np.flatnonzero(s==2),np.flatnonzero(s<2)).size for s in blocks),
                source_keyed_hash_is_source_holdout=False,
                source_keyed_roles={source:sorted(set(pair_assignments(dict(screen, accession=source)).tolist())) for source in ('S1','S2')},
                context_holdout_oracle='entire context/source excluded from train and validation, not merely included in hash')

            # E6: independent endpoint-multiplicity reconstruction and pseudoreplication refusal.
            p = np.array([[0,1],[0,2],[1,2],[2,3]])
            delta = np.array([-2.,1.,-1.,7.])
            observed = paired_gene_uncertainty(delta,p,draws=97,seed=51)
            vals=[]
            prng=np.random.default_rng(51)
            for _ in range(97):
                draw=prng.integers(4,size=4).tolist()
                w=np.array([draw.count(a)*draw.count(b) for a,b in p])
                if sum(w): vals.append(sum(w*delta)/sum(w))
            try:
                paired_gene_uncertainty(np.repeat(delta,2),np.repeat(p,2,axis=0),draws=97,seed=51)
                refuses_duplicates=False
            except ValueError:
                refuses_duplicates=True
            global_result=paired_gene_uncertainty(np.ones(len(p)),p,draws=97,seed=51)
            result['E6_uncertainty']=dict(exact_independent_interval=bool(np.array_equal(observed['interval'],np.quantile(vals,[.025,.975]))),
                discarded_empty_draws=observed['discarded_empty_draws'], independent_empty_draws=97-len(vals),
                refuses_dependent_duplicate_pairs=refuses_duplicates,
                fixed_global_effect_interval=global_result['interval'],
                limitation='conditional gene resampling cannot estimate independent global-effect variation')

            # E7: characterize the previous opt-in strict loader, including schema shape holes.
            audit=audit_module()
            payload=dict(rank=2,penalty=.01,steps=25,seed=1,format_version=2,
                         normalized_upper=np.zeros(6,np.float32),support=np.ones(3,bool),
                         scale=1.,best_step=25,parameter_count=12)
            changes={'nan_operator':('normalized_upper',np.full(6,np.nan,np.float32)),
                     'negative_scale':('scale',-1.),'nan_scale':('scale',np.nan),
                     'scale_vector':('scale',np.array([1.])), 'penalty_vector':('penalty',np.array([.01])),
                     'bool_penalty':('penalty',True),'support_int':('support',np.ones(3,int)),
                     'best_step':('best_step',26), 'parameter_count':('parameter_count',11),
                     'rank_float':('rank',2.5), 'upper_float64':('normalized_upper',np.zeros(6,float))}
            corruptions={}
            for name,(key,value) in changes.items():
                data=dict(payload);data[key]=value
                path=tmp/(name+'.npz');np.savez(path,**data)
                try:
                    audit.strict_load_completion(path)
                    corruptions[name]=dict(accepted=True)
                except Exception as error:
                    corruptions[name]=dict(accepted=False,error_type=type(error).__name__,reason=str(error))
            path=tmp/'valid.npz';np.savez(path,**payload)
            valid=audit.strict_load_completion(path,digest(path))
            original=ScalarCompletion.load(path)
            expected=digest(path)
            with path.open('ab') as stream:stream.write(b'tamper')
            try:
                audit.strict_load_completion(path,expected);tamper=False
            except ValueError:tamper=True
            result['E7_checkpoints']=dict(mutations=corruptions,valid_exact_replay=bool(np.array_equal(valid.matrix,original.matrix)),hash_tamper_refused=tamper)
            result['E5_E8_supplement']='Bound historical synthetic test_wave3_evaluator_v3 covers missing exact controls, unequal constructs, held-label/train transforms and output scaling; see separate pytest receipt. C covers feature arithmetic extremes.'
            check()
        if hashes != {name:digest(ROOT/name) for name in code_names}:
            raise RuntimeError('Code drift during experiment')
        result.update(schema='opal-wave5-synthetic-evaluator-v1', seed=SEED, synthetic_only=True,
                      biological_fits=0, biological_scores=0, source_response_access=False,
                      resources=usage(), resource_enforcement='cooperative checks, caller timeout1200s; OS lifetime peak includes imports',
                      code_sha256=hashes, freeze_sha256=digest(freeze))
        write(args.output,result)
        print(json.dumps({'output':str(args.output),'resources':result['resources'],
                          'metric_cases':result['E1_metrics'],'checkpoint_probes':result['E7_checkpoints']}))
    except BaseException as error:
        write(Path(str(args.output)+'.failure.json'),dict(error=repr(error),resources=usage()))
        raise


if __name__ == '__main__':
    main()
