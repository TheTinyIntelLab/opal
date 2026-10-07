"""Nested global chemical-pair/whole-state development, with immutable metadata splits."""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from opal.artifacts import read_json, write_json, sha256
from opal.contextual.contracts import ReferenceBank
from opal.state.contracts import registered_split, validate_isolation, validate_response_sources
from tools.run_stage1 import load_data, select_kernel, kernel_prediction

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiments/joint-development-20261007.json'


def metadata(root):
    directory = root / 'prepared/emeraldbay-pilot-v1'; bank = ReferenceBank.load(directory)
    with np.load(directory / 'profiles.npz', allow_pickle=False) as f:
        pairs, ci = f['pairs'], f['contexts']
    bank.validate_rows(pairs, ci)
    names = np.array([str(t).split('@')[0] for t in bank.arrays['targets']])
    canonical = np.sort(names[pairs], axis=1)
    groups = np.array([bank.metadata['context_records'][str(bank.arrays['contexts'][i])]['biological_state'] for i in ci])
    return bank, canonical, groups


def joint_folds(canonical, groups):
    keys = sorted(set(map(tuple, canonical)), key=lambda k: (-int(np.all(canonical == k, axis=1).sum()), k))
    panels = [[], []]; sizes = [0, 0]
    for key in keys:
        panel = int(np.argmin(sizes)); panels[panel].append(key); sizes[panel] += int(np.all(canonical == key, axis=1).sum())
    output = []
    for state in np.unique(groups):
        for panel, held in enumerate(panels):
            if not any(groups[i] == state and tuple(canonical[i]) in held for i in range(len(groups))):
                continue
            train, query, guard = registered_split('joint_pair_context', groups, pairs=canonical, held_groups=[state], held_pairs=held)
            inner = []; exclusions = []
            for inner_state in np.unique(groups[train]):
                for inner_pair in sorted(set(map(tuple, canonical[train]))):
                    if not np.any((groups[train] == inner_state) & np.all(canonical[train] == inner_pair, axis=1)):
                        exclusions.append(dict(held_state=str(inner_state), held_pair=list(inner_pair), reason='empty metadata query'))
                        continue
                    t, v, g = registered_split('joint_pair_context', groups[train], pairs=canonical[train], held_groups=[inner_state], held_pairs=[inner_pair])
                    feasible = len(t) >= 5 and len(np.unique(groups[train[t]])) >= 3 and len(set(map(tuple, canonical[train[t]]))) >= 2 and len(v) >= 1
                    if feasible:
                        inner.append(dict(train=t.tolist(), query=v.tolist(), guard=g.tolist(), held_state=str(inner_state), held_pair=list(inner_pair)))
                    else:
                        exclusions.append(dict(held_state=str(inner_state), held_pair=list(inner_pair), reason='registered metadata minimum not met'))
            if len(inner) < 2:
                raise ValueError('Insufficient feasible inner folds; no redraw')
            validate_isolation(train, query, groups, pairs=canonical, held_pair=True)
            output.append(dict(held_state=str(state), held_panel=panel, held_pairs=[list(p) for p in held], train=train.tolist(), query=query.tolist(), guard=guard.tolist(), inner=inner, inner_exclusions=exclusions))
    queried = [i for fold in output for i in fold['query']]
    if sorted(queried) != list(range(len(groups))):
        raise ValueError('Each qualified condition must be queried exactly once')
    return output


def qualify(root, output):
    bank, pairs, groups = metadata(root); folds = joint_folds(pairs, groups)
    write_json(output / 'qualification.json', dict(status='qualified_exposed_joint_development',
        canonical_pairs=pairs.tolist(), groups=groups.tolist(), folds=folds, conditions=len(groups),
        global_pair_scope='chemical identities across all doses', outcome_access=False, sealed=False,
        license=bank.metadata['license'], source_sha256=sha256(root / 'prepared/emeraldbay-pilot-v1/profiles.npz'),
        references_sha256=sha256(root / 'prepared/emeraldbay-pilot-v1/references.npz'), protocol_sha256=sha256(CONFIG),
        query_reference_adaptation='Legacy control-selected gene panel and exact measured query singles/controls permitted',
        missing_for_confirmation=['Unexposed sealed outcomes', 'Independently initiated cultures with aligned references', 'State/culture power and variance model']))
    print('Qualified', len(folds), 'folds;', len(groups), 'query conditions total', flush=True)


def run(root, qualification, output):
    q = read_json(qualification); config = read_json(CONFIG); started = time.perf_counter()
    directory = root / 'prepared/emeraldbay-pilot-v1'
    if sha256(directory / 'profiles.npz') != q['source_sha256'] or sha256(directory / 'references.npz') != q['references_sha256']:
        raise ValueError('Qualified data changed')
    if sha256(CONFIG) != q['protocol_sha256']:
        raise ValueError('Qualified protocol changed')
    data = load_data(root, 'emeraldbay-pilot-v1'); canonical = np.array(q['canonical_pairs']); groups = np.array(q['groups'])
    menu = dict(alphas=config['menu']['alphas'], bandwidths=config['menu']['bandwidths'])
    all_rows, predictions, records = [], {}, []
    for index, fold in enumerate(q['folds']):
        train, query = np.array(fold['train']), np.array(fold['query']); inner = [(np.array(f['train']), np.array(f['query'])) for f in fold['inner']]
        def validate(indices, held_state, held_pairs):
            rows = [dict(biological_group=str(groups[i]), interventions=canonical[i].tolist(), dose='exact matched pair dose', response_kind='pair') for i in indices]
            validate_response_sources('joint_pair_context', dict(fitting=rows, preprocessing=rows, pretraining=[], calibration=[]), held_groups=[held_state], held_pairs=held_pairs)
        validate(train, fold['held_state'], fold['held_pairs'])
        for f in fold['inner']:
            validate(train[np.array(f['train'])], f['held_state'], [f['held_pair']])
        selections, searches = {}, {}; pred = dict(additive=(data['a'] + data['b'])[query])
        for family in config['menu']['families']:
            if time.perf_counter() - started > config['budget']['fitting_seconds'] or psutil.Process().memory_info().rss > config['budget']['rss_bytes']:
                raise RuntimeError('Registered joint budget exceeded')
            choice, search = select_kernel(data, train, inner, family, menu)
            selections[family], searches[family] = choice, search
            pred[family] = pred['additive'] + kernel_prediction(data, family, menu, train, query, choice['config'])
        reference = min(['mean_residual', 'global_correction', 'single_sum'], key=lambda k: selections[k]['mse'])
        pred['selected_simple'] = pred[reference]
        np.savez_compressed(output / f'fold-{index}-predictions.npz', rows=query, **pred)
        records.append(dict(**fold, selections=selections, search=searches, selected_simple_family=reference))
        write_json(output / 'selection.json', records)
        for name, value in pred.items():
            predictions.setdefault(name, []).append(value)
        all_rows.extend(query); print('joint fold', index, 'source', len(train), 'query', len(query), 'inner', len(inner), flush=True)
    rows = np.array(all_rows); predictions = {k: np.concatenate(v) for k, v in predictions.items()}
    errors = {k: np.mean((v - data['y'][rows]) ** 2, axis=1) for k, v in predictions.items()}
    states = np.unique(groups); drugs = np.unique(canonical)
    si = np.searchsorted(states, groups[rows]); di = np.searchsorted(drugs, canonical[rows])
    rng = np.random.default_rng(config['seeds']['bootstrap']); weights = []
    for _ in range(2000):
        sc = np.bincount(rng.integers(len(states), size=len(states)), minlength=len(states))
        dc = np.bincount(rng.integers(len(drugs), size=len(drugs)), minlength=len(drugs))
        w = sc[si] * dc[di[:, 0]] * dc[di[:, 1]]
        if w.sum():
            weights.append(w)
    metrics = {}
    for name, error in errors.items():
        d = errors['selected_simple'] - error; boot = [np.average(d, weights=w) for w in weights]
        lower, upper = np.quantile(boot, [.025, .975]); gain = 1 - error.mean() / errors['selected_simple'].mean()
        metrics[name] = dict(mse=float(error.mean()), relative_gain=float(gain), paired_lower=float(lower), paired_upper=float(upper), gate=bool(gain >= .05 and lower > 0))
    np.savez_compressed(output / 'predictions.npz', rows=rows, truth=data['y'][rows], **predictions)
    conditions = [dict(row=int(row), state=str(groups[row]), canonical_pair=' + '.join(canonical[row]),
                       exact_pair=' + '.join(str(t) for t in data['targets'][data['pairs'][row]]), **{name:float(error[i]) for name,error in errors.items()}) for i,row in enumerate(rows)]
    with (output / 'condition-errors.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(conditions[0]));writer.writeheader();writer.writerows(conditions)
    write_json(output / 'results.json', dict(status='completed_exposed_joint_development', models=metrics, conditions=len(rows), folds=len(records),
        canonical_pair_count=len(set(map(tuple, canonical))), biological_states=len(states), diagnostic_bootstrap_nonempty_draws=len(weights),
        uncertainty=config['uncertainty'], elapsed_seconds=time.perf_counter()-started, completion_rss_bytes=psutil.Process().memory_info().rss,
        qualification_sha256=sha256(qualification), protocol_sha256=sha256(CONFIG), information=config['information'], independent_confirmation=False,
        original_evaluator_access=False, query_reference_adaptation=q['query_reference_adaptation']))
    print(metrics, flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qualify-only',action='store_true');p.add_argument('--qualification',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'execution.json',dict(status='started',protocol_sha256=sha256(CONFIG)))
    try:
        with threadpool_limits(limits=1):
            if args.qualify_only:qualify(args.data_root,args.output)
            elif args.qualification:run(args.data_root,args.qualification,args.output)
            else:raise ValueError('Qualified split manifest required')
        write_json(args.output/'execution.json',dict(status='completed',protocol_sha256=sha256(CONFIG),code_sha256=sha256(__file__)))
    except Exception as error:
        write_json(args.output/'execution.json',dict(status='failed',exception=repr(error),protocol_sha256=sha256(CONFIG)));raise


if __name__=='__main__':main()
