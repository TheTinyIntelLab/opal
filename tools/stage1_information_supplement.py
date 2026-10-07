"""Explicit supplementary identity control and separate baseline-only main track."""
import argparse
import time
from pathlib import Path
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from opal.artifacts import write_json, sha256, load_npz
from opal.contextual.contracts import ReferenceBank
from opal.state.benchmark import groups_for, query_rows, fit_model
from opal.state.contracts import validate_isolation
from tools.run_stage1 import (ROOT, load_data, splits, inner_splits, scale_features, feature_candidates,
                              smoother, prediction_error, interval)


def main_track(bank, output):
    output.mkdir(parents=True, exist_ok=False); started = time.perf_counter(); a = bank.arrays; groups = groups_for(bank)
    menu = [dict(kind='mean')] + [dict(kind=k, rank=3, alpha=alpha) for k in ['ridge', 'kernel'] for alpha in [.001, .01, .1, 1, 10, 100]]
    learned = [dict(width=16, rank=16, epochs=e, alpha=.01) for e in [40, 80]]
    seeds = [11, 29]; all_pred = {}; ys, cis, tis, searches, exclusions = [], [], [], [], []
    permutation = np.random.default_rng(20261008).permutation(len(groups))
    for held in np.unique(groups):
        source, query = np.flatnonzero(groups != held), np.flatnonzero(groups == held)
        validate_isolation(source, query, groups)
        ci, ti = query_rows(a['available'], source, query)
        exclusions += [dict(context=str(a['contexts'][c]), target=str(a['targets'][t]), reason='no exact source single')
                       for c in query for t in np.flatnonzero(a['available'][c] & ~a['available'][source].any(0))]
        source_supported = a['available'][source].any(0)
        if not len(ci):
            raise ValueError('Registered main population empty')
        searches_inner = []
        for candidate in menu + learned:
            errors = []
            for inner_held in np.unique(groups[source]):
                fit_rows = source[groups[source] != inner_held]; val_rows = source[groups[source] == inner_held]
                vc, vt = query_rows(a['available'], fit_rows, val_rows)
                members = []
                for seed in seeds if 'kind' not in candidate else seeds[:1]:
                    if time.perf_counter() - started > 1800:
                        raise TimeoutError('Frozen baseline-only main budget exhausted')
                    model = fit_model(bank, fit_rows, candidate, seed=seed)
                    members.append(model.predict(a['controls'][vc], vt))
                errors.extend(np.mean((np.mean(members, 0) - a['singles'][vc, vt]) ** 2, 1).tolist())
            searches_inner.append(dict(config=candidate, mse=float(np.mean(errors))))
        simple = min(searches_inner[:len(menu)], key=lambda r: r['mse'])
        neural = min(searches_inner[len(menu):], key=lambda r: r['mse'])
        predictions = {}
        for name, candidate in [('mean', menu[0]), ('selected_simple', simple['config']), ('learned', neural['config'])]:
            members = []
            for seed in seeds if name == 'learned' else seeds[:1]:
                model = fit_model(bank, source, candidate, seed=seed)
                members.append(model.predict(a['controls'][ci], ti))
            predictions[name] = np.mean(members, 0)
        # Same reference labels and selected candidate, with deliberately mismatched
        # pretreatment controls in fitting and query. No query responses consumed.
        from opal.state.models import MainReference
        shuffled = MainReference(**simple['config']).fit(a['controls'][permutation[source]], a['singles'][source], a['available'][source])
        predictions['shuffled_control'] = shuffled.predict(a['controls'][permutation[ci]], ti)
        np.savez_compressed(output / f'{held}-predictions.npz', contexts=ci, targets=ti, **predictions)
        searches.append(dict(held=str(held), source=source.tolist(), query=query.tolist(), simple=simple, learned=neural, search=searches_inner))
        write_json(output / 'selection.json', searches)
        for name, pred in predictions.items():
            all_pred.setdefault(name, []).append(pred)
        ys.append(a['singles'][ci, ti]); cis.extend(ci); tis.extend(ti)
        print('Main track', output.name, held, len(ci), flush=True)
    y = np.concatenate(ys); ci = np.asarray(cis); ti = np.asarray(tis)
    predictions = {k: np.concatenate(v) for k, v in all_pred.items()}
    np.savez_compressed(output / 'predictions.npz', contexts=ci, targets=ti, **predictions)
    errors = {k: np.mean((p - y) ** 2, 1) for k, p in predictions.items()}
    results = {}
    for name, error in errors.items():
        gain = 1 - float(error.mean()) / float(errors['selected_simple'].mean())
        uncertainty = interval(error, errors['selected_simple'], groups[ci])
        results[name] = dict(mse=float(error.mean()), relative_gain=gain, **uncertainty,
            materiality_screen=gain >= .05, development_gate=gain >= .05 and uncertainty['diagnostic_95_interval'][0] > 0,
            by_state={str(g): float(error[groups[ci] == g].mean()) for g in np.unique(groups)},
            by_intervention={str(t): float(error[ti == j].mean()) for j, t in enumerate(a['targets']) if np.any(ti == j)})
    record = dict(status='complete_exposed_baseline_only_main_development', models=results, excluded=exclusions,
                  conditions=len(ci), biological_states=len(np.unique(groups)), seconds=time.perf_counter() - started,
                  reference_hash=bank.fingerprint, prediction_sha256=sha256(output / 'predictions.npz'),
                  query_single_access='evaluator-only', chemical_descriptor_claim=False,
                  process_peak_working_set_bytes=psutil.Process().memory_info().peak_wset)
    write_json(output / 'results.json', record); return record


def identity_control(data, output, kind, config, original):
    output.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    pairs = data['pairs']; onehot = np.eye(len(data['targets']))
    identities = onehot[pairs[:, 0]] + onehot[pairs[:, 1]]
    permutation = np.random.default_rng(20261008).permutation(len(pairs)); identities = identities[permutation]
    preds, rows, records = [], [], []
    def menu(train):
        blocks = [scale_features(data['a'] + data['b'], train) / np.sqrt(data['a'].shape[1]),
                  scale_features(identities, train) / np.sqrt(identities.shape[1])]
        x = np.concatenate(blocks, 1) * np.sqrt(sum(v.shape[1] for v in blocks))
        return feature_candidates(x, train, config, standardized=True)
    for f, (train, query) in enumerate(splits(data, kind, config)):
        inner = inner_splits(data, train, kind, config, f); scores = []
        for candidate, _ in menu(train):
            errors, sizes = [], []
            for ti, vi in inner:
                t, v = train[ti], train[vi]
                k = next(g for c, g in menu(t) if c == candidate); k /= max(float(np.diag(k)[t].mean()), 1e-12)
                errors.append(prediction_error(smoother(k, t, v, candidate['alpha']), t, v, data['residual_gram'])); sizes.append(len(v))
            scores.append(dict(config=candidate, mse=float(np.average(errors, weights=sizes))))
        chosen = min(scores, key=lambda r: r['mse']); k = next(g for c, g in menu(train) if c == chosen['config'])
        k /= max(float(np.diag(k)[train].mean()), 1e-12)
        pred = data['a'][query] + data['b'][query] + smoother(k, train, query, chosen['config']['alpha']) @ data['residual'][train]
        preds.append(pred); rows.extend(query.tolist()); records.append(dict(fold=f, selection=chosen, search=scores))
    rows = np.asarray(rows); prediction = np.concatenate(preds)
    saved = load_npz(original / 'predictions.npz')
    if not np.array_equal(rows, saved['rows']):
        raise ValueError('Supplementary population differs from registered original')
    error = np.mean((prediction - data['y'][rows]) ** 2, 1)
    ref = np.mean((saved['selected_simple'] - data['y'][rows]) ** 2, 1)
    units = data['groups'][rows] if data['contextual'] else pairs[rows]
    gain = 1 - float(error.mean()) / float(ref.mean())
    np.savez_compressed(output / 'predictions.npz', rows=rows, shuffled_identity_rows=prediction, permutation=permutation)
    write_json(output / 'results.json', dict(status='complete_amended_exposed_development', mse=float(error.mean()),
        relative_gain=gain, **interval(error, ref, units), seconds=time.perf_counter() - started,
        permutation=permutation.tolist(), selection=records, original_prediction_sha256=sha256(original / 'predictions.npz'),
        prediction_sha256=sha256(output / 'predictions.npz'), comparator='original inner-selected simple reference'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--original', type=Path, required=True); p.add_argument('--main-only', action='store_true'); args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', {'registration_sha256': sha256(ROOT / 'configs/experiments/stage1-information-supplement-20261007.json')})
    import json
    config = json.loads((ROOT / 'configs/experiments/stage1-20261007.json').read_text())
    with threadpool_limits(limits=1):
        for name in ['emeraldbay-pilot-v1', 'sciplex-state-v1']:
            bank = ReferenceBank.load(args.data_root / 'prepared' / name)
            main_track(bank, args.output / name / 'main')
        if not args.main_only:
            for name in ['emeraldbay-pilot-v1', 'norman-log2', 'wessels-log2']:
                data = load_data(args.data_root, name)
                for kind in ['context'] if data['contextual'] else ['random', 'partner']:
                    identity_control(data, args.output / name / kind, kind, config, args.original / name / kind)
    write_json(args.output / 'status.json', {'status': 'complete'})


if __name__ == '__main__':
    main()
