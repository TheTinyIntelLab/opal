"""Registered dose development and prospective precision sensitivity, never confirmation."""
import argparse
import csv
import itertools
import time
from pathlib import Path
import numpy as np
import psutil
from scipy.stats import norm
from threadpoolctl import threadpool_limits
from opal.artifacts import read_json, write_json, sha256, environment
from opal.contextual.contracts import ReferenceBank
from opal.state.contracts import registered_split, validate_response_sources

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiments/dose-design-20261007.json'


def dose_predict(x, y, query, config):
    """Fit one known-compound/context curve using source doses only."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or not len(x) or np.any(x <= 0) or query <= 0 or not np.isfinite(y).all():
        raise ValueError('Dose curve requires available finite source effects and positive nM doses')
    kind = config['kind']
    if kind == 'zero':
        return np.zeros(y.shape[1])
    if kind == 'mean':
        return y.mean(0)
    if kind == 'nearest':
        return y[np.argmin(np.abs(np.log10(x) - np.log10(query)))]
    center, scale = np.log10(x).mean(), max(np.log10(x).std(), 1e-8)
    z, q = (np.log10(x) - center) / scale, (np.log10(query) - center) / scale
    ym = y.mean(0)
    if kind == 'ridge':
        features = np.column_stack([z ** d for d in range(1, config['degree'] + 1)])
        fm = features.mean(0); features -= fm
        cross = np.array([q ** d for d in range(1, config['degree'] + 1)]) - fm
        return ym + cross @ np.linalg.solve(features.T @ features + config['alpha'] * np.eye(features.shape[1]), features.T @ (y - ym))
    if kind != 'kernel':
        raise ValueError('Unknown registered dose model')
    k = np.exp(-(z[:, None] - z[None]) ** 2 / config['bandwidth'])
    km = k.mean(0); kc = k - km[None] - km[:, None] + k.mean()
    cross = np.exp(-(z - q) ** 2 / config['bandwidth'])
    return ym + (cross - km - cross.mean() + km.mean()) @ np.linalg.solve(kc + config['alpha'] * np.eye(len(x)), y - ym)


def precision_variance(sd, states, cultures, pairs, cells):
    """Variance of the grand paired mean under explicitly crossed random effects."""
    ids = np.asarray(pairs, int)
    if states < 1 or cultures < 1 or cells < 1 or ids.ndim != 2 or ids.shape[1] != 2:
        raise ValueError('Independent units and genuine pairs required')
    counts = np.bincount(ids.ravel())
    weights = counts / (2 * len(ids))
    return (sd['state'] ** 2 / states + (sd['culture'] ** 2 + sd['shared_control'] ** 2) / (states * cultures)
            + sd['target'] ** 2 * np.sum(weights ** 2) + sd['pair'] ** 2 / len(ids)
            + sd['cell'] ** 2 / (states * cultures * len(ids) * cells))


def qualify(raw, output):
    import anndata as ad
    config = read_json(ROOT / 'configs/datasets/sciplex-state.json')
    path = raw / config['filename']; verified = read_json(raw / 'verified.json')
    digest = sha256(path)
    if digest != verified['source_sha256'] or config['license'] != 'CC-BY-4.0':
        raise ValueError('Source checksum or license changed')
    f = ad.read_h5ad(path, backed='r'); obs = f.obs
    required = {'cell_line', 'perturbation', 'dose_value', 'dose_unit', 'time', 'plate', 'well'}
    if not required <= set(obs) or set(obs.dose_unit.astype(str)) != {'nM'}:
        raise ValueError('Exact metadata not qualified')
    o = obs[(obs.cell_line.isin(['A549', 'K562', 'MCF7'])) & (obs.time == 24)]
    grid = read_json(CONFIG)['doses_nM']
    sets = [set(o.loc[(o.cell_line == line) & (o.dose_value == d), 'perturbation'].astype(str)) - {'control'}
            for line in ['A549', 'K562', 'MCF7'] for d in grid]
    shared = sorted(set.intersection(*sets))[:24]
    control_strata = set(map(tuple, o.loc[o.perturbation == 'control', ['cell_line', 'plate']].astype(str).to_numpy()))
    treatments = o[o.perturbation.isin(shared) & o.dose_value.isin(grid)]
    missing = set(map(tuple, treatments[['cell_line', 'plate']].astype(str).to_numpy())) - control_strata
    f.file.close()
    if len(shared) < 4 or missing:
        raise ValueError('Insufficient grid overlap or matched controls')
    write_json(output / 'qualification.json', dict(status='qualified_exposed_dose_development_only', source_sha256=digest,
        source_url=config['url'], license=config['license'], processing_sha256=verified['processing_sha256'],
        compounds=shared, doses_nM=grid, biological_states=['A549', 'K562', 'MCF7'], endpoint_hours=24,
        cells=len(treatments), matched_controls=True, response_matrix_access=False,
        exposure='Existing sciPlex source and benchmark already exposed; additional doses do not create independent confirmation',
        missing=['Original culture/sample mapping removed in harmonization', 'No pairs', 'No sealed independent states'],
        protocol_sha256=sha256(CONFIG), metadata_columns=list(obs.columns)))


def dose_study(prepared, output, qualification):
    started = time.perf_counter(); config = read_json(CONFIG); a = ReferenceBank.load(prepared).arrays
    names = [str(t).split('@')[0] for t in a['targets']]
    doses = np.array([float(str(t).split('@')[1].split('nM')[0]) for t in a['targets']])
    compounds = sorted(set(names)); grid = config['doses_nM']
    if compounds != read_json(qualification)['compounds'] or set(doses) != set(grid):
        raise ValueError('Prepared source differs from qualified metadata')
    menu = [dict(kind=k) for k in config['menu']['reference']]
    menu += [dict(kind='ridge', degree=d, alpha=alpha) for d in config['menu']['ridge']['degree'] for alpha in config['menu']['ridge']['alpha']]
    menu += [dict(kind='kernel', bandwidth=b, alpha=alpha) for b in config['menu']['kernel']['bandwidth'] for alpha in config['menu']['kernel']['alpha']]
    rows = []
    def predict(source_doses, query_dose, candidate):
        pred, truth, units = [], [], []
        for ci in range(len(a['contexts'])):
            for compound in compounds:
                selected = [i for i, n in enumerate(names) if n == compound and doses[i] in source_doses and a['available'][ci, i]]
                query = [i for i, n in enumerate(names) if n == compound and doses[i] == query_dose and a['available'][ci, i]]
                if not selected or not query:
                    raise ValueError('Missingness changed the registered common population')
                pred.append(dose_predict(doses[selected], a['singles'][ci, selected], query_dose, candidate))
                truth.append(a['singles'][ci, query[0]]); units.append(compound)
        return np.array(pred), np.array(truth), np.array(units)
    selections = []
    for held in grid:
        source = [d for d in grid if d != held]
        train, query, guard = registered_split('dose', np.zeros(len(doses)), doses=doses, held_doses=[held])
        stage_rows = [dict(biological_group='known state', interventions=[names[i]], dose=float(doses[i]), response_kind='single') for i in train]
        validate_response_sources('dose', {'fitting': stage_rows, 'preprocessing': stage_rows, 'pretraining': [], 'calibration': []}, held_doses=[held])
        search = []
        for candidate in menu:
            if time.perf_counter() - started > config['budget']['fitting_seconds'] or psutil.Process().memory_info().rss > config['budget']['rss_bytes']:
                raise RuntimeError('Registered fitting budget exceeded')
            errors = []
            for inner in source:
                p, y, _ = predict([d for d in source if d != inner], inner, candidate)
                errors.append(float(np.mean((p - y) ** 2)))
            search.append(dict(config=candidate, mse=float(np.mean(errors))))
        reference = min(search[:3], key=lambda r: r['mse']); smooth = min(search[3:], key=lambda r: r['mse'])
        predictions = {}; errors = {}
        for name, candidate in [('selected_reference', reference['config']), ('selected_smooth', smooth['config']), *[(k, dict(kind=k)) for k in config['menu']['reference']]]:
            p, y, units = predict(source, held, candidate); predictions[name] = p
            errors[name] = np.mean((p - y) ** 2, 1)
        # Whole compound resampling preserves all shared line/plate profiles; controls remain fixed.
        difference = errors['selected_reference'] - errors['selected_smooth']
        by_compound = np.array([difference[units == c].mean() for c in compounds])
        rng = np.random.default_rng(config['seeds']['diagnostic_bootstrap'])
        draws = rng.choice(by_compound, (2000, len(compounds)), replace=True).mean(1)
        lower, upper = np.quantile(draws, [.025, .975]); gain = 1 - errors['selected_smooth'].mean() / errors['selected_reference'].mean()
        row = dict(held_dose_nM=held, shift='interpolation' if held in [100, 1000] else 'extrapolation', profiles=len(y),
                   reference_mse=float(errors['selected_reference'].mean()), smooth_mse=float(errors['selected_smooth'].mean()),
                   relative_gain=float(gain), paired_lower=float(lower), paired_upper=float(upper), gate=bool(gain >= .05 and lower > 0))
        rows.append(row)
        np.savez_compressed(output / f'dose-{held}-predictions.npz', truth=y, compounds=units, **predictions)
        np.savez_compressed(output / f'dose-{held}-errors.npz', **errors)
        selections.append(dict(held_dose_nM=held, source_doses_nM=source, source_targets=train.tolist(), query_targets=query.tolist(),
                               guard_targets=guard.tolist(), reference=reference, smooth=smooth, search=search))
        write_json(output / 'selection.json', selections)
        print(row, flush=True)
    with (output / 'comparisons.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output / 'results.json', dict(status='completed', evidence_level='exposed development', comparisons=rows,
        elapsed_seconds=time.perf_counter() - started, rss_bytes=psutil.Process().memory_info().rss, configurations=len(menu),
        protocol_sha256=sha256(CONFIG), qualification_sha256=sha256(qualification), environment=environment(),
        uncertainty='Conditional compound bootstrap, fixed shared controls and states; not biological confirmation',
        exclusions=[], query_reference_adaptation='All declared vehicle controls used for gene panel and subtraction; no held-dose effects used in fitting',
        scope='Known compounds and states, estimated main effects at held global doses; no pair, identity or context OOD claim'))


def power_study(output):
    started = time.perf_counter(); p = read_json(CONFIG)['power_simulation']; rng = np.random.default_rng(20261007)
    all_pairs = np.array(list(itertools.combinations(range(12), 2)))
    # Seeded balanced panel selected using identities only; greedily minimize current target degree.
    order = rng.permutation(len(all_pairs)); chosen = []; degrees = np.zeros(12, int)
    for _ in range(max(p['pairs'])):
        eligible = [i for i in order if i not in chosen]
        i = min(eligible, key=lambda j: (degrees[all_pairs[j]].sum(), np.max(degrees[all_pairs[j]])))
        chosen.append(i); degrees[all_pairs[i]] += 1
    rows = []
    for scenario, sd in p['variance_scenarios'].items():
        for states, cultures, pairs, cells, gain in itertools.product(p['states'], p['cultures'], p['pairs'], p['cells_per_condition'], p['normalized_gains']):
            ids = all_pairs[chosen[:pairs]]; variance = precision_variance(sd, states, cultures, ids, cells)
            # Exact random-effect Gaussian marginal; effects remain correlated, not independent profiles.
            means = gain + rng.normal(0, np.sqrt(variance), p['draws'])
            threshold = max(.05, norm.ppf(.95) * np.sqrt(variance))
            probability = float(np.mean(means >= threshold))
            rows.append(dict(scenario=scenario, sealed_states=states, cultures_per_state=cultures, held_pairs=pairs,
                cells_per_condition=cells, assumed_relative_gain=gain, standard_error=float(np.sqrt(variance)),
                gate_probability=probability, monte_carlo_se=float(np.sqrt(probability * (1 - probability) / p['draws'])),
                broad_state_claim_eligible=states > 1))
    with (output / 'precision-sensitivity.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output / 'precision-design.json', dict(status='completed_hypothetical_sensitivity_not_empirical_power',
        protocol_sha256=sha256(CONFIG), scenarios=len(rows), draws_per_scenario=p['draws'],
        balanced_pair_panel=all_pairs[chosen].tolist(), target_degrees=degrees.tolist(), elapsed_seconds=time.perf_counter() - started,
        assumptions=p, fitted_variances=False, empirical_sample_size_recommendation=None,
        missing=['Independent-culture paired loss variance', 'State-level variance across multiple states',
                 'Target/pair/shared-control covariance', 'Mean-estimation variance versus cell count', 'Frozen successor with positive development gate'],
        limitation='Known Gaussian variance is optimistic; pilot must estimate covariance and final power must include variance-estimation uncertainty. Exactly 5% true gain is a boundary, not high-power assurance.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--qualify-raw', type=Path)
    p.add_argument('--prepared', type=Path)
    p.add_argument('--qualification', type=Path)
    p.add_argument('--power-only', action='store_true')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'execution.json', dict(status='started', protocol_sha256=sha256(CONFIG)))
    try:
        with threadpool_limits(limits=1):
            if args.qualify_raw:
                qualify(args.qualify_raw, args.output)
            elif args.power_only:
                power_study(args.output)
            elif args.prepared and args.qualification:
                dose_study(args.prepared, args.output, args.qualification)
            else:
                raise ValueError('Choose metadata qualification, precision sensitivity or qualified dose study')
        write_json(args.output / 'execution.json', dict(status='completed', protocol_sha256=sha256(CONFIG), code_sha256=sha256(__file__)))
    except Exception as error:
        write_json(args.output / 'execution.json', dict(status='failed', exception=repr(error), protocol_sha256=sha256(CONFIG)))
        raise


if __name__ == '__main__':
    main()
