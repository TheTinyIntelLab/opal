"""Qualified annotation transfer with global response-identity isolation."""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from opal.artifacts import read_json, write_json, sha256, environment
from opal.contextual.contracts import ReferenceBank
from opal.state.contracts import validate_response_sources

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiments/identity-reference-20261007.json'


def annotation_tokens(row):
    tokens = {'target:' + t.strip() for t in row['target'].split(',') if t.strip() and t.strip().lower() not in {'nan', 'none', 'others'}}
    for field in ['pathway_level_1', 'pathway_level_2']:
        value = row[field].strip()
        if value and value.lower() not in {'nan', 'none', 'others'}:
            tokens.add(field + ':' + value)
    return tokens


def target_components(rows):
    """Transitive shared target-token closure, independent of outcomes."""
    components = [{i} for i in range(len(rows))]
    targets = [{t for t in annotation_tokens(row) if t.startswith('target:')} for row in rows]
    changed = True
    while changed:
        changed = False
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                if set.union(*(targets[k] for k in components[i])) & set.union(*(targets[k] for k in components[j])):
                    components[i] |= components[j]; components.pop(j); changed = True; break
            if changed:
                break
    return sorted(components, key=lambda s: (-len(s), min(s)))


def source_features(rows, source, query):
    vocabulary = sorted(set.union(*(annotation_tokens(rows[i]) for i in source)))
    def encode(indices):
        out = []
        for i in indices:
            tokens = annotation_tokens(rows[i])
            out.append([float(t in tokens) for t in vocabulary] + [float(not tokens), float(bool(tokens - set(vocabulary)))])
        return np.array(out)
    x, q = encode(source), encode(query)
    scale = np.maximum(x.std(0), 1.)
    return x / scale, q / scale, vocabulary


def predict(rows, source, query, effects, config):
    """Condition axis is known line/dose, response fitting axis is source identity."""
    y = effects[source]; mean = y.mean(0)
    kind = config['kind']
    if kind == 'zero':
        return np.zeros((len(query), *mean.shape))
    if kind == 'mean':
        return np.broadcast_to(mean, (len(query), *mean.shape)).copy()
    if kind == 'pooled':
        pooled = y.mean(0).reshape(3, 4, -1).mean(0)
        return np.broadcast_to(np.tile(pooled, (3, 1)), (len(query), *mean.shape)).copy()
    x, q, _ = source_features(rows, source, query)
    flat = y.reshape(len(source), -1); ym = flat.mean(0)
    if kind == 'class_mean':
        output = []
        for i in query:
            targets = {t for t in annotation_tokens(rows[i]) if t.startswith('target:')}
            support = [j for j in source if targets & annotation_tokens(rows[j])]
            output.append((effects[support].sum(0) + config['alpha'] * mean) / (len(support) + config['alpha']))
        return np.array(output)
    if kind == 'ridge':
        xm = x.mean(0); xc = x - xm
        coef = np.linalg.solve(xc.T @ xc + config['alpha'] * np.eye(x.shape[1]), xc.T @ (flat - ym))
        return (ym + (q - xm) @ coef).reshape(len(query), *mean.shape)
    if kind != 'kernel':
        raise ValueError('Unregistered identity model')
    d = np.sum((x[:, None] - x[None]) ** 2, -1); positive = d[d > 1e-12]
    length = (np.median(positive) if len(positive) else 1.) * config['bandwidth']
    k = np.exp(-d / length); km = k.mean(0)
    coef = np.linalg.solve(k - km[None] - km[:, None] + k.mean() + config['alpha'] * np.eye(len(x)), flat - ym)
    cross = np.exp(-np.sum((q[:, None] - x[None]) ** 2, -1) / length)
    return (ym + (cross - km - cross.mean(1, keepdims=True) + km.mean()) @ coef).reshape(len(query), *mean.shape)


def qualify(raw, output):
    import anndata as ad
    source = raw / 'SrivatsanTrapnell2020_sciplex3.h5ad'; verified = read_json(raw / 'verified.json')
    if sha256(source) != verified['source_sha256'] or sha256(raw / 'publisher_processing.py') != verified['processing_sha256']:
        raise ValueError('Qualified source or processor changed')
    publication = (raw / 'publisher_processing.py').read_text()
    if "obs2 = pd.read_csv(path+folder+'/'+metadata, sep=' ')" not in publication:
        raise ValueError('Primary metadata inheritance not established')
    selected = read_json(ROOT / 'results/dose-design-20261007/qualification-v1/qualification.json')['compounds']
    f = ad.read_h5ad(source, backed='r'); o = f.obs
    rows = []
    for compound in selected:
        part = o.loc[o.perturbation.astype(str) == compound, ['target', 'pathway_level_1', 'pathway_level_2']].astype(str).drop_duplicates()
        if len(part) != 1:
            raise ValueError('Annotation inconsistent across states/doses: ' + compound)
        row = dict(compound=compound, **part.iloc[0].to_dict()); rows.append(row)
    f.file.close()
    components = target_components(rows); groups = np.zeros(len(rows), int); sizes = np.zeros(4, int)
    for component in components:
        fold = int(np.argmin(sizes)); groups[list(component)] = fold; sizes[fold] += len(component)
    if np.any(sizes == 0):
        raise ValueError('Insufficient independent target components')
    write_json(output / 'qualification.json', dict(status='qualified_publisher_annotation_development_only',
        source_sha256=verified['source_sha256'], processing_sha256=verified['processing_sha256'], license='CC-BY-4.0',
        license_verified='https://zenodo.org/api/records/13350497 metadata.license.id=cc-by-4.0',
        annotations=rows, target_components=[sorted(s) for s in components], target_fold_assignment=groups.tolist(),
        response_matrix_access=False, missing_annotations=[r['compound'] for r in rows if not annotation_tokens(r)],
        prior_exposure='All previous Opal sciPlex outcomes and published benchmarks exposed; new scores development only',
        limits=['Publisher annotations can be inconsistent with modern mechanism knowledge; no mechanistic validation',
                'No chemical structure/scaffold representation', 'Known biological lines only'], protocol_sha256=sha256(CONFIG)))


def run(prepared, qualification, output):
    config = read_json(CONFIG); q = read_json(qualification); rows = q['annotations']
    bank = ReferenceBank.load(prepared); a = bank.arrays
    if bank.metadata['source_sha256'] != q['source_sha256']:
        raise ValueError('Prepared and annotation provenance mismatch')
    groups = np.array([bank.metadata['context_records'][str(c)]['biological_state'] for c in a['contexts']])
    names = np.array([str(t).split('@')[0] for t in a['targets']]); doses = np.array([float(str(t).split('@')[1].split('nM')[0]) for t in a['targets']])
    effects = []
    for row in rows:
        profiles = []
        for state in np.unique(groups):
            for dose in [10, 100, 1000, 10000]:
                ti = np.flatnonzero((names == row['compound']) & (doses == dose))
                if len(ti) != 1:
                    raise ValueError('Exact identity/dose mismatch')
                ci = np.flatnonzero((groups == state) & a['available'][:, ti[0]])
                if not len(ci):
                    raise ValueError('Missing registered response condition')
                profiles.append(a['singles'][ci, ti[0]].mean(0))
        effects.append(profiles)
    effects = np.array(effects)
    base = [dict(kind=k) for k in ['zero', 'mean', 'pooled']] + [dict(kind='class_mean', alpha=alpha) for alpha in config['candidate_menu']['annotation_class_mean_alpha']]
    menu = [dict(kind='ridge', alpha=alpha) for alpha in config['candidate_menu']['ridge_alpha']]
    menu += [dict(kind='kernel', alpha=alpha, bandwidth=b) for alpha in config['candidate_menu']['kernel_alpha'] for b in config['candidate_menu']['kernel_bandwidth']]
    permutation = np.random.default_rng(config['seeds']['descriptor_shuffle']).permutation(len(rows))
    shuffled = [rows[i] for i in permutation]
    for track, folds in [('identity', np.arange(len(rows)) % 4), ('target_class', np.array(q['target_fold_assignment']))]:
        dest = output / track; dest.mkdir(); started = time.perf_counter(); predictions = {}; query_indices = []; searches = []
        for held in range(4):
            source, query = np.flatnonzero(folds != held), np.flatnonzero(folds == held)
            held_names = [rows[i]['compound'] for i in query]
            stage_rows = [dict(biological_group='known lines', interventions=[rows[i]['compound']], dose=dose, response_kind='single') for i in source for dose in [10, 100, 1000, 10000]]
            validate_response_sources('intervention_identity', dict(fitting=stage_rows, preprocessing=stage_rows,
                                      pretraining=[], calibration=[]), held_interventions=held_names)
            _, _, vocabulary = source_features(rows, source, query)
            search = []
            for family, candidates, features in [('simple', base, rows), ('annotation', menu, rows), ('shuffled', menu, shuffled)]:
                family_search = []
                for candidate in candidates:
                    if time.perf_counter() - started > config['budget']['seconds_per_track'] or psutil.Process().memory_info().rss > config['budget']['rss_bytes']:
                        raise RuntimeError('Registered identity budget exceeded')
                    errors = []
                    for inner in sorted(set(folds[source])):
                        fit = source[folds[source] != inner]; val = source[folds[source] == inner]
                        p = predict(features, fit, val, effects, candidate)
                        errors.extend(np.mean((p - effects[val]) ** 2, (1, 2)).tolist())
                    family_search.append(dict(config=candidate, mse=float(np.mean(errors))))
                choice = min(family_search, key=lambda r: r['mse'])
                predictions.setdefault(family, []).append(predict(features, source, query, effects, choice['config']))
                search.append(dict(family=family, choice=choice, candidates=family_search))
            query_indices.extend(query)
            searches.append(dict(held_fold=held, source=source.tolist(), query=query.tolist(), vocabulary=vocabulary,
                query_unknown_tokens={rows[i]['compound']: sorted(annotation_tokens(rows[i]) - set(vocabulary)) for i in query}, searches=search))
            write_json(dest / 'selection.json', searches)
        indices = np.array(query_indices); truth = effects[indices]
        pred = {k: np.concatenate(v) for k, v in predictions.items()}
        errors = {k: np.mean((v - truth) ** 2, (1, 2)) for k, v in pred.items()}
        comparisons = {}
        rng = np.random.default_rng(config['seeds']['compound_bootstrap'])
        for family, error in errors.items():
            differences = errors['simple'] - error
            boot = rng.choice(differences, (2000, len(differences)), replace=True).mean(1)
            lower, upper = np.quantile(boot, [.025, .975]); gain = 1 - error.mean() / errors['simple'].mean()
            comparisons[family] = dict(mse=float(error.mean()), relative_gain=float(gain), paired_lower=float(lower), paired_upper=float(upper), gate=bool(gain >= .05 and lower > 0))
        np.savez_compressed(dest / 'predictions.npz', query_compounds=np.array([rows[i]['compound'] for i in indices]), truth=truth, **pred)
        per_condition = []
        for position, i in enumerate(indices):
            for ci, state in enumerate(np.unique(groups)):
                for di, dose in enumerate([10, 100, 1000, 10000]):
                    condition = ci * 4 + di
                    per_condition.append(dict(compound=rows[i]['compound'], state=state, dose_nM=dose,
                        **{k: float(np.mean((v[position, condition] - truth[position, condition]) ** 2)) for k, v in pred.items()}))
        with (dest / 'condition-errors.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(per_condition[0])); writer.writeheader(); writer.writerows(per_condition)
        write_json(dest / 'results.json', dict(status='completed', track=track, comparisons=comparisons, profiles=288,
            compounds=len(rows), groups=folds.tolist(), elapsed_seconds=time.perf_counter() - started,
            completion_rss_bytes=psutil.Process().memory_info().rss, configurations=dict(simple=len(base), annotation=len(menu), shuffled=len(menu)),
            qualifier_sha256=sha256(qualification), protocol_sha256=sha256(CONFIG), prepared_sha256=sha256(prepared / 'references.npz'),
            exclusions=[], uncertainty=config['uncertainty'], information=config['allowed_information'], environment=environment(),
            evidence_level='exposed development', annotation_vs_shuffled_mse_reduction=float(errors['shuffled'].mean() - errors['annotation'].mean())))
        print(track, comparisons, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--qualify-raw', type=Path); p.add_argument('--prepared', type=Path); p.add_argument('--qualification', type=Path)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'execution.json', dict(status='started', protocol_sha256=sha256(CONFIG)))
    try:
        with threadpool_limits(limits=1):
            if args.qualify_raw:
                qualify(args.qualify_raw, args.output)
            elif args.prepared and args.qualification:
                run(args.prepared, args.qualification, args.output)
            else:
                raise ValueError('Choose qualification or a qualified study')
        write_json(args.output / 'execution.json', dict(status='completed', protocol_sha256=sha256(CONFIG), code_sha256=sha256(__file__)))
    except Exception as error:
        write_json(args.output / 'execution.json', dict(status='failed', exception=repr(error), protocol_sha256=sha256(CONFIG)))
        raise


if __name__ == '__main__':
    main()
