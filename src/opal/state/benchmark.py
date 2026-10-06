"""Nested main-effect context evaluation with a fixed supported population."""
import time
from pathlib import Path
import numpy as np
from opal.artifacts import array_hash, environment, fingerprint, sha256, source_hash, write_json
from .contracts import group_folds, validate_isolation, InformationContract
from .models import MainReference, FactorizedResponse


def descriptors(bank):
    """Exact-condition numeric dose only, without invented chemistry descriptors."""
    values = np.zeros((len(bank.arrays['targets']), 1))
    mask = np.zeros_like(values, dtype=bool)
    for i, target in enumerate(bank.arrays['targets']):
        record = bank.metadata['intervention_records'][str(target)]
        try:
            dose, unit = record['dose'].split()
            if unit == 'nM':
                dose = float(dose) / 1000.
            elif unit == 'uM':
                dose = float(dose)
            else:
                continue
            if dose > 0:
                values[i, 0] = np.log10(dose); mask[i, 0] = True
        except (ValueError, TypeError):
            pass
    return values, mask


def groups_for(bank):
    return np.asarray([bank.metadata['context_records'][str(c)].get('selection_group',
                      bank.metadata['context_records'][str(c)]['biological_state']) for c in bank.arrays['contexts']])


def query_rows(available, sources, query):
    # One population for every comparison. No support-related model exclusions.
    supported = available[sources].any(0)
    ci, ti = np.nonzero(available[query] & supported[None])
    return query[ci], ti


def fit_model(bank, rows, config, *, device='cpu', seed=41, route='combined', state=True,
              shuffle_descriptors=False, shuffle_state=False, resume=None, protocol=None):
    controls = bank.arrays['controls'][rows].copy()
    effects, available = bank.arrays['singles'][rows], bank.arrays['available'][rows]
    if shuffle_state:
        controls = controls[np.random.default_rng(seed).permutation(len(controls))]
    if config.get('kind') in {'mean', 'ridge', 'kernel'}:
        return MainReference(**config).fit(controls, effects, available)
    desc, mask = descriptors(bank)
    if shuffle_descriptors:
        order = np.random.default_rng(seed).permutation(len(desc)); desc, mask = desc[order], mask[order]
    model = FactorizedResponse(**config, seed=seed, route=route, state=state, device=device).fit(
        controls, effects, available, desc, mask, resume=resume, protocol_fingerprint=protocol)
    model.source_evidence = {str(target): [
        {'context': str(bank.arrays['contexts'][c]), **bank.metadata['context_records'][str(bank.arrays['contexts'][c])]}
        for c in rows if bank.arrays['available'][c, ti]]
        for ti, target in enumerate(bank.arrays['targets'])}
    return model


def score_model(model, bank, ci, ti):
    p = model.predict(bank.arrays['controls'][ci], ti)
    return float(np.mean((p - bank.arrays['singles'][ci, ti]) ** 2)), p


def select(bank, source, groups, candidates, device):
    records = []
    for config in candidates:
        errors = []
        for held in np.unique(groups[source]):
            train = source[groups[source] != held]; query = source[groups[source] == held]
            ci, ti = query_rows(bank.arrays['available'], train, query)
            if not len(ci):
                raise ValueError('Inner fold has no common supported population')
            model = fit_model(bank, train, config, device=device)
            _, p = score_model(model, bank, ci, ti)
            errors.extend(np.mean((p - bank.arrays['singles'][ci, ti]) ** 2, 1).tolist())
        records.append({'config': config, 'mse': float(np.mean(errors)), 'conditions': len(errors)})
    return min(records, key=lambda r: r['mse'])['config'], records


def summarize(prediction, response, ci, ti, groups, context_names, targets, controls=None):
    error = np.mean((prediction - response) ** 2, 1)
    magnitude = np.mean(response ** 2, 1)
    median = np.median(magnitude)
    summary = {'mse': float(error.mean()), 'conditions': len(error),
               'by_context': {str(context_names[c]): float(error[ci == c].mean()) for c in np.unique(ci)},
               'by_intervention': {str(targets[t]): float(error[ti == t].mean()) for t in np.unique(ti)},
               'by_biological_group': {str(g): float(error[groups[ci] == g].mean()) for g in np.unique(groups[ci])},
               'weak_response_mse': float(error[magnitude <= median].mean()),
               'strong_response_mse': float(error[magnitude > median].mean()) if np.any(magnitude > median) else None,
               'response_magnitude_boundary': float(median),
               'response_strata_status': 'evaluator_only_diagnostic; does_not_define_population'}
    return summary


def run_main_benchmark(bank, output, config, *, device='cpu'):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    a = bank.arrays; groups = groups_for(bank)
    assay = next(iter(bank.metadata['context_records'].values()))['assay']
    contract = InformationContract('main_context', bank.metadata['scale'], assay,
        'development; source exposure described in source audit; not independent confirmation',
        query_reference_adaptation=bank.metadata.get('gene_panel_uses_query_controls', assay == 'MOSAIC_scRNAseq'))
    write_json(output / 'protocol.json', {'config': config, 'information_contract': contract.table(),
        'reference_hash': bank.fingerprint, 'metadata_fingerprint': fingerprint(bank.metadata), 'source_hash': source_hash(),
        'groups': groups.tolist(), 'preprocessing': 'control PCA, scales, source means and response bases fitted per training fold',
        'gene_panel': 'source-declared controls-only panel; query-control gene selection adaptation is declared for legacy EmeraldBay',
        'fixed_evaluation_population': 'observed singles with same exact intervention available in outer training',
        'unsupported_population': 'reported before modeling, identical across candidates',
        'oracle': 'measured query singles; evaluator diagnostic only'})
    write_json(output / 'environment.json', environment())
    predictions, truth, indices, target_indices, selections, exclusions = {}, [], [], [], [], []
    for held, source, query in group_folds(groups):
        validate_isolation(source, query, groups)
        ci, ti = query_rows(a['available'], source, query)
        exclusions.extend([{'context': str(a['contexts'][c]), 'intervention': str(a['targets'][t]),
                            'reason': 'no exact source intervention'} for c in query
                           for t in np.flatnonzero(a['available'][c] & ~a['available'][source].any(0))])
        simple, search = select(bank, source, groups, config['main_references'], device)
        neural, neural_search = select(bank, source, groups, config['learned_candidates'], device)
        selections.append({'held_group': held, 'simple': simple, 'simple_search': search,
                           'learned': neural, 'learned_search': neural_search})
        truth.append(a['singles'][ci, ti]); indices.extend(ci.tolist()); target_indices.extend(ti.tolist())
        routes = {'mean': {'kind': 'mean'}, 'selected_simple': simple, 'learned': neural}
        for name, candidate in routes.items():
            member_predictions = []
            for seed in config['seeds'] if name == 'learned' else [config['seeds'][0]]:
                model = fit_model(bank, source, candidate, device=device, seed=seed,
                    resume=output / 'resume' / f'{held}-{name}-{seed}.pt' if name == 'learned' else None,
                    protocol=contract.fingerprint)
                _, pred = score_model(model, bank, ci, ti); member_predictions.append(pred)
            predictions.setdefault(name, []).append(np.mean(member_predictions, 0))
        for ablation in config['ablations']:
            if ablation == 'combined':
                continue
            members = []
            for seed in config['seeds']:
                model = fit_model(bank, source, neural, device=device, seed=seed,
                    route=ablation if ablation in {'identity', 'effect', 'descriptor'} else 'combined',
                    state=ablation != 'no_state', shuffle_descriptors=ablation == 'shuffled_descriptors',
                    shuffle_state=ablation == 'shuffled_state')
                _, pred = score_model(model, bank, ci, ti); members.append(pred)
            predictions.setdefault(ablation, []).append(np.mean(members, 0))
        write_json(output / 'progress.json', {'completed_outer_groups': [r['held_group'] for r in selections]})
        print('Main-effect fold:', held, 'conditions:', len(ci), flush=True)
    y = np.concatenate(truth); ci = np.asarray(indices); ti = np.asarray(target_indices)
    predictions = {name: np.concatenate(p) for name, p in predictions.items()}
    # Freeze predictions before numerical evaluation and save condition identities.
    np.savez_compressed(output / 'predictions.npz', contexts=ci, targets=ti, **predictions)
    write_json(output / 'selection.json', selections)
    results = {name: summarize(p, y, ci, ti, groups, a['contexts'], a['targets']) for name, p in predictions.items()}
    skill = 1 - results['learned']['mse'] / results['selected_simple']['mse']
    state_skill = 1 - results['selected_simple']['mse'] / results['mean']['mse']
    # A group-resampling sensitivity is descriptive, especially for three lines.
    differences = np.mean((predictions['selected_simple'] - y) ** 2, 1) - np.mean((predictions['learned'] - y) ** 2, 1)
    group_differences = [float(differences[groups[ci] == g].mean()) for g in np.unique(groups[ci])]
    rng = np.random.default_rng(41)
    bootstrap = np.mean(rng.choice(group_differences, (2000, len(group_differences)), replace=True), 1)
    result = {'status': 'complete_development', 'models': results, 'excluded_conditions': exclusions,
        'state_reference_skill_over_mean': state_skill, 'learned_skill_over_selected_simple': skill,
        'material_threshold': config['material_improvement'], 'representation_gate': 'passed_development' if skill >= config['material_improvement'] else 'failed',
        'main_effect_gate': 'passed_development' if state_skill >= config['material_improvement'] else 'failed',
        'paired_group_bootstrap_difference_95': np.quantile(bootstrap, [.025, .975]).tolist(),
        'bootstrap_scope': 'descriptive; few shared-study groups; no measurement uncertainty or confirmation',
        'prediction_sha256': sha256(output / 'predictions.npz'), 'seconds': time.perf_counter() - started,
        'oracle_mse': 0., 'oracle_scope': 'same measured condition means; not independent measurement ceiling',
        'independent_confirmation': False, 'reliability_status': 'unavailable',
        'biological_replication': bank.metadata.get('replication', 'not established')}
    write_json(output / 'results.json', result)
    # Export a source-development-selected model, not an outer-score-selected one.
    selected, full_search = select(bank, np.arange(len(groups)), groups, config['learned_candidates'], device)
    main = fit_model(bank, np.arange(len(groups)), selected, device=device,
                     resume=output / 'resume' / 'full.pt', protocol=contract.fingerprint)
    from .inference import export_state, StatePredictor
    export_state(main, output / 'checkpoint', genes=a['genes'], targets=a['targets'], assay=assay,
                 scale=bank.metadata['scale'], intervention_records=bank.metadata['intervention_records'],
                 provenance={'protocol_sha256': sha256(output / 'protocol.json'), 'full_source_search': full_search,
                             'scope': 'development-fitted experimental state model, not a confirmed winner'})
    ci0, ti0 = np.nonzero(a['available']); ci0, ti0 = ci0[:12], ti0[:12]
    direct = main.predict(a['controls'][ci0], ti0)
    restored = StatePredictor(output / 'checkpoint')
    recovered = np.array([restored.predict(str(a['targets'][t]), baseline=a['controls'][c], genes=a['genes'],
                         assay=assay, scale=bank.metadata['scale'])['effect'] for c, t in zip(ci0, ti0)])
    difference = float(np.max(np.abs(direct - recovered)))
    if difference > 1e-5:
        raise ValueError('Portable state export does not reproduce predictions')
    write_json(output / 'restoration.json', {'maximum_absolute_error': difference, 'tolerance': 1e-5,
                                           'evaluator_labels_included': False})
    return result
