"""Controlled staged and joint combination diagnostics on source-development rows."""
import copy
from pathlib import Path
import time
import numpy as np
from opal.artifacts import sha256, source_hash, write_json
from opal.contextual.contracts import ReferenceBank
from opal.contextual.model import ContextualKernel
from .benchmark import fit_model, groups_for, select
from .contracts import error_decomposition, validate_isolation
from .models import SymmetricResidual
from .inference import export_state, StatePredictor


def run_combinations(bank, fitting, output, config, *, device='cpu'):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    a, groups = bank.arrays, groups_for(bank)
    source_contexts = np.unique(fitting['contexts'])
    groups = groups.copy()
    protocol = {'scope': 'Retrospective source-development pair/context transfer; no historic evaluator opened',
                'config': config, 'source_hash': source_hash(), 'reference_hash': bank.fingerprint, 'pairs_used': len(fitting['pairs']),
                'main_route': 'Source singles only; query controls allowed, query singles evaluator-only unless observed route',
                'residual_target': 'measured pair effect minus measured constituent effects',
                'joint_loss': 'equal gene-vector pair MSE plus single MSE with weight one',
                'pretraining': 'staged source-single supervision; no external pretrained examples',
                'selection': 'main epochs source-context nested; residual configurations preregistered bounded diagnostics',
                'pair_identity_holdout': False, 'fresh_confirmation': False}
    write_json(output / 'protocol.json', protocol)
    reference_candidates = [{'kind': 'shared_offset'}, {'kind': 'pooled_sum', 'alpha': .01, 'bandwidth': 4.},
                            {'kind': 'contextual_sum', 'alpha': 1., 'bandwidth': 1.}]
    predictions, mains, measured, outcomes, identities, contexts, summaries = {}, {}, [], [], [], [], []
    last_model, last_residual = None, None
    for held in np.unique(groups[source_contexts]):
        train_c = source_contexts[groups[source_contexts] != held]
        query_c = source_contexts[groups[source_contexts] == held]
        validate_isolation(train_c, query_c, groups)
        train = np.isin(fitting['contexts'], train_c); test = np.isin(fitting['contexts'], query_c)
        pairs, ci = fitting['pairs'][test], fitting['contexts'][test]
        # Enforce a common population, including exact source single support.
        supported = a['available'][train_c].any(0)
        use = supported[pairs].all(1)
        if not use.all():
            raise ValueError('Pair experiment contains unsupported exact interventions; register exclusions before fitting')
        chosen, _ = select(bank, train_c, groups, config['learned_candidates'], device)
        main = fit_model(bank, train_c, chosen, device=device)
        base_single = np.stack([main.predict(a['controls'][ci], pairs[:, i]) for i in [0, 1]], axis=1)
        true_single = a['singles'][ci[:, None], pairs]
        measured.append(true_single.sum(1)); outcomes.append(fitting['responses'][test])
        identities.extend(fitting['identities'][test].tolist()); contexts.extend(ci.tolist())
        predictions.setdefault('estimated_additive', []).append(base_single.sum(1)); mains.setdefault('estimated_additive', []).append(base_single.sum(1))
        predictions.setdefault('observed_additive', []).append(true_single.sum(1)); mains.setdefault('observed_additive', []).append(true_single.sum(1))
        train_pairs, train_ci = fitting['pairs'][train], fitting['contexts'][train]
        training_singles = a['singles'][train_ci[:, None], train_pairs]
        sc, st = np.nonzero(a['available'][train_c]); single_ci = train_c[sc]
        for construction in config['combination_constructions']:
            for state in [False, True]:
                name = f'staged_{construction}_' + ('state' if state else 'pooled')
                members = []
                for seed in config['seeds']:
                    residual = SymmetricResidual(main, construction=construction, state=state,
                        epochs=config['combination_epochs'], seed=seed).fit(a['controls'][train_ci], train_pairs,
                            fitting['responses'][train], training_singles)
                    members.append(residual.predict_residual(a['controls'][ci], pairs))
                prediction_residual = np.mean(members, 0)
                predictions.setdefault(name, []).append(base_single.sum(1) + prediction_residual)
                mains.setdefault(name, []).append(base_single.sum(1))
                predictions.setdefault(name + '_oracle', []).append(true_single.sum(1) + prediction_residual)
                mains.setdefault(name + '_oracle', []).append(true_single.sum(1))
        # Same labels and same pair epochs, with or without single pretraining.
        for name, pretrain, tune_encoders in [('joint', True, True), ('frozen_encoder_joint', True, False), ('no_pretraining_joint', False, True)]:
            pair_members, main_members, single_members = [], [], []
            qc, qt = np.nonzero(a['available'][query_c] & supported[None]); qc = query_c[qc]
            for seed in config['seeds']:
                candidate = copy.deepcopy(main) if pretrain else fit_model(bank, train_c, {**chosen, 'epochs': 0}, device=device, seed=seed)
                residual = SymmetricResidual(candidate, epochs=config['combination_epochs'], seed=seed).fit(
                    a['controls'][train_ci], train_pairs, fitting['responses'][train], training_singles,
                    joint=True, finetune_encoders=tune_encoders, single_controls=a['controls'][single_ci], single_targets=st,
                    single_responses=a['singles'][single_ci, st])
                main_query = sum(candidate.predict(a['controls'][ci], pairs[:, i]) for i in [0, 1])
                pair_members.append(main_query + residual.predict_residual(a['controls'][ci], pairs))
                main_members.append(main_query); single_members.append(candidate.predict(a['controls'][qc], qt))
                if name == 'joint':
                    last_model, last_residual = candidate, residual
            predictions.setdefault(name, []).append(np.mean(pair_members, 0)); mains.setdefault(name, []).append(np.mean(main_members, 0))
            # Independently score all held singles, not only pair constituents.
            before = main.predict(a['controls'][qc], qt); after = np.mean(single_members, 0)
            summaries.append({'held_group': str(held), 'model': name,
                'staged_single_mse': float(np.mean((before - a['singles'][qc, qt]) ** 2)),
                'joint_single_mse': float(np.mean((after - a['singles'][qc, qt]) ** 2))})
        # Kernel information budget uses only source singles for fitting/scaling.
        masked = {k: v.copy() for k, v in a.items()}
        masked['singles'][query_c] = 0; masked['available'][query_c] = False
        # Also mask contexts outside this declared source-development experiment.
        excluded = np.setdiff1d(np.arange(len(groups)), source_contexts)
        masked['singles'][excluded] = 0; masked['available'][excluded] = False
        restricted = ReferenceBank(masked, bank.metadata)
        # Tune against estimated-single validation, with inner query singles masked
        # from kernel preprocessing and its main model, to align information access.
        searches = []
        for kernel_config in reference_candidates:
            errors = []
            for inner_group in np.unique(groups[train_c]):
                inner_c = train_c[groups[train_c] != inner_group]
                validation_c = train_c[groups[train_c] == inner_group]
                inner_train = train & np.isin(fitting['contexts'], inner_c)
                inner_val = train & np.isin(fitting['contexts'], validation_c)
                if not inner_val.any():
                    continue
                inner_arrays = {k: v.copy() for k, v in restricted.arrays.items()}
                inner_arrays['singles'][validation_c] = 0; inner_arrays['available'][validation_c] = False
                inner_bank = ReferenceBank(inner_arrays, bank.metadata)
                inner_main = fit_model(bank, inner_c, chosen, device=device)
                vp, vc = fitting['pairs'][inner_val], fitting['contexts'][inner_val]
                inner_singles = np.stack([inner_main.predict(a['controls'][vc], vp[:, i]) for i in [0, 1]], 1)
                inner_kernel = ContextualKernel(inner_bank, kernel_config).fit(
                    fitting['pairs'][inner_train], fitting['contexts'][inner_train], fitting['responses'][inner_train])
                predicted = inner_kernel.predict_profiles(inner_singles, vc)
                errors.extend(np.mean((predicted - fitting['responses'][inner_val]) ** 2, 1).tolist())
            searches.append({'config': kernel_config, 'mse': float(np.mean(errors))})
        chosen_kernel = min(searches, key=lambda r: r['mse'])
        kernel = ContextualKernel(restricted, chosen_kernel['config']).fit(train_pairs, train_ci, fitting['responses'][train])
        for name, singles in [('selected_kernel', base_single), ('selected_kernel_oracle', true_single)]:
            predictions.setdefault(name, []).append(kernel.predict_profiles(singles, ci)); mains.setdefault(name, []).append(singles.sum(1))
        print('Combination fold:', held, 'conditions:', len(ci), flush=True)
    y, true_main = np.concatenate(outcomes), np.concatenate(measured)
    prediction = {k: np.concatenate(v) for k, v in predictions.items()}
    main_values = {k: np.concatenate(v) for k, v in mains.items()}
    np.savez_compressed(output / 'predictions.npz', identities=np.asarray(identities), contexts=np.asarray(contexts), **prediction)
    models = {k: error_decomposition(p, main_values[k], true_main, y) for k, p in prediction.items()}
    # Report each registered construction. Do not select a winner on outer errors.
    gains = {k: 1 - models[k]['total_mse'] / models['selected_kernel']['total_mse']
             for k in models if k.startswith('staged_') and not k.endswith('_oracle')}
    result = {'status': 'complete_development', 'models': models, 'skill_over_matched_selected_kernel': gains,
        'joint_main_effect_validation': summaries, 'independent_confirmation': False,
        'residual_gain_status': 'development_only_no_selected_confirmation',
        'seconds': time.perf_counter() - started, 'prediction_sha256': sha256(output / 'predictions.npz')}
    write_json(output / 'results.json', result)
    assay = next(iter(bank.metadata['context_records'].values()))['assay']
    export_state(last_model, output / 'checkpoint', genes=a['genes'], targets=a['targets'], assay=assay,
        scale=bank.metadata['scale'], intervention_records=bank.metadata['intervention_records'], residual=last_residual,
        provenance={'scope': 'last outer fold joint diagnostic; not selected best model', 'held_group': str(held),
                    'member_seed': config['seeds'][-1], 'protocol_sha256': sha256(output / 'protocol.json')})
    restored = StatePredictor(output / 'checkpoint')
    pair = pairs[0]; baseline = a['controls'][ci[0]]
    kwargs = dict(baseline=baseline, genes=a['genes'], assay=assay, scale=bank.metadata['scale'])
    result1 = restored.predict(*map(str, a['targets'][pair]), **kwargs)
    result2 = restored.predict(*map(str, a['targets'][pair[::-1]]), **kwargs)
    direct = sum(last_model.predict(baseline[None], np.array([t]))[0] for t in pair)
    direct += last_residual.predict_residual(baseline[None], pair[None])[0]
    error = float(np.max(np.abs(result1['effect'] - direct)))
    if error > 1e-5 or not np.allclose(result1['effect'], result2['effect'], atol=1e-10):
        raise ValueError('Combination export restoration or symmetry failed')
    write_json(output / 'restoration.json', {'maximum_absolute_error': error, 'pair_swap_maximum_error': float(np.max(np.abs(result1['effect'] - result2['effect'])))})
    return result
