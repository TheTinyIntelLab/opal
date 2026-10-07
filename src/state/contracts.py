"""Task contracts and group isolation for state-response evaluation."""
from dataclasses import dataclass
import numpy as np
from opal.artifacts import fingerprint

TASKS = {
    'unseen_pair_observed': ('pair', True, True),
    'held_partner': ('partner', True, True),
    'main_context': ('biological_group', True, False),
    'pair_context_observed': ('biological_group', True, True),
    'joint_pair_context': ('pair_and_biological_group', True, True),
    'intervention_identity': ('intervention_all_response_sources', True, False),
    'dose': ('declared_exact_doses', True, False),
    'reliability': ('risk_fit_calibration_evaluation_groups', True, False),
    'higher_order': ('genuine_higher_order_conditions', True, True),
}


@dataclass(frozen=True)
class InformationContract:
    task: str
    scale: str
    assay: str
    exposure: str
    query_reference_adaptation: bool = False
    external_resources: tuple = ()

    def __post_init__(self):
        if self.task not in TASKS or self.scale not in {'log2', 'count'}:
            raise ValueError('Unknown task or expression scale')
        if not self.assay or not self.exposure:
            raise ValueError('Assay and exposure record are required')

    def table(self):
        separation, controls, singles = TASKS[self.task]
        return {'task': self.task, 'separation': separation, 'scale': self.scale,
                'assay': self.assay, 'exposure': self.exposure,
                'training': {'controls': True, 'singles': 'source_only',
                             'combinations': 'allowed_source_labels_except_registered_holdouts',
                             'external_resources': list(self.external_resources), 'pretraining': 'none'},
                'inference': {'controls': controls, 'singles': singles, 'combination_labels': False,
                              'metadata': 'pretreatment_or_treatment_identity_only'},
                'query_reference_adaptation': self.query_reference_adaptation}

    @property
    def fingerprint(self):
        return fingerprint(self.table())


def group_folds(groups):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(np.unique(groups)) < 3:
        raise ValueError('Nested context evaluation requires at least three biological groups')
    for group in np.unique(groups):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero(groups != group)
        yield str(group), train, test


def validate_isolation(train, test, groups, *, pairs=None, held_pair=False, held_partner=None,
                       pretraining_interventions=(), forbidden_interventions=()):
    train, test, groups = np.asarray(train), np.asarray(test), np.asarray(groups)
    if set(train.tolist()) & set(test.tolist()) or set(groups[train]) & set(groups[test]):
        raise ValueError('Biological unit crosses training and evaluation')
    if pairs is not None:
        pairs = np.asarray(pairs)
        if held_pair and set(map(tuple, np.sort(pairs[train], axis=1))) & set(map(tuple, np.sort(pairs[test], axis=1))):
            raise ValueError('Pair identity crosses joint pair/context holdout')
        if held_partner is not None and np.any(pairs[train] == held_partner):
            raise ValueError('Held partner has training combination labels')
    if set(pretraining_interventions) & set(forbidden_interventions):
        raise ValueError('Prohibited intervention appears in pretraining')


def error_decomposition(prediction, main, measured_main, response):
    values = [np.asarray(v, dtype=float) for v in [prediction, main, measured_main, response]]
    if any(v.shape != values[0].shape or not np.isfinite(v).all() for v in values) or values[0].ndim != 2:
        raise ValueError('Error decomposition needs aligned finite condition/gene matrices')
    prediction, main, measured_main, response = values
    me = main - measured_main
    re = (prediction - main) - (response - measured_main)
    total, main_mse, residual = (float(np.mean(v ** 2)) for v in [prediction - response, me, re])
    cross = float(2 * np.mean(me * re))
    if not np.isclose(total, main_mse + residual + cross, rtol=1e-8, atol=1e-10):
        raise ArithmeticError('Main/residual error identity failed')
    return {'total_mse': total, 'main_effect_mse': main_mse,
            'residual_mse': residual, 'error_cross_term': cross}


def registered_split(task, groups, *, pairs=None, interventions=None, doses=None,
                     held_groups=(), held_pairs=(), held_interventions=(), held_doses=()):
    """Split condition identities at the registered claim unit, before labels.

    Query identities and source identities can leave an unused guard population.
    This does not qualify a dataset or authorize external pretraining exposure.
    """
    if task not in TASKS:
        raise ValueError('Unknown benchmark task')
    groups = np.asarray(groups)
    n = len(groups)
    source = np.ones(n, bool); query = np.zeros(n, bool)
    if task in {'main_context', 'pair_context_observed', 'joint_pair_context'}:
        if not held_groups:
            raise ValueError('Biological-group holdout must be declared')
        query = np.isin(groups, held_groups); source &= ~query
    if task in {'unseen_pair_observed', 'joint_pair_context'}:
        if pairs is None or not held_pairs:
            raise ValueError('Global held pair identities must be declared')
        canonical = np.sort(np.asarray(pairs), axis=1)
        keys = set(tuple(sorted(p)) for p in held_pairs)
        held = np.array([tuple(p) in keys for p in canonical])
        source &= ~held
        query = query & held if task == 'joint_pair_context' else held
    if task in {'held_partner', 'intervention_identity'}:
        values = np.asarray(pairs if pairs is not None else interventions)
        if not held_interventions or values.shape[0] != n:
            raise ValueError('Held interventions and response identities must be declared')
        held = np.isin(values, held_interventions)
        if held.ndim == 2:
            held = held.any(1)
        query = held; source &= ~held
    if task == 'dose':
        if doses is None or not held_doses:
            raise ValueError('Exact held dose values in declared units are required')
        dose_values = np.asarray(doses)
        if dose_values.shape != (n,):
            raise ValueError('Dose transfer requires one exact dose per condition')
        query = np.isin(dose_values, held_doses); source &= ~query
    if task in {'reliability', 'higher_order'}:
        raise ValueError('Use separately qualified reliability partitions or genuine higher-order protocol')
    if not source.any() or not query.any() or np.any(source & query):
        raise ValueError('Empty or overlapping registered split')
    return np.flatnonzero(source), np.flatnonzero(query), np.flatnonzero(~(source | query))


def validate_response_sources(task, stages, *, held_groups=(), held_pairs=(), held_interventions=(),
                              held_doses=()):
    """Apply claim-specific exclusions to every response-consuming fitting stage.

    Rows contain biological_group, interventions, dose and response_kind.
    Query controls and measured singles for observed-reference contracts are
    explicitly separate from fitting/pretraining/calibration response labels.
    This validates isolation, not source qualification or sealed exposure.
    """
    if task not in TASKS:
        raise ValueError('Unknown task')
    pair_keys = {tuple(sorted(p)) for p in held_pairs}
    observed = task in {'unseen_pair_observed', 'held_partner', 'pair_context_observed', 'joint_pair_context'}
    for stage, rows in stages.items():
        if stage not in {'fitting', 'preprocessing', 'pretraining', 'calibration', 'query_references'}:
            raise ValueError('Undeclared response access stage')
        for row in rows:
            identities = tuple(row['interventions'])
            kind = row['response_kind']
            if kind not in {'control', 'single', 'pair'} or len(identities) != {'control': 0, 'single': 1, 'pair': 2}[kind]:
                raise ValueError('Response kind and identities disagree')
            if stage == 'query_references':
                if kind == 'control' or (observed and kind == 'single'):
                    continue
                raise ValueError('Query response forbidden by information contract')
            if task in {'main_context', 'pair_context_observed', 'joint_pair_context'} and row['biological_group'] in held_groups:
                raise ValueError('Held biological state enters response-consuming stage')
            if task in {'unseen_pair_observed', 'joint_pair_context'} and kind == 'pair' and tuple(sorted(identities)) in pair_keys:
                raise ValueError('Globally held pair enters response-consuming stage')
            if task in {'held_partner', 'intervention_identity'} and set(identities) & set(held_interventions):
                if task == 'intervention_identity' or kind == 'pair':
                    raise ValueError('Held intervention enters response-consuming stage')
            if task == 'dose' and row['dose'] in held_doses and kind != 'control':
                raise ValueError('Held exact dose enters response-consuming stage')
