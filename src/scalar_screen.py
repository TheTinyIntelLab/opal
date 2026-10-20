"""Shared, outcome-blind split/control contracts for exposed scalar screens.

This module fits no model. It retains two infection means as dependent assay
measurements and never treats guide counts as biological replication.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from opal.artifacts import load_npz, sha256


TASKS = ('horlbeck_K562', 'horlbeck_Jurkat', 'environment1', 'environment2',
         'parp2022', 'parp2023')
CONTROL_SEEDS = (20261023, 20261024, 20261025)
ORIGINAL_HASH = '7a0a6a4c7f79e79983071deb59b18d52493f4333133dfa89c100d6720b12cb94'
ORIGINAL_SOURCE_HASH = 'ba9772f3b0e929bcf5b23bebf17d4caab86c7f2ccd50dbd62fbc47e40c2aac21'
EXTERNAL_PREPARATION_HASH = '2ce58689cec6a7224b096eb97b12fdde451c74eefb9fd71857ff64ce82536c03'
EXTERNAL_HASHES = {
    'environment1': '90c65f4e4e17f2a6fa04267a7d08d80cf18e7953b833323d897d2d588ac74f47',
    'environment2': '39b0844bc8ae040d4465f3700c066f2a0def7e1079f6303b9cbb63560d6292f0',
    'parp2022': '0b3cf899fcf7b138482c6b1093fa0c72b0962ed006102b075f0135fef7158b81',
    'parp2023': '1392657b02f7421b2b89f3a3c153bb5bfed17cc4569e470870c158bdff5c6f62',
}


def _digest(text):
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16)


def canonical_pair(a, b):
    """Normalize whitespace/orientation, refusing self-pairs and empty names.

    Symbols remain case-sensitive. No biological alias resolution is inferred.
    """
    a, b = str(a).strip(), str(b).strip()
    if not a or not b or a == b or '|' in a or '|' in b:
        raise ValueError('Distinct unambiguous gene names required')
    return tuple(sorted((a, b)))


def validate_screen(screen):
    genes, pairs = np.asarray(screen['genes']), np.asarray(screen['pairs'])
    n, m = len(genes), len(pairs)
    if genes.ndim != 1 or genes.dtype.kind != 'U' or len(set(genes)) != n:
        raise ValueError('Unique Unicode gene names required')
    if any(not g or g != g.strip() or '|' in g for g in genes):
        raise ValueError('Canonical gene names required')
    if pairs.shape != (m, 2) or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError('Integer pair matrix required')
    if np.any(pairs < 0) or np.any(pairs >= n):
        raise ValueError('Pair index outside gene schema')
    if np.any(pairs[:, 0] >= pairs[:, 1]) or len(np.unique(pairs, axis=0)) != m:
        raise ValueError('Unique canonical pairs required')
    if m and np.any(genes[pairs[:, 0]] >= genes[pairs[:, 1]]):
        raise ValueError('Pair names must also have canonical orientation')
    singles, reps, counts = [np.asarray(screen[k]) for k in ('singles', 'replicates', 'counts')]
    if singles.shape != (n,) or reps.shape != (m, 2) or counts.shape != (m, 2):
        raise ValueError('Scalar two-infection schema mismatch')
    if not np.isfinite(singles[pairs]).all() or not np.isfinite(reps).all():
        raise ValueError('Queried singles and targets must be finite')
    if not np.isfinite(counts).all() or np.any(counts < 2) or np.any(counts != counts.astype(np.int64)):
        raise ValueError('At least two integer construct counts per infection required')
    responses = np.asarray(screen['responses'])
    if responses.shape != (m,) or not np.allclose(responses, reps.mean(1), rtol=0, atol=1e-15):
        raise ValueError('Equal-infection aggregation mismatch')
    if screen.get('units') != 'gamma':
        raise ValueError('Only declared gamma units supported by this adapter')
    for key in ('source', 'accession', 'context', 'exclusions'):
        if key not in screen:
            raise ValueError(f'Missing metadata: {key}')


def _verified(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'Historical bank hash mismatch: {path}')
    return load_npz(path)


def _verify_optional_raw(path, expected):
    if not path.exists():
        return False
    if sha256(path) != expected:
        raise ValueError(f'Historical source hash mismatch: {path}')
    return True


def load_screen_roster(original_root, external_prepared):
    """Load precisely the six Q2/Q3 populations, verifying historical banks.

    original_root can be the preparation directory or its containing data root.
    External genes/pairs are source-local, without original-bank restriction.
    """
    original_root, external_prepared = Path(original_root), Path(external_prepared)
    original = original_root / 'profiles.npz'
    if not original.exists():
        original = original_root / 'prepared' / 'profiles.npz'
    d = _verified(original, ORIGINAL_HASH)
    original_prep = json.loads(original.with_name('preparation.json').read_text(encoding='utf-8'))
    if original_prep['source_sha256'] != ORIGINAL_SOURCE_HASH or original_prep['profiles_sha256'] != ORIGINAL_HASH:
        raise ValueError('Historical original preparation receipt changed')
    original_raw_verified = _verify_optional_raw(original.parent.parent/'GSE116198_sgRNA_pair_phenotypes.txt.gz', ORIGINAL_SOURCE_HASH)
    if d['genes'].shape != (467,) or d['pairs'].shape != (81466, 2):
        raise ValueError('Historical Horlbeck population changed')
    roster = {}
    for c, context in enumerate(('K562', 'Jurkat')):
        if str(d['contexts'][c]) != context:
            raise ValueError('Historical context alignment changed')
        screen = dict(genes=d['genes'].copy(), pairs=d['pairs'].copy(),
            singles=d['singles'][c].mean(0), single_replicates=d['singles'][c].T.copy(),
            single_counts=d['single_counts'][c].T.copy(),
            replicates=d['replicates'][c].T.copy(), responses=d['responses'][c].copy(),
            counts=d['counts'][c].T.copy(), source='Horlbeck 2018', accession='GSE116198',
            context=context, units='gamma', exclusions={'input_pairs': 81466, 'excluded_pairs': 0},
            bank_path=str(original.resolve()), bank_sha256=ORIGINAL_HASH,
            source_sha256=ORIGINAL_SOURCE_HASH, raw_source_hash_verified=original_raw_verified,
            control_singles_summary=d['singles'].mean(axis=(0, 1)),
            control_counts=d['counts'].min(axis=0).T.copy(),
            exposure='development, previously scored; not independent confirmation')
        validate_screen(screen)
        roster['horlbeck_'+context] = screen
    prep_path = external_prepared/'preparation.json'
    if sha256(prep_path) != EXTERNAL_PREPARATION_HASH:
        raise ValueError('Historical external preparation receipt changed')
    prep = json.loads(prep_path.read_text(encoding='utf-8'))
    for task, digest in EXTERNAL_HASHES.items():
        if prep['cohorts'][task]['output_sha256'] != digest:
            raise ValueError('Historical preparation receipt disagrees with bank identity')
        path = external_prepared / (task+'.npz')
        raw_verified = _verify_optional_raw(external_prepared.parent/prep['cohorts'][task]['source_path'],
                                          prep['cohorts'][task]['source_sha256'])
        d = _verified(path, digest)
        names, genes = d['pair_names'], d['genes']
        if genes.dtype.kind != 'U' or len(np.unique(genes)) != len(genes):
            raise ValueError('Unique Unicode source-local genes required')
        if names.ndim != 2 or names.shape[1] != 2 or np.any(names[:, 0] >= names[:, 1]) or len(np.unique(names, axis=0)) != len(names):
            raise ValueError('Duplicate or noncanonical source pairs')
        if len(np.unique(d['single_genes'])) != len(d['single_genes']):
            raise ValueError('Duplicate source single identity')
        lookup = {str(g): i for i, g in enumerate(genes)}
        if any(str(g) not in lookup for g in names.ravel()) or any(str(g) not in lookup for g in d['single_genes']):
            raise ValueError('Pair/single identity outside source genes')
        pairs = np.array([[lookup[str(a)], lookup[str(b)]] for a, b in names], dtype=np.int64)
        sr = np.full((len(genes), 2), np.nan)
        sc = np.zeros((len(genes), 2), dtype=np.int64)
        ids = np.array([lookup[str(g)] for g in d['single_genes']], dtype=np.int64)
        if d['singles'].shape != (len(ids), 2) or d['single_counts'].shape != (len(ids), 2):
            raise ValueError('Invalid source single measurement schema')
        sr[ids], sc[ids] = d['singles'], d['single_counts']
        single_ok = np.isfinite(sr).all(1) & (sc >= 2).all(1)
        local_singles = single_ok[pairs].all(1)
        targets_ok = np.isfinite(d['gamma']).all(1)
        counts_ok = np.isfinite(d['counts']).all(1) & (d['counts'] >= 2).all(1)
        keep = local_singles & targets_ok & counts_ok
        excluded = dict(input_pairs=len(pairs), excluded_pairs=int((~keep).sum()),
            missing_or_low_count_local_single_pairs=int((~local_singles).sum()),
            nonfinite_target_pairs=int((~targets_ok).sum()), low_count_pairs=int((~counts_ok).sum()))
        screen = dict(genes=genes.copy(), pairs=pairs[keep], singles=sr.mean(1),
            single_replicates=sr, single_counts=sc, replicates=d['gamma'][keep],
            responses=d['gamma'][keep].mean(1), counts=d['counts'][keep],
            source='environmental' if task.startswith('environment') else 'PARPi',
            accession='GSE312636' if task.startswith('environment') else 'GSE284018',
            context='K562_'+task, units='gamma', exclusions=excluded,
            bank_path=str(path.resolve()), bank_sha256=digest,
            source_sha256=prep['cohorts'][task]['source_sha256'],
            raw_source_hash_verified=raw_verified,
            exposure='exploratory reuse, previously scored source; not independent confirmation')
        validate_screen(screen)
        roster[task] = screen
    return roster


def pair_assignments(screen):
    """Plan's ordinary 60/20/20 source-accession pair split."""
    genes, accession = screen['genes'], screen['accession']
    codes = np.array([_digest(f'opal-reset-v2|{accession}|{genes[a]}|{genes[b]}') % 10
                      for a, b in screen['pairs']])
    return np.where(codes < 6, 0, np.where(codes < 8, 1, 2)).astype(np.int8)


def block_assignments(screen):
    """Five disjoint within-gene-fold blocks; remaining pairs hash 80/20."""
    genes, pairs, accession = screen['genes'], screen['pairs'], screen['accession']
    folds = np.array([_digest('opal-reset-block-v1|'+str(g)) % 5 for g in genes])
    validation = np.array([_digest(f'opal-reset-v2|{accession}|{genes[a]}|{genes[b]}') % 5 == 0
                           for a, b in pairs])
    return [np.where((folds[pairs] == fold).all(1), 2, validation.astype(np.int8)).astype(np.int8)
            for fold in range(5)]


def _split(screen, split):
    split = np.asarray(split)
    if split.shape != (len(screen['pairs']),) or not np.isin(split, [0, 1, 2]).all():
        raise ValueError('Aligned train/validation/test assignment required')
    return split


def split_report(screen, split):
    split = _split(screen, split)
    pairs, n = screen['pairs'], len(screen['genes'])
    degree = np.bincount(pairs[split == 0].ravel(), minlength=n)
    test = pairs[split == 2]
    unsupported = np.unique(pairs[split != 0][degree[pairs[split != 0]] == 0])
    return dict(train_pairs=int((split == 0).sum()), validation_pairs=int((split == 1).sum()),
        test_pairs=len(test), test_genes=len(np.unique(test)),
        unsupported_query_genes=screen['genes'][unsupported].tolist(),
        eligible=bool(len(test) >= 50 and len(np.unique(test)) >= 10 and len(unsupported) == 0))


def training_statistics(screen, split, singles_summary=None):
    """Freeze graph/single/count strata without accessing pair responses."""
    split = _split(screen, split)
    pairs, n = screen['pairs'], len(screen['genes'])
    train = np.flatnonzero(split == 0)
    if not len(train):
        raise ValueError('Nonempty training graph required')
    degree = np.bincount(pairs[train].ravel(), minlength=n)
    active = degree > 0
    singles = np.asarray(screen.get('control_singles_summary', screen['singles'])
                         if singles_summary is None else singles_summary)
    if singles.shape != (n,) or not np.isfinite(singles[active]).all():
        raise ValueError('Finite single summary for active training genes required')
    degree_cuts = np.quantile(degree[active], [1/3, 2/3])
    single_cuts = np.quantile(abs(singles[active]), [1/3, 2/3])
    degree_bins = np.searchsorted(degree_cuts, degree, side='right')
    single_bins = np.searchsorted(single_cuts, abs(singles), side='right')
    ends = pairs[train]
    db, sb = np.sort(degree_bins[ends], axis=1), np.sort(single_bins[ends], axis=1)
    counts = np.min(screen.get('control_counts', screen['counts'])[train], axis=1)
    if not np.isfinite(counts).all() or np.any(counts < 2):
        raise ValueError('Eligible construct counts required')
    count_bins = np.where(counts < 4, 0, np.where(counts < 8, 1, 2))
    strata = np.column_stack((db, sb, count_bins)).astype(np.int8)
    return dict(train_indices=train, degree=degree, active_genes=np.flatnonzero(active),
        degree_cutpoints=degree_cuts, single_cutpoints=single_cuts, strata=strata)


def residual_permutations(screen, split, singles_summary=None, seeds=CONTROL_SEEDS):
    """Return absolute training-row permutations, never shuffling graph edges.

    For residual vectors ordered like train_indices, use residual[local_permutation].
    For arrays indexed by all pairs, use residual[permutation]. Joint contexts
    must pass the same shared single summary and reuse these exact permutations.
    """
    stats = training_statistics(screen, split, singles_summary)
    train, strata = stats['train_indices'], stats['strata']
    _, labels = np.unique(strata, axis=0, return_inverse=True)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    eligible = sum(len(group) for group in groups if len(group) > 1) / len(train)
    result = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = np.arange(len(train))
        for group in groups:
            if len(group) > 1:
                order[group] = rng.permutation(group)
        result.append(dict(seed=int(seed), train_indices=train.copy(),
            permutation=train[order], local_permutation=order,
            moved_fraction=float(np.mean(order != np.arange(len(train)))),
            permutable_fraction=float(eligible), sufficient=bool(eligible >= .8),
            strata=strata.copy(), degree_cutpoints=stats['degree_cutpoints'].tolist(),
            single_cutpoints=stats['single_cutpoints'].tolist()))
    return result


def metrics(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.ndim != 1 or not len(target) or target.shape != prediction.shape:
        raise ValueError('Aligned nonempty one-dimensional vectors required')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Finite targets and predictions required')
    error = prediction-target
    pearson = None if len(target) < 2 or target.std() == 0 or prediction.std() == 0 else float(np.corrcoef(target, prediction)[0, 1])
    return dict(mse=float(np.mean(error**2)), mae=float(np.mean(abs(error))),
        bias=float(error.mean()), pearson=pearson, target_variance=float(target.var()), pairs=len(target))


def paired_gene_uncertainty(loss_difference, pairs, n_genes=None, draws=2000, seed=20261022):
    """Endpoint-multiplicity bootstrap, conditional on the measured screen.

    Resample active genes, use the product of endpoint multiplicities as edge
    weight, discard empty draws, and report leave-one-gene-out influence.
    These bounds do not estimate independent-culture or source uncertainty.
    """
    delta, pairs = np.asarray(loss_difference), np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError('Integer endpoint matrix required')
    if delta.shape != (len(pairs),) or not len(delta) or not np.isfinite(delta).all() or draws < 1:
        raise ValueError('Finite paired differences and positive draw count required')
    if np.any(pairs < 0) or np.any(pairs[:, 0] >= pairs[:, 1]) or len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError('Unique canonical pairs required')
    if n_genes is not None and np.any(pairs >= n_genes):
        raise ValueError('Pair index outside gene schema')
    active, inverse = np.unique(pairs, return_inverse=True)
    edges, g = inverse.reshape(-1, 2), len(active)
    rng, values = np.random.default_rng(seed), []
    for _ in range(draws):
        count = np.bincount(rng.integers(g, size=g), minlength=g)
        weights = count[edges[:, 0]]*count[edges[:, 1]]
        if weights.sum():
            values.append(float(weights @ delta / weights.sum()))
    if not values:
        raise ValueError('All gene-bootstrap draws empty')
    influence = {}
    for gene in active:
        keep = ~(pairs == gene).any(1)
        influence[str(int(gene))] = float(delta[keep].mean()) if keep.any() else None
    interval = np.quantile(values, [.025, .975]).tolist()
    return dict(mean=float(delta.mean()), interval=interval, gene_conditional_95=interval,
        requested_draws=int(draws), nonempty_draws=len(values), discarded_empty_draws=draws-len(values),
        active_genes=g, leave_one_gene_out=influence,
        uncertainty='conditional gene sensitivity; no biological-replication confidence claim')
