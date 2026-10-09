"""Metadata-first eligibility and global response access contracts."""
import hashlib
import json
from decimal import Decimal, InvalidOperation
from collections import Counter
import numpy as np


REQUIRED = {'row_id', 'study_id', 'source_id', 'culture_id', 'culture_initiation',
            'state_id', 'assay_id', 'processing_batch_id', 'control_stratum_id',
            'condition_kind', 'components', 'reagent_ids', 'doses', 'dose_units',
            'treatment_time', 'treatment_order', 'replicate_type', 'count_provenance',
            'partition', 'culture_evidence'}


def canonical_component(component, aliases=None):
    """Resolve alias chains; cycles cannot define a global intervention identity."""
    aliases = aliases or {}
    current = str(component)
    seen = set()
    while current in aliases:
        target = str(aliases[current])
        if target == current:
            break
        if current in seen:
            raise ValueError('Cyclic component aliases')
        seen.add(current)
        current = target
    if not current.strip():
        raise ValueError('Empty component identity')
    return current


def identity(row, aliases=None, dose_specific=False):
    """Global pair IDs ignore dose/state; condition IDs preserve assigned doses."""
    aliases = aliases or {}
    components = [canonical_component(c, aliases) for c in row['components']]
    if len(components) != len(set(components)):
        raise ValueError('Repeated intervention is outside the distinct-pair contract')
    if dose_specific:
        values = []
        for c, d, u, reagent in zip(components, row['doses'], row['dose_units'], row['reagent_ids']):
            try:
                d = Decimal(str(d))
            except InvalidOperation as exc:
                raise ValueError('Unknown dose') from exc
            if not d.is_finite() or d < 0 or not u:
                raise ValueError('Invalid dose or units')
            values.append((c, str(d.normalize()), str(u), str(reagent)))
    else:
        values = components
    if row['treatment_order'] not in ('simultaneous', None):
        raise ValueError('Sequential treatment needs a separate contract')
    return json.dumps(sorted(values), separators=(',', ':'))


def validate_rows(rows):
    ids = set()
    for row in rows:
        if REQUIRED - row.keys():
            raise ValueError('Missing metadata fields: ' + ', '.join(sorted(REQUIRED-row.keys())))
        if not row['row_id'] or row['row_id'] in ids:
            raise ValueError('Duplicate or empty row identity')
        ids.add(row['row_id'])
        kind = row['condition_kind']
        if kind not in {'control', 'single', 'pair'} or len(row['components']) != {'control': 0, 'single': 1, 'pair': 2}[kind]:
            raise ValueError('Invalid condition assignment')
        if any(len(row[k]) != len(row['components']) for k in ('doses', 'dose_units', 'reagent_ids')):
            raise ValueError('Component-dose/reagent lengths differ')
        if row['partition'] not in {'reference', 'train', 'validation', 'test', 'guard'}:
            raise ValueError('Unknown partition')
        identity(row, dose_specific=True)
        if not all(row[k] for k in ('study_id', 'source_id', 'state_id', 'assay_id', 'control_stratum_id', 'count_provenance')):
            raise ValueError('Missing measurement provenance')


def stratum(row):
    return tuple(row[k] for k in ('study_id', 'source_id', 'culture_id', 'state_id',
                                  'assay_id', 'control_stratum_id', 'treatment_time'))


def qualify(rows, aliases=None):
    validate_rows(rows)
    refs = {}
    for row in rows:
        if row['condition_kind'] != 'pair':
            refs.setdefault((stratum(row), identity(row, aliases, True)), []).append(row['row_id'])
    matches, rejected = {}, {}
    eligible_pairs, degrees = set(), Counter()
    for row in rows:
        if row['condition_kind'] != 'pair':
            continue
        ids = []
        for indices in ([], [0], [1]):
            ref = dict(row)
            for k in ('components', 'doses', 'dose_units', 'reagent_ids'):
                ref[k] = [row[k][i] for i in indices]
            ids.append(refs.get((stratum(row), identity(ref, aliases, True)), []))
        if not all(ids):
            rejected[row['row_id']] = 'missing exact-dose/time/state/assay/stratum control or single'
        else:
            matches[row['row_id']] = dict(zip(('controls', 'single_a', 'single_b'), ids))
            eligible_pairs.add(identity(row, aliases))
    for p in eligible_pairs:
        degrees.update(json.loads(p))
    def initiation_evidence(r):
        evidence=r['culture_evidence']
        return (isinstance(evidence,dict) and evidence.get('kind')=='independent_initiation'
                and evidence.get('culture_id')==r['culture_id'] and evidence.get('record_id')
                and evidence.get('source'))
    verified = bool(rows) and all(r['culture_id'] and r['culture_initiation'] and
        r['replicate_type'] == 'independent_culture' and initiation_evidence(r) for r in rows)
    return {'matches': matches, 'rejected': rejected, 'eligible_distinct_pairs': len(eligible_pairs),
            'conditions': dict(Counter(r['condition_kind'] for r in rows)),
            'cultures': sorted({r['culture_id'] for r in rows if r['culture_id']}),
            'independent_cultures_verified': verified,
            'role': 'prospective_confirmation_candidate' if verified and matches else 'replicated_diagnostic_only',
            'partner_degree': dict(degrees),
            'degree_summary': [float(f(list(degrees.values()))) for f in (min, np.median, max)] if degrees else [],
            'pair_coverage': len(eligible_pairs)/(len(degrees)*(len(degrees)-1)/2) if len(degrees)>1 else 0,
            'precision_qualified': False}


def split_pairs(rows, counts, seed, aliases=None):
    """One deterministic metadata-only greedy allocation with degree balancing."""
    validate_rows(rows)
    pairs = sorted({identity(r, aliases) for r in rows if r['condition_kind']=='pair'})
    if len(counts) != 3 or min(counts) < 1 or sum(counts) != len(pairs):
        raise ValueError('Split counts must cover distinct global pairs with nonempty partitions')
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    assignment, degrees, used = {}, [Counter() for _ in counts], [0]*3
    names = ['train', 'validation', 'test']
    for p in pairs:
        comps = json.loads(p)
        choices = [i for i, n in enumerate(counts) if used[i]<n]
        i = min(choices, key=lambda i: (sum(degrees[i][c] for c in comps)/counts[i], used[i]/counts[i], i))
        assignment[p] = names[i]
        degrees[i].update(comps)
        used[i] += 1
    train_targets = set(degrees[0])
    support = {p: sum(c in train_targets for c in json.loads(p)) for p in assignment}
    return {'seed': seed, 'assignment': assignment, 'counts': dict(zip(names, used)),
            'partner_degrees': dict(zip(names, map(dict, degrees))), 'supervised_components': support,
            'rows': {r['row_id']: assignment[identity(r, aliases)] for r in rows if r['condition_kind']=='pair'}}


def validate_access(rows, access, held_pairs, confirmation_cultures, aliases=None,
                    allowed_references=('normalization', 'prediction', 'report')):
    validate_rows(rows)
    lookup = {r['row_id']: r for r in rows}
    held, cultures = set(held_pairs), set(confirmation_cultures)
    partitions = {}
    for row in rows:
        if row['condition_kind']=='pair':
            key = identity(row, aliases)
            partitions.setdefault(key, set()).add(row['partition'])
    if any(len(v - {'guard'}) > 1 for v in partitions.values()):
        raise ValueError('Global pair variants cross partitions')
    for stage, ids in access.items():
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate access rows')
        for row_id in ids:
            if row_id not in lookup:
                raise ValueError('Unknown response ID')
            row = lookup[row_id]
            if row['condition_kind']=='pair':
                if identity(row, aliases) in held or row['partition'] in {'test','guard'} or row['culture_id'] in cultures:
                    raise ValueError('Forbidden pair response in ' + stage)
            elif row['culture_id'] in cultures and stage not in allowed_references:
                raise ValueError('Query references not permitted in ' + stage)
    return True


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
