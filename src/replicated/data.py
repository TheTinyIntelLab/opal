"""Aggregate-read adapter with physically separate outcome preparation."""
import numpy as np
from .contracts import qualify, validate_access, identity, canonical_component


def validate_preparation_access(rows, contract, include_outcomes=False):
    """Authorize metadata before opening or converting a response count bank."""
    if not include_outcomes and any(r['condition_kind']=='pair' and
            (r['partition'] in {'test','guard'} or r['culture_id'] in contract['confirmation_cultures']) for r in rows):
        raise ValueError('Analyst adapter received sealed pair rows')
    if not include_outcomes:
        validate_access(rows, {'normalization': [r['row_id'] for r in rows]},
                        contract.get('held_pair_ids', []), contract['confirmation_cultures'],
                        contract.get('aliases'))


def prepare(rows, counts, genes, contract, include_outcomes=False):
    """Counts are aggregate counts per metadata row, never logged cell means.

    The custodian invokes include_outcomes on separate storage. The analyst
    entry point rejects all test/confirmation pair rows before touching counts.
    """
    validate_preparation_access(rows,contract,include_outcomes)
    counts = np.asarray(counts, float)
    if counts.shape != (len(rows), len(genes)) or not len(set(genes))==len(genes) or not len(genes):
        raise ValueError('Count/gene schema mismatch')
    if not np.isfinite(counts).all() or np.any(counts<0) or np.any(counts.sum(1)<=0):
        raise ValueError('Invalid aggregate counts')
    if contract['normalization'] != 'library_total' or contract['transform'] != 'log2_after_reference_mean':
        raise ValueError('Unsupported assay normalization contract')
    target, pseudo = contract['library_target'], contract['pseudocount']
    if not np.isfinite([target, pseudo]).all() or target<=0 or pseudo<=0:
        raise ValueError('Normalization constants must be positive and frozen')
    q = qualify(rows, contract.get('aliases'))
    by_id = {r['row_id']: i for i,r in enumerate(rows)}
    normalized = counts/counts.sum(1)[:,None]*target
    def mean(ids):
        return normalized[[by_id[i] for i in ids]].mean(0)
    pair_rows = [r for r in rows if r['row_id'] in q['matches']]
    singles, responses, pair_ids, culture_ids, row_ids, partitions = [], [], [], [], [], []
    for r in pair_rows:
        refs = q['matches'][r['row_id']]
        control = np.log2(pseudo+mean(refs['controls']))
        order=np.argsort([canonical_component(c,contract.get('aliases')) for c in r['components']])
        keys=('single_a','single_b')
        singles.append([np.log2(pseudo+mean(refs[keys[i]]))-control for i in order])
        responses.append(np.log2(pseudo+normalized[by_id[r['row_id']]])-control)
        pair_ids.append(identity(r, contract.get('aliases')))
        culture_ids.append(r['culture_id']); row_ids.append(r['row_id']); partitions.append(r['partition'])
    if not singles:
        raise ValueError('No eligible matched pair measurements')
    # Explicit labels only; query reference preparation is handled separately.
    return {'singles': np.asarray(singles), 'responses': np.asarray(responses),
            'pair_ids': np.asarray(pair_ids), 'culture_ids': np.asarray(culture_ids),
            'row_ids': np.asarray(row_ids), 'partitions': np.asarray(partitions),
            'genes': np.asarray(genes), 'scale': np.asarray(contract['scale']),
            'units': np.asarray(contract['units'])}, q


def query_references(rows, counts, genes, queries, contract):
    """Query identities are metadata; their outcomes are never present."""
    if any(r['condition_kind']=='pair' for r in rows):
        raise ValueError('Query reference bank contains pair outcomes')
    # Zero-count placeholders are only used for eligibility, never normalized.
    from .contracts import validate_rows
    validate_rows(rows+queries)
    if not queries or any(r['condition_kind']!='pair' or r['partition']!='test' for r in queries):
        raise ValueError('Query identities must be nonempty test pairs')
    if not len(genes) or len(set(genes))!=len(genes):
        raise ValueError('Count/gene schema mismatch')
    target,pseudo=contract['library_target'],contract['pseudocount']
    if not np.isfinite([target,pseudo]).all() or target<=0 or pseudo<=0:
        raise ValueError('Normalization constants must be positive and frozen')
    x = np.asarray(counts, float)
    if x.shape != (len(rows),len(genes)) or np.any(x<0) or not np.isfinite(x).all() or np.any(x.sum(1)<=0):
        raise ValueError('Invalid query reference counts')
    if contract['normalization'] != 'library_total' or contract['transform'] != 'log2_after_reference_mean':
        raise ValueError('Unsupported assay normalization')
    q = qualify(rows+queries, contract.get('aliases'))
    if q['rejected']:
        raise ValueError('Unsupported query: missing matched reference')
    lookup = {r['row_id']: i for i,r in enumerate(rows)}
    x = x/x.sum(1)[:,None]*contract['library_target']
    profiles=[]
    for r in queries:
        refs=q['matches'][r['row_id']]
        avg=lambda key: x[[lookup[i] for i in refs[key]]].mean(0)
        ctrl=np.log2(contract['pseudocount']+avg('controls'))
        order=np.argsort([canonical_component(c,contract.get('aliases')) for c in r['components']])
        keys=('single_a','single_b')
        profiles.append([np.log2(contract['pseudocount']+avg(keys[i]))-ctrl for i in order])
    return {'singles':np.asarray(profiles), 'pair_ids':np.asarray([identity(r,contract.get('aliases')) for r in queries]),
            'culture_ids':np.asarray([r['culture_id'] for r in queries]),
            'row_ids':np.asarray([r['row_id'] for r in queries]),'genes':np.asarray(genes),
            'scale':np.asarray(contract['scale']),'units':np.asarray(contract['units'])}
