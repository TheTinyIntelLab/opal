"""Development-only depth allocation and complete outcome interpretation."""
import numpy as np
from .data import prepare,validate_preparation_access
from .models import select,fit_candidate,predict_candidate,balanced_weights


def reference_allocation(rows,counts,genes,contract,menu,fractions=(.25,.5,1.),seeds=(1729,2718)):
    """Read-count thinning, with each reference row drawn once for all pairs.

    This technical allocation diagnostic preserves shared count draws. Aggregate
    reads need not be independent molecules, so this is not culture variance.
    """
    validate_preparation_access(rows,contract)
    x=np.asarray(counts)
    if np.any(x!=np.floor(x)):raise ValueError('Read-count thinning requires original integer counts')
    bank,_=prepare(rows,x,genes,contract)
    chosen=select(bank,menu);train=bank['partitions']=='train';val=bank['partitions']=='validation'
    query={k:v[val] if k in {'singles','responses','pair_ids','culture_ids','row_ids','partitions'} else v for k,v in bank.items()}
    weights=balanced_weights(query['pair_ids'],query['culture_ids'])
    report=[]
    for kind in ('control','single','pair'):
        for fraction in fractions:
            if not 0<fraction<=1:raise ValueError('Thinning fraction must lie in (0,1]')
            for seed in seeds:
                rng=np.random.default_rng(seed);draw=x.copy()
                for i,row in enumerate(rows):
                    if row['condition_kind']==kind:draw[i]=rng.binomial(x[i].astype(np.int64),fraction)
                sampled,_=prepare(rows,draw,genes,contract)
                model=fit_candidate(chosen['candidate'],sampled,train)
                query_draw={k:v[val] if k in {'singles','responses','pair_ids','culture_ids','row_ids','partitions'} else v for k,v in sampled.items()}
                pred=predict_candidate(chosen['candidate'],model,query_draw)
                report.append({'varied':kind,'fraction':fraction,'seed':seed,
                    'mse_against_full_measured_development_target':float(weights@np.mean((pred-query['responses'])**2,axis=1))})
    return {'rows':report,'selection':chosen,'evaluator_access':False,
            'reference_draw_policy':'each complete control/single count row drawn once and reused across affected pairs',
            'scope':'technical read-count thinning; not an independent-reference or biological precision estimate'}


def interpret(evidence, materiality=.05):
    """Section 18 branches, with absent dependencies represented explicitly."""
    if evidence.get('release_integrity') is False:
        return {'decision':'release_checks_failed','next_action':'Repair the named packaging or reporting defect'}
    if evidence.get('independent_cultures_verified') is False:
        return {'decision':'culture_provenance_unqualified','next_action':'Downgrade to processing diagnostic or obtain primary initiation records'}
    if evidence.get('precision_adequate') is False:
        return {'decision':'inconclusive_precision','next_action':'Report precision limits; follow the frozen stopping rule'}
    if evidence.get('measurement_pass') is False:
        return {'decision':'measurement_failure','next_action':'Improve measurement or register a different endpoint before another experiment'}
    if evidence.get('gain_disappears_with_independent_references') is True:
        return {'decision':'main_reference_repair','next_action':'Evaluate calibration and measurement utility without an interaction-recovery claim'}
    if evidence.get('aligned_profile_information') is False:
        return {'decision':'alignment_unestablished','next_action':'Do not claim profile-derived interaction information'}
    if evidence.get('primary_claim_pass') is True:
        if evidence.get('state_transfer_pass') is False:
            return {'decision':'within_state_only','next_action':'Retain fixed-state scope and investigate the named state shift separately'}
        return {'decision':'advance_fixed_state','next_action':'Replicate, then register separate crossed-state confirmation'}
    if evidence.get('component_additive_pass') is True and evidence.get('pair_product_pass') is False:
        return {'decision':'component_coverage_only','next_action':'Validate a small component reference and its unsupported-component limits'}
    if evidence.get('primary_claim_pass') is False and evidence.get('registered_secondary_pass') is True:
        return {'decision':'secondary_only','next_action':'Report the failed primary and narrower secondary; use a new task-specific confirmation'}
    upper=evidence.get('gain_upper')
    if upper is not None and upper<materiality and evidence.get('measurement_pass') is True:
        return {'decision':'precise_prediction_failure','next_action':'Report the excluded useful gain; investigate one named information hypothesis'}
    return {'decision':'inconclusive_or_unqualified','next_action':'Resolve named evidence dependencies before advancing'}
