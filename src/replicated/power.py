"""Procedure-wide fixed-panel sensitivity with repeated development selection.

These conditional local alternatives calibrate an exact gain after each fitted
procedure. They test inference/selection machinery, not a biological generator.
"""
import time
import json
import numpy as np
from scipy.stats import beta
from .models import select, fit_candidate, predict_candidate, shuffled_profiles
from .evaluation import cube, score


def probability_interval(successes, trials):
    """Exact binomial Monte Carlo uncertainty, separate from study uncertainty."""
    if not trials:
        return [None, None]
    return [0. if successes == 0 else float(beta.ppf(.025, successes, trials-successes+1)),
            1. if successes == trials else float(beta.ppf(.975, successes+1, trials-successes))]


def calibrated_mean(reference, candidate, noise_variance, gain):
    """Choose a local alternative with known conditional expected MSE gain.

    For mean r+t(c-r), the expected gain is (2t-1)D/(t*t*D+V).
    A negative discriminant means the requested gain is infeasible, not a
    reason to discard this fit and repeat until a favorable procedure appears.
    """
    distance=float(np.mean((candidate-reference)**2))
    if distance <= 1e-15:
        return reference.copy() if gain == 0 else None
    if gain == 0:
        return (reference+candidate)/2
    discriminant=1-gain*(1+gain*noise_variance/distance)
    if discriminant < 0:
        return None
    # Rationalized root avoids cancellation at small gains.
    t=(1+gain*noise_variance/distance)/(1+np.sqrt(discriminant))
    return reference+t*(candidate-reference)


def procedure_power(pilot, menu, designs, draws=100, seed=1729,
                    effects=(0.,.02,.05,.075,.10), variance_multipliers=(1.,2.25,4.),
                    shuffle_seeds=(1729,2718,3141), maximum_seconds=300):
    """Repeat selection/refit/shuffles, then simulate withheld outcomes.

    Only allowed development rows enter the pilot. Whole culture blocks preserve
    empirical pair/component/shared-reference dependence. Query reference blocks
    and outcome blocks are independently drawn, an explicit sensitivity
    assumption rather than an asserted decomposition of biological variance.
    """
    started=time.monotonic()
    if draws < 10 or not designs or not shuffle_seeds:
        raise ValueError('Nonempty designs/shuffles and at least ten draws required')
    if np.any(~np.isin(pilot['partitions'],['train','validation'])):
        raise ValueError('Procedure power accepts development responses only')
    if any(not 0<=e<1 for e in effects) or any(not np.isfinite(v) or v<=0 for v in variance_multipliers):
        raise ValueError('Invalid power effect or variance sensitivity')
    singles,cs,pairs=cube(pilot['singles'],pilot['culture_ids'],pilot['pair_ids'])
    responses,_,_=cube(pilot['responses'],pilot['culture_ids'],pilot['pair_ids'])
    if len(cs)<3 or len(pairs)<6:
        raise ValueError('At least three complete pilot cultures and six global pairs required')
    mean_y=responses.mean(0)
    noise=(responses-mean_y)*np.sqrt(len(cs)/(len(cs)-1))
    rng=np.random.default_rng(seed)
    reports=[]
    for design in designs:
        development=int(design['development_cultures']);confirmation=int(design['confirmation_cultures'])
        if development<3 or confirmation<3:
            raise ValueError('At least three development and confirmation units required')
        fraction=float(design.get('pair_fraction',1.))
        if not 0<fraction<=1:
            raise ValueError('Pair density fraction must lie in (0,1]')
        targets=sorted({c for p in pairs for c in json.loads(str(p))})
        target_count=int(design.get('target_count',len(targets)))
        if not 4<=target_count<=len(targets):raise ValueError('Unsupported target count')
        retained=set(rng.choice(targets,target_count,replace=False))
        eligible=np.asarray([i for i,p in enumerate(pairs) if set(json.loads(str(p)))<=retained])
        rng.shuffle(eligible)
        indices=eligible[:int(np.floor(len(eligible)*fraction))]
        if len(indices)<6:raise ValueError('Design leaves too few supported pair identities')
        ntrain=max(2,int(.6*len(indices)));nval=max(1,int(.2*len(indices)))
        train=indices[:ntrain];val=indices[ntrain:ntrain+nval];held=indices[ntrain+nval:]
        if not len(held):raise ValueError('Design leaves no held pair identities')
        reference_multiplier=float(design.get('reference_noise_multiplier',1.))
        pair_multiplier=float(design.get('pair_noise_multiplier',1.))
        if min(reference_multiplier,pair_multiplier)<0 or not np.isfinite([reference_multiplier,pair_multiplier]).all():
            raise ValueError('Invalid reference/pair noise allocation')
        mean_s=singles.mean(0)
        reference_noise=(singles-mean_s)*np.sqrt(len(cs)/(len(cs)-1))
        def bank(count, chosen, labels):
            samples=rng.integers(0,len(cs),count)
            refs=mean_s[chosen][None,:,:,:]+reference_multiplier*reference_noise[samples][:,chosen]
            values=mean_y[chosen][None,:,:]+pair_multiplier*noise[rng.integers(0,len(cs),count)][:,chosen]
            return {'singles':refs.reshape(-1,2,len(pilot['genes'])),
                    'responses':values.reshape(-1,len(pilot['genes'])),
                    'pair_ids':np.tile(pairs[chosen],count),
                    'culture_ids':np.repeat([f'unit-{i}' for i in range(count)],len(chosen)),
                    'partitions':np.tile(labels,count)}
        for multiplier in variance_multipliers:
            for effect in effects:
                positive=gate=covered=feasible=0;widths=[];selected={};skipped=0
                for _ in range(draws):
                    if time.monotonic()-started>maximum_seconds:
                        raise RuntimeError('Procedure-wide simulation exceeded the frozen compute budget')
                    dev=bank(development,np.r_[train,val],np.array(['train']*len(train)+['validation']*len(val)))
                    query=bank(confirmation,held,np.array(['test']*len(held)))
                    # The query outcomes generated above are never consumed by selection.
                    query.pop('responses')
                    chosen=select(dev,menu);mask=np.ones(len(dev['responses']),bool)
                    predictions={'additive':query['singles'].sum(1)}
                    for role in ('candidate','reference'):
                        cfg=chosen[role]
                        predictions[role]=predict_candidate(cfg,fit_candidate(cfg,dev,mask),query)
                    key=chosen['candidate']['id']+' / '+chosen['reference']['id']
                    selected[key]=selected.get(key,0)+1
                    for shuffle_seed in shuffle_seeds:
                        shuffled=shuffled_profiles(dev,shuffle_seed);sq=shuffled_profiles(query,shuffle_seed)
                        cfg=select(shuffled,menu)['candidate']
                        predictions['shuffle_'+str(shuffle_seed)]=predict_candidate(cfg,fit_candidate(cfg,shuffled,mask),sq)
                    pool=pair_multiplier*np.sqrt(multiplier)*noise[:,held]
                    variance=float(np.mean(pool**2))
                    truth_mean=calibrated_mean(predictions['reference'],predictions['candidate'],variance,effect)
                    if truth_mean is None:
                        skipped+=1
                        continue
                    errors=pool[rng.integers(0,len(cs),confirmation)].reshape(truth_mean.shape)
                    report=score(predictions,truth_mean+errors,query['culture_ids'],query['pair_ids'],
                                 predictions['additive'],measurement_pass=True,attribution_pass=True)
                    interval=report['paired_intervals']['candidate'];low,high=interval['difference_interval']
                    expected=float(np.mean((predictions['reference']-truth_mean)**2-(predictions['candidate']-truth_mean)**2))
                    positive+=low>0;gate+=report['primary_claim_pass'];covered+=low<=expected<=high
                    feasible+=1;widths.append(high-low)
                reports.append({'design':dict(design),'pairs':len(indices),'train_pairs':len(train),
                    'validation_pairs':len(val),'held_pairs':len(held),'target_count':target_count,
                    'effect':effect,'variance_multiplier':multiplier,'draws':draws,'feasible_draws':feasible,
                    'infeasible_draws':skipped,'positive_detection_power':positive/draws,
                    'full_gate_power':gate/draws,'interval_coverage_given_feasible':covered/feasible if feasible else None,
                    'mean_interval_width':float(np.mean(widths)) if widths else None,
                    'positive_detection_mc_interval':probability_interval(positive,draws),
                    'full_gate_mc_interval':probability_interval(gate,draws),'selections':selected})
    null=[r for r in reports if r['effect']==0]
    return {'schema':'opal-procedure-power-v1','reports':reports,'seed':seed,
            'selection_repeated':True,'shuffle_selection_repeated':True,'pilot_cultures':len(cs),
            'elapsed_seconds':time.monotonic()-started,
            'maximum_null_false_positive':max((r['positive_detection_power'] for r in null),default=None),
            'confirmation_design_qualified':False,'scientific_sample_size_recommended':None,
            'scope':'conditional local-alternative fixed-panel procedure sensitivity; not biological design qualification',
            'assumptions':['whole empirical culture blocks are exchangeable',
                'reference and outcome blocks are independently sampled',
                'mean alternative is calibrated along the fitted candidate/reference contrast',
                'pilot covariance does not identify biological and technical variance separately',
                'target subsets are sensitivities within the pilot graph, not new-target population inference'],
            'all_simulated_inputs_are_development_derived':True,'confirmation_outcomes_used_for_selection':False,
            'measurement_attribution_gates':'assumed passed for conditional software power; not empirically qualified'}
