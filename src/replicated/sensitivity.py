"""Deletion diagnostics for frozen predictions, never replacement endpoints."""
import json
import numpy as np
from .evaluation import cube, paired_interval


def deletion_sensitivity(reference, candidate, truth, cultures, pairs):
    """Omit each whole culture or all unordered pairs incident to a component.

    Technical repeats collapse before deletion. Component deletion changes the
    fixed panel; it describes influence without estimating unseen-target risk.
    Predictions are already frozen and no model is selected or refitted here.
    """
    truth=np.asarray(truth,float)
    for prediction in (reference,candidate):
        if np.shape(prediction)!=truth.shape or not np.isfinite(prediction).all():
            raise ValueError('Invalid sensitivity prediction schema')
    if truth.ndim!=2 or not np.isfinite(truth).all():
        raise ValueError('Invalid sensitivity outcome schema')
    ref,cs,ps=cube((reference-truth)**2,cultures,pairs)
    cand,_,_=cube((candidate-truth)**2,cultures,pairs)
    components=[]
    for pair in ps:
        try:parts=json.loads(str(pair))
        except (TypeError,ValueError) as exc:raise ValueError('Canonical pair identities required') from exc
        if not isinstance(parts,list) or len(parts)!=2 or any(not isinstance(p,str) or not p for p in parts) or len(set(parts))!=2:
            raise ValueError('Distinct unordered component identities required')
        components.append(set(parts))
    def summarize(culture_mask,pair_mask):
        r=ref[culture_mask][:,pair_mask].mean(2)
        c=cand[culture_mask][:,pair_mask].mean(2)
        result={'cultures':int(culture_mask.sum()),'pairs':int(pair_mask.sum()),
                'interval':None,'gain':None,'reference_mse':None,'candidate_mse':None}
        if not r.size:
            result['blocked_on']='Deletion leaves no supported pair panel'
            return result
        result.update(reference_mse=float(r.mean()),candidate_mse=float(c.mean()))
        if r.mean()<=0:
            result['blocked_on']='Relative gain requires a positive reference loss'
            return result
        result['gain']=float(1-c.mean()/r.mean())
        if len(r)<3:
            result['blocked_on']='Fewer than three remaining cultures; no paired interval'
        else:
            result['interval']=paired_interval(r.mean(1),c.mean(1))
        return result
    all_cultures=np.ones(len(cs),bool);all_pairs=np.ones(len(ps),bool)
    culture_results={str(c):summarize(cs!=c,all_pairs) for c in cs}
    component_results={component:summarize(all_cultures,np.array([component not in p for p in components]))
                       for component in sorted(set().union(*components))}
    return {'scope':'Exploratory influence on frozen fixed-panel predictions; no unseen-target population inference',
            'technical_rows_collapsed':True,'selection_repeated':False,'refitted':False,
            'can_replace_primary':False,'can_select_favorable_subset':False,
            'leave_one_culture_out':culture_results,'leave_one_component_out':component_results}
