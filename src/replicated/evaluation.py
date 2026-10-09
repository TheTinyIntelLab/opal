"""Fixed-panel culture inference, reference accounting and decision branches."""
import numpy as np
from scipy.stats import t


def cube(values, cultures, pairs):
    """Collapse technical rows before equal-culture, equal-pair aggregation."""
    cultures=np.asarray(cultures); pairs=np.asarray(pairs); values=np.asarray(values)
    cs,ps=np.unique(cultures),np.unique(pairs)
    result=[]
    for c in cs:
        row=[]
        for p in ps:
            idx=(cultures==c)&(pairs==p)
            if not idx.any():raise ValueError('Incomplete confirmation block; balanced support is required')
            row.append(values[idx].mean(0))
        result.append(row)
    return np.asarray(result),cs,ps


def paired_interval(reference, candidate):
    """Student interval for paired culture-mean error difference, fixed panel."""
    ref=np.asarray(reference,float); cand=np.asarray(candidate,float)
    if ref.shape!=cand.shape or ref.ndim!=1 or len(ref)<3 or not np.isfinite(ref).all() or not np.isfinite(cand).all() or ref.mean()<=0:
        raise ValueError('At least three finite independent culture units and positive baseline required')
    d=ref-cand; mean=float(d.mean()); half=float(t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
    gain=float(1-cand.mean()/ref.mean())
    return {'gain':gain,'difference':mean,'difference_interval':[mean-half,mean+half],
            'gain_interval_scaled_by_observed_reference':[(mean-half)/ref.mean(),(mean+half)/ref.mean()],
            'cultures':len(d),'method':'paired culture Student t; fixed registered pair panel',
            'scope':'conditional on this intervention panel; no unseen-target population claim'}


def decomposition(input_additive, predicted, reference_additive, truth, weights=None):
    main=np.asarray(input_additive)-reference_additive
    residual=np.asarray(predicted)-input_additive-(np.asarray(truth)-reference_additive)
    def average(values):
        return float(np.mean(values)) if weights is None else float(np.asarray(weights)@np.mean(values,axis=1))
    lm=average(main**2);lr=average(residual**2);cross=2*average(main*residual)
    total=average((np.asarray(predicted)-truth)**2)
    if not np.isclose(total,lm+lr+cross,rtol=1e-10,atol=1e-12):raise ValueError('Decomposition identity failed')
    return {'main':lm,'residual':lr,'cross_term':cross,'total':total,
            'same_anchor_residual_mse':average((predicted-input_additive-(truth-input_additive))**2)}


def shared_reference_draws(control, singles, pairs, rng, control_sd, single_sd):
    """One complete draw shared by all affected pairs, never per-pair singles."""
    ctrl=np.asarray(control)+rng.normal(size=np.shape(control))*control_sd
    single=np.asarray(singles)+rng.normal(size=np.shape(singles))*single_sd
    effects=single-ctrl
    return effects[pairs[:,0]]+effects[pairs[:,1]],ctrl,single


def score(predictions, truth, cultures, pairs, additive, materiality=.05,
          measurement_pass=False, attribution_pass=False):
    truth=np.asarray(truth,float)
    losses={};mae={};pairwise={};culturewise={};intervals={}
    for name,pred in predictions.items():
        if pred.shape!=truth.shape or not np.isfinite(pred).all():raise ValueError('Prediction schema mismatch')
        squared,cs,ps=cube((pred-truth)**2,cultures,pairs)
        absolute,_,_=cube(np.abs(pred-truth),cultures,pairs)
        losses[name]=float(squared.mean());mae[name]=float(absolute.mean())
        culturewise[name]=squared.mean(axis=(1,2)).tolist()
        pairwise[name]=squared.mean(axis=(0,2)).tolist()
    for name in predictions:
        intervals[name]=paired_interval(culturewise['reference'],culturewise[name])
    controls=[n for n in predictions if n.startswith('shuffle_')]
    shuffle=None
    if controls:
        shuffle=paired_interval(np.mean([culturewise[n] for n in controls],axis=0),culturewise['candidate'])
    candidate=intervals['candidate'];lo,hi=candidate['difference_interval']
    if not measurement_pass:branch='measurement_unqualified'
    elif not attribution_pass:branch='attribution_unqualified'
    elif shuffle is None or shuffle['difference_interval'][0]<=0:branch='alignment_unestablished'
    elif candidate['gain']>=materiality and lo>0 and losses['candidate']<losses['additive']:branch='advance_fixed_state'
    elif hi<materiality*losses['reference']:branch='precise_failure'
    else:branch='inconclusive'
    return {'mse':losses,'mae':mae,'paired_intervals':intervals,'aligned_vs_shuffled':shuffle,
            'pair_ids':ps.tolist(),'culture_ids':cs.tolist(),'pair_errors':pairwise,'culture_errors':culturewise,
            'decision':branch,'materiality':materiality,'measurement_pass':measurement_pass,
            'attribution_pass':attribution_pass,'primary_claim_pass':branch=='advance_fixed_state',
            'all_registered_comparisons_reported':True}


def reliability(bank):
    """Leave-one-culture-out shared correction and forbidden repeated-pair control."""
    residual=bank['responses']-bank['singles'].sum(1)
    raw,cs,ps=cube(residual,bank['culture_ids'],bank['pair_ids'])
    if len(cs)<3:raise ValueError('At least three development cultures required')
    departures=[];repeat_errors=[];shared_errors=[]
    for i in range(len(cs)):
        others=np.delete(raw,i,axis=0)
        shared=others.mean(axis=(0,1))
        departures.append(raw[i]-shared)
        repeat_errors.append(float(np.mean((raw[i]-others.mean(0))**2)))
        shared_errors.append(float(np.mean((raw[i]-shared)**2)))
    departures=np.asarray(departures)
    vectors=departures.reshape(len(cs),-1)
    covariance=np.cov(vectors).tolist()
    interval=paired_interval(shared_errors,repeat_errors)
    return {'repeated_pair_vs_shared':interval,'cross_culture_departure_covariance':covariance,
            'residual_norms':np.linalg.norm(raw,axis=2).tolist(),
            'departure_norms':np.linalg.norm(departures,axis=2).tolist(),
            'cultures':cs.tolist(),'pairs':ps.tolist(),
            'measurement_pass':interval['difference_interval'][0]>0,
            'forbidden_information_control':True,'evaluator_access':False}


def empirical_power(reference, candidate, designs, draws=1000, seed=1729):
    """Empirical culture-block pilot sensitivity for a fixed fitted procedure.

    Whole culture error vectors retain pair/component/shared-reference dependence.
    This cannot extrapolate new target panels or repeat model selection without
    raw development measurements; those designs remain explicitly unqualified.
    """
    ref=np.asarray(reference,float);cand=np.asarray(candidate,float)
    if ref.shape!=cand.shape or ref.ndim!=2 or ref.shape[0]<3 or draws<100:
        raise ValueError('Empirical culture-by-pair errors and >=100 simulations required')
    if not np.isfinite(ref).all() or not np.isfinite(cand).all() or np.any(ref<0) or np.any(cand<0) or ref.mean()<=0:
        raise ValueError('Invalid pilot errors')
    rng=np.random.default_rng(seed)
    noise=ref-cand;noise-=noise.mean(0)
    reports=[]
    for cultures in designs:
        if cultures<3:raise ValueError('At least three cultures per design')
        for sensitivity in (1.,1.5,2.):
            for effect in (0.,.02,.05,.075,.10):
                positive=gate=0; widths=[]
                for _ in range(draws):
                    idx=rng.integers(0,len(ref),cultures)
                    r=ref[idx].mean(1)
                    difference=effect*r+sensitivity*noise[idx].mean(1)
                    c=r-difference
                    result=paired_interval(r,c)
                    low,high=result['difference_interval'];positive+=low>0
                    gate+=low>0 and result['gain']>=.05
                    widths.append(high-low)
                reports.append({'cultures':int(cultures),'effect':effect,'variance_multiplier':sensitivity**2,
                                'positive_detection_power':positive/draws,'point_and_interval_gate_power':gate/draws,
                                'mean_interval_width':float(np.mean(widths))})
    qualified=[r['cultures'] for r in reports if r['effect']==.05 and r['variance_multiplier']==1 and r['positive_detection_power']>=.8]
    null=[r['positive_detection_power'] for r in reports if r['effect']==0]
    return {'reports':reports,'draws':draws,'seed':seed,'pilot_cultures':len(ref),
            'smallest_detection_design':min(qualified) if qualified else None,
            'maximum_null_false_positive':max(null),'null_calibration_pass':max(null)<=.05,
            'scope':'empirical fixed-panel culture block sensitivity; fitted predictor only',
            'selection_repeated':False,'new_target_population_power':False,
            'assumptions':['development cultures exchangeable','same fixed pair panel','whole culture blocks retain shared errors'],
            'confirmation_design_qualified':False}
