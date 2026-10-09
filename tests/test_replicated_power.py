"""Procedure-wide simulation must preserve isolation and honest design limits."""
import numpy as np
import pytest
from opal.replicated.power import calibrated_mean,procedure_power,probability_interval


def pilot():
    rng=np.random.default_rng(42)
    components=list('ABCDEF')
    profiles=rng.normal(size=(len(components),3))
    pairs=[(a,b) for a in range(6) for b in range(a+1,6)]
    refs=[];ys=[];ids=[];cultures=[]
    import json
    for c in range(4):
        control=rng.normal(0,.05,3)
        measured=profiles+rng.normal(0,.1,profiles.shape)-control
        for a,b in pairs:
            refs.append([measured[a],measured[b]])
            ys.append(profiles[a]+profiles[b]+.1*profiles[a]*profiles[b]+rng.normal(0,.1,3))
            ids.append(json.dumps([components[a],components[b]],separators=(',',':')))
            cultures.append(str(c))
    return {'singles':np.asarray(refs),'responses':np.asarray(ys),'pair_ids':np.array(ids),
            'culture_ids':np.array(cultures),'partitions':np.array(['train']*len(ids)),
            'genes':np.array(['g1','g2','g3'])}


def test_local_alternative_has_exact_expected_gain_and_reports_infeasibility():
    r=np.zeros((4,3));c=np.ones((4,3));v=.2
    for gain in (0,.02,.05,.075,.10):
        mean=calibrated_mean(r,c,v,gain)
        actual=(np.mean((r-mean)**2)-np.mean((c-mean)**2))/(np.mean((r-mean)**2)+v)
        assert actual==pytest.approx(gain,abs=1e-12)
    assert calibrated_mean(r,r,v,.05) is None
    assert calibrated_mean(r,c,10000,.05) is None


def test_procedure_simulation_repeats_selection_without_qualifying_a_screen():
    menu=[{'id':'shared','kind':'shared_offset','alpha':.1,'role':'reference'},
          {'id':'global','kind':'global','alpha':1,'role':'reference'},
          {'id':'pair','kind':'global_plus_pair','alpha':.1,'role':'candidate'}]
    report=procedure_power(pilot(),menu,[{'development_cultures':3,'confirmation_cultures':4}],
                           draws=10,effects=(0,.05),variance_multipliers=(1,),shuffle_seeds=(1729,))
    assert report['selection_repeated'] and report['shuffle_selection_repeated']
    assert not report['confirmation_design_qualified']
    assert not report['confirmation_outcomes_used_for_selection']
    for row in report['reports']:
        assert sum(row['selections'].values())==10
        assert row['feasible_draws']+row['infeasible_draws']==10
        assert 0<=row['full_gate_power']<=row['positive_detection_power']<=1
        assert row['train_pairs']+row['validation_pairs']+row['held_pairs']==row['pairs']


def test_procedure_simulation_refuses_held_pilot_and_exceeded_budget():
    bank=pilot();bank['partitions'][0]='test'
    with pytest.raises(ValueError,match='development responses only'):
        procedure_power(bank,[],[{'development_cultures':3,'confirmation_cultures':4}],draws=10)
    with pytest.raises(RuntimeError,match='compute budget'):
        procedure_power(pilot(),[],[{'development_cultures':3,'confirmation_cultures':4}],draws=10,maximum_seconds=-1)


def test_monte_carlo_precision_does_not_invent_trials():
    assert probability_interval(0,10)[0]==0
    assert probability_interval(10,10)[1]==1
    assert probability_interval(0,0)==[None,None]
