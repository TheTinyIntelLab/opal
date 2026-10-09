import json
import numpy as np
import pytest
from opal.replicated.sensitivity import deletion_sensitivity


def arrays(cultures=4):
    pairs=np.tile(['["A","B"]','["A","C"]','["B","C"]'],cultures)
    units=np.repeat([str(i) for i in range(cultures)],3)
    truth=np.zeros((len(pairs),2));ref=np.ones_like(truth);cand=ref.copy()
    cand[np.array(['A' in json.loads(p) for p in pairs])]=.5
    return ref,cand,truth,units,pairs


def test_component_deletion_removes_both_positions_and_reveals_concentration():
    report=deletion_sensitivity(*arrays())
    without_a=report['leave_one_component_out']['A']
    assert without_a['pairs']==1 and without_a['gain']==pytest.approx(0)
    assert report['leave_one_component_out']['C']['gain']==pytest.approx(.75)
    assert not report['can_replace_primary'] and not report['refitted']


def test_technical_duplicate_cannot_change_deletion_influence():
    data=arrays();original=deletion_sensitivity(*data)
    repeated=[np.concatenate([a,a[:1]]) for a in data]
    assert deletion_sensitivity(*repeated)==original


def test_insufficient_cultures_and_empty_component_panel_remain_visible():
    report=deletion_sensitivity(*arrays(3))
    assert all(r['interval'] is None and r['cultures']==2 for r in report['leave_one_culture_out'].values())
    ref,cand,truth,units,pairs=arrays()
    keep=pairs=='["A","B"]'
    report=deletion_sensitivity(ref[keep],cand[keep],truth[keep],units[keep],pairs[keep])
    assert report['leave_one_component_out']['A']['pairs']==0
    assert report['leave_one_component_out']['A']['gain'] is None
