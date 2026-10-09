import pytest
from opal.replicated.planning import quote_design


def design():
    return {'interventions':12,'train_pairs':34,'validation_pairs':12,'test_pairs':20,
        'development_cultures':4,'confirmation_cultures':8,'control_measurements':2,'single_measurements':2}


def test_unknown_prices_remain_unknown_and_technical_repeats_do_not_add_cultures():
    report=quote_design(design(),[{'name':'Assay','quantity_basis':'total_measurements','unit_price':None}])
    assert report['quantities']['total_measurements']==656
    assert report['quantities']['culture_initiations']==12
    assert report['complete_quoted_cost'] is None and not report['scientific_sample_size_qualified']
    more=design();more['single_measurements']=3
    assert quote_design(more,[])['quantities']['culture_initiations']==12


def test_costs_require_quotes_and_cannot_silently_mix_currency():
    item={'name':'Assay','quantity_basis':'total_measurements','unit_price':1,
          'currency':'X','source':'supplier quote accession','quote_date':'2026-10-08'}
    assert quote_design(design(),[item])['complete_quoted_cost']==656
    other=dict(item,currency='Y')
    with pytest.raises(ValueError,match='Mixed currencies'):quote_design(design(),[item,other])
    with pytest.raises(ValueError,match='currency, source'):quote_design(design(),[dict(item,source=None)])
