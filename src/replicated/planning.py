"""Unpriced experimental quantities and quote arithmetic, without procurement."""
import math


def quote_design(design,items):
    required=('interventions','train_pairs','validation_pairs','test_pairs',
              'development_cultures','confirmation_cultures','control_measurements','single_measurements')
    if any(not isinstance(design.get(k),int) or isinstance(design[k],bool) or design[k]<1 for k in required):
        raise ValueError('Positive integer design quantities required')
    n=design['interventions'];pairs=sum(design[k] for k in ('train_pairs','validation_pairs','test_pairs'))
    if pairs>n*(n-1)//2:raise ValueError('Distinct pair allocation exceeds the possible unordered graph')
    references=design['control_measurements']+n*design['single_measurements']
    development=design['development_cultures']*(references+design['train_pairs']+design['validation_pairs'])
    confirmation=design['confirmation_cultures']*(references+design['test_pairs'])
    quantities={'development_cultures':design['development_cultures'],
        'confirmation_cultures':design['confirmation_cultures'],
        'culture_initiations':design['development_cultures']+design['confirmation_cultures'],
        'development_measurements':development,'confirmation_measurements':confirmation,
        'total_measurements':development+confirmation,'setup':1,'data_delivery':1,'custody':1}
    total=0.;missing=[];priced=[];currencies=set()
    for item in items:
        key=item['quantity_basis']
        if key not in quantities:raise ValueError('Unknown quote quantity basis')
        unit=item.get('unit_price')
        if unit is None:
            missing.append(item['name']);continue
        if not isinstance(unit,(int,float)) or not math.isfinite(unit) or unit<0:
            raise ValueError('Finite nonnegative quoted unit price required')
        if not item.get('currency') or not item.get('source') or not item.get('quote_date'):
            raise ValueError('Priced items need currency, source and quote date')
        currencies.add(item['currency']);subtotal=unit*quantities[key];total+=subtotal
        priced.append({**item,'quantity':quantities[key],'subtotal':subtotal})
    if len(currencies)>1:raise ValueError('Mixed currencies require an explicit external conversion quote')
    return {'schema':'opal-study-quote-v1','design':design,'quantities':quantities,'priced_items':priced,
        'missing_quote_items':missing,'currency':next(iter(currencies),None),
        'known_subtotal':total,'complete_quoted_cost':total if not missing else None,
        'scientific_sample_size_qualified':False,'procurement_performed':False,
        'technical_reference_measurements_increase_biological_units':False,
        'scope':'itemized feasibility scenario; doses, assay layout, empirical power and vendor quote remain required'}
