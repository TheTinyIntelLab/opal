"""Finite protocol validation and explicit amendment/exposure records."""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np
from .contracts import sha,write_json,identity


def validate_protocol(config, rows=None):
    menu=config.get('menu',[])
    if not menu or len({c.get('id') for c in menu})!=len(menu):
        raise ValueError('Nonempty menu with unique configuration IDs required')
    families={'additive','shared_offset','global','single_sum','component_additive','global_plus_pair','pair_product'}
    roles=set()
    for c in menu:
        if not isinstance(c.get('id'),str) or not c['id'].strip() or c.get('kind') not in families:
            raise ValueError('Unknown or unnamed frozen family')
        if c.get('role') not in {'reference','candidate'}:raise ValueError('Unknown menu role')
        roles.add(c['role'])
        if c['role']=='reference' and c['kind'] in {'global_plus_pair','pair_product'}:
            raise ValueError('Pair-dependent family cannot be the non-product reference')
        if c['role']=='candidate' and c['kind'] not in {'global_plus_pair','pair_product'}:
            raise ValueError('Candidate must test the declared pair-dependent hypothesis')
        if not np.isfinite(c.get('alpha',0)) or c.get('alpha',0)<=0:
            raise ValueError('Finite positive penalty required')
        if not np.isfinite(c.get('bandwidth',1)) or c.get('bandwidth',1)<=0:
            raise ValueError('Finite positive bandwidth required')
    if roles!={'reference','candidate'}:raise ValueError('Candidate and reference roles required')
    for key in ('maximum_seconds','materiality','secondary_panel_fraction'):
        value=config.get(key,.1 if key=='secondary_panel_fraction' else None)
        if value is None or not np.isfinite(value) or value<=0:
            raise ValueError('Invalid frozen protocol field: '+key)
    if config['materiality']>=1 or config.get('secondary_panel_fraction',.1)>1:
        raise ValueError('Invalid materiality/panel fraction')
    memory=config.get('maximum_memory_bytes')
    if memory is not None and (not isinstance(memory,int) or isinstance(memory,bool) or memory<=0):
        raise ValueError('Invalid frozen memory budget')
    seeds=config.get('shuffle_seeds',[])
    if not seeds or len(set(seeds))!=len(seeds) or any(not isinstance(s,int) or s<0 for s in seeds):
        raise ValueError('Distinct nonnegative integer shuffle seeds required')
    scientific=config.get('data_role') in {'registered_secondary_analysis','prospective_confirmation'}
    if scientific:
        design=config.get('statistical_design',{})
        required={'pilot_report_sha256','interval_coverage_report_sha256','review_source',
                  'confirmation_cultures','stopping_rule','estimand'}
        if required-design.keys() or not all(design[k] for k in required):
            raise ValueError('Scientific protocol requires reviewed empirical power and interval design')
        for key in ('pilot_report_sha256','interval_coverage_report_sha256'):
            value=design[key]
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid statistical evidence hash')
        if design['confirmation_cultures']!=len(config['confirmation_cultures']):
            raise ValueError('Frozen sample size differs from the confirmation allocation')
        if design['stopping_rule']!='fixed_sample_size_no_outcome_peeking':
            raise ValueError('Unsupported stopping rule; register an explicit amended implementation')
        if design['estimand']!='fixed_pair_panel_new_cultures':
            raise ValueError('This interval implementation does not support new-target population inference')
        if rows:
            pairs=[r for r in rows if r['condition_kind']=='pair']
            for key in ('study_id','source_id','state_id','assay_id','treatment_time'):
                if len({str(r[key]) for r in pairs})!=1:
                    raise ValueError('Scientific fixed-state contract mixes '+key)
            assignments={}
            for row in pairs:
                for component,dose,unit,reagent in json.loads(identity(row,config.get('aliases'),True)):
                    assignments.setdefault(component,set()).add((dose,unit,reagent))
            if any(len(values)!=1 for values in assignments.values()):
                raise ValueError('Scientific fixed-dose contract mixes component dose/reagent assignments')
    return {'valid':True,'scientific_role':scientific,'configurations':len(menu),
            'external_registration_verified':False,'empirical_evidence_content_reviewed':False}


def statistical_evidence(config,root):
    """Bind real reviewed statistical records, not hashes with no backing files."""
    if config.get('data_role') not in {'registered_secondary_analysis','prospective_confirmation'}:return {}
    design=config['statistical_design'];paths={}
    for name in ('pilot_report','interval_coverage_report'):
        if not design.get(name):raise ValueError('Missing reviewed statistical evidence file: '+name)
        path=(Path(root)/design[name]).resolve()
        if sha(path)!=design[name+'_sha256']:raise ValueError('Changed reviewed statistical evidence')
        record=json.loads(path.read_text(encoding='utf-8'))
        if (record.get('confirmation_design_qualified') is not True or
            record.get('independent_cultures_verified') is not True or
            record.get('qualification_status')!='accepted_for_fixed_panel'):
            raise ValueError('Statistical evidence does not qualify independent-culture confirmation')
        paths[str(path)]=sha(path)
    return paths


def amendment(original, output, reason, outcomes_seen, affected_inputs, replacement=None):
    """Record an amendment without editing the original or granting fresh status."""
    if not isinstance(outcomes_seen,bool) or not reason.strip() or not affected_inputs:
        raise ValueError('Reason, affected inputs and explicit outcome exposure required')
    if Path(output).exists():raise FileExistsError('Preserve prior amendment records')
    record={'schema':'opal-amendment-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'original_name':Path(original).name,'original_sha256':sha(original),
        'reason':reason,'outcomes_seen':outcomes_seen,'affected_inputs':list(affected_inputs),
        'replacement_sha256':sha(replacement) if replacement else None,
        'classification':'retrospective_exposed_amendment' if outcomes_seen else 'pre_outcome_amendment',
        'fresh_confirmation_required_for_changed_claim':outcomes_seen,
        'public_timestamp_claimed':False,'original_preserved':True}
    write_json(output,record)
    return record
