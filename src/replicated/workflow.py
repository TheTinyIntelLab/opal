"""Immutable prediction freeze and separately invoked custodian evaluation."""
import json
from pathlib import Path
from datetime import datetime, timezone
import time
import sys
import numpy as np
from .contracts import sha, write_json, validate_access, identity, validate_rows, qualify
from .models import select, fit_candidate, predict_candidate, shuffled_profiles, balanced_weights
from .evaluation import score, decomposition
from .protocol import validate_protocol,statistical_evidence


def load_bank(path, outcomes=False):
    with np.load(path,allow_pickle=False) as f:bank={k:f[k] for k in f.files}
    required={'singles','pair_ids','culture_ids','row_ids','genes','scale','units'}
    if outcomes:required|={'responses','partitions'}
    if required-bank.keys():raise ValueError('Incomplete bank schema')
    allowed=required | ({'partitions'} if outcomes else set())
    if bank.keys()-allowed:raise ValueError('Unexpected bank fields; outcomes cannot hide under other names')
    if not outcomes and ('responses' in bank or 'doubles' in bank or 'outcomes' in bank):
        raise ValueError('Query bank contains forbidden pair outcomes')
    n=len(bank['row_ids']);g=len(bank['genes'])
    if len(set(bank['row_ids']))!=n or len(set(bank['genes']))!=g or not n or not g:
        raise ValueError('Duplicate/empty row or gene identities')
    for key in ('row_ids','genes','culture_ids','pair_ids'):
        if bank[key].dtype.kind not in {'U','S'} or not all(str(v).strip() for v in bank[key]):
            raise ValueError('Invalid text identities: '+key)
    for key in ('scale','units'):
        if bank[key].shape!=() or bank[key].dtype.kind not in {'U','S'} or not str(bank[key]).strip():
            raise ValueError('Invalid scalar scale/units: '+key)
    for pair in bank['pair_ids']:
        try:components=json.loads(str(pair))
        except (ValueError,TypeError) as exc:raise ValueError('Invalid canonical pair identity') from exc
        if (not isinstance(components,list) or len(components)!=2 or
            any(not isinstance(c,str) or not c.strip() for c in components) or
            len(set(components))!=2 or str(pair)!=json.dumps(sorted(components),separators=(',',':'))):
            raise ValueError('Invalid canonical pair identity')
    if bank['singles'].shape!=(n,2,g) or not np.isfinite(bank['singles']).all():raise ValueError('Invalid reference arrays')
    for key in ('pair_ids','culture_ids'):
        if bank[key].shape!=(n,):raise ValueError('Row identity shape mismatch')
    if outcomes:
        if bank['responses'].shape!=(n,g) or not np.isfinite(bank['responses']).all() or bank['partitions'].shape!=(n,):
            raise ValueError('Invalid development response schema')
    return bank


def aligned_schema(left,right):
    for key in ('genes','scale','units'):
        if not np.array_equal(left[key],right[key]):raise ValueError('Changed gene order/scale/units: '+key)


def metadata_bank_check(bank, rows, aliases=None):
    lookup={r['row_id']:r for r in rows}
    for row_id,pair,culture in zip(bank['row_ids'],bank['pair_ids'],bank['culture_ids']):
        if str(row_id) not in lookup:raise ValueError('Bank row absent from metadata')
        row=lookup[str(row_id)]
        if row['condition_kind']!='pair' or identity(row,aliases)!=str(pair) or row['culture_id']!=str(culture):
            raise ValueError('Bank identities disagree with metadata')
    if 'partitions' in bank:
        if any(lookup[str(i)]['partition']!=str(p) for i,p in zip(bank['row_ids'],bank['partitions'])):
            raise ValueError('Bank partitions disagree with metadata')


def freeze(development, query, metadata, protocol, output):
    output=Path(output)
    if output.exists():raise FileExistsError('Prediction freeze already exists; preserve it')
    started=time.monotonic()
    dev=load_bank(development,True);q=load_bank(query)
    aligned_schema(dev,q)
    rows=json.loads(Path(metadata).read_text(encoding='utf-8'))
    config=json.loads(Path(protocol).read_text(encoding='utf-8'))
    aliases=config.get('aliases',{})
    validate_rows(rows);metadata_bank_check(dev,rows,aliases);metadata_bank_check(q,rows,aliases)
    qualification=qualify(rows,aliases)
    scientific=config.get('data_role') in {'registered_secondary_analysis','prospective_confirmation'}
    processing=config.get('data_role') in {'processing_diagnostic','registered_processing_secondary'}
    if scientific:
        if not qualification['independent_cultures_verified']:
            raise ValueError('Scientific freeze requires verified independent-initiation metadata')
        required_pairs=set(dev['row_ids'])|set(q['row_ids'])
        if required_pairs-set(qualification['matches']):raise ValueError('Scientific freeze lacks exact matched reference metadata')
        if config.get('status')!='registered' or not config.get('registration_url'):
            raise ValueError('Scientific freeze requires an actual registered protocol record')
    if processing:
        required_pairs=set(dev['row_ids'])|set(q['row_ids'])
        if required_pairs-set(qualification['matches']):
            raise ValueError('Processing diagnostic lacks exact matched reference metadata')
        if config.get('data_role')=='registered_processing_secondary' and (config.get('status')!='registered' or not config.get('registration_url')):
            raise ValueError('Registered processing analysis requires a registered protocol record')
    validate_protocol(config,rows)
    statistical_inputs=statistical_evidence(config,Path(protocol).resolve().parent)
    metadata_lookup={r['row_id']:r for r in rows}
    if any(metadata_lookup[str(i)]['partition']!='test' for i in q['row_ids']):raise ValueError('Query identities must have test membership')
    if set(dev['pair_ids'])&set(q['pair_ids']):raise ValueError('Globally held pair entered development')
    if not processing and set(dev['culture_ids'])&set(q['culture_ids']):raise ValueError('Confirmation culture entered development')
    if set(q['culture_ids'])!=set(config['confirmation_cultures']):raise ValueError('Query culture contract mismatch')
    access={'preprocessing':dev['row_ids'].tolist(),
            'gene_selection':dev['row_ids'].tolist(),
            'fitting':dev['row_ids'][dev['partitions']=='train'].tolist(),
            'selection':dev['row_ids'][dev['partitions']=='validation'].tolist(),
            'refit':dev['row_ids'][np.isin(dev['partitions'],['train','validation'])].tolist()}
    validate_access(rows,access,q['pair_ids'],[] if processing else q['culture_ids'],aliases)
    selection=select(dev,config['menu'])
    refit=np.isin(dev['partitions'],['train','validation'])
    predictions={'additive':q['singles'].sum(1)}
    # Secondary endpoints are frozen from development responses only.
    signal=balanced_weights(dev['pair_ids'],dev['culture_ids'])@dev['responses']**2
    weights=np.maximum(signal,1e-12);weights=weights/weights.sum()
    panel=np.zeros(len(signal),bool)
    panel[np.argsort(-signal,kind='stable')[:max(1,int(np.ceil(len(signal)*config.get('secondary_panel_fraction',.1))))]]=True
    models={}
    for role in ('reference','candidate'):
        cfg=selection[role];model=fit_candidate(cfg,dev,refit)
        predictions[role]=predict_candidate(cfg,model,q)
        models[role]=model
    for kind in sorted({r['config']['kind'] for r in selection['comparisons']}):
        chosen=min((r for r in selection['comparisons'] if r['config']['kind']==kind),
                   key=lambda r:(r['validation_mse'],r['config']['id']))['config']
        model=fit_candidate(chosen,dev,refit)
        label='family_'+kind
        predictions[label]=predict_candidate(chosen,model,q)
        models[label]=model
    if 'family_component_additive' in predictions:
        placebo={k:v.copy() for k,v in dev.items()}
        order=np.flatnonzero(refit)
        placebo['pair_ids'][order]=placebo['pair_ids'][np.random.default_rng(config['shuffle_seeds'][0]).permutation(order)]
        cfg=min((r for r in selection['comparisons'] if r['config']['kind']=='component_additive'),
                key=lambda r:(r['validation_mse'],r['config']['id']))['config']
        predictions['incidence_placebo']=predict_candidate(cfg,fit_candidate(cfg,placebo,refit),q)
    shuffles=[]
    for seed in config['shuffle_seeds']:
        shuffled=shuffled_profiles(dev,seed)
        # Use the same seed/map policy on allowed query-culture references.
        sq=shuffled_profiles(q,seed)
        ss=select(shuffled,config['menu'])
        model=fit_candidate(ss['candidate'],shuffled,refit)
        predictions['shuffle_'+str(seed)]=predict_candidate(ss['candidate'],model,sq)
        shuffles.append({'seed':seed,'selection':ss})
    elapsed=time.monotonic()-started
    if elapsed>config['maximum_seconds']:raise RuntimeError('Frozen compute budget exceeded')
    from .resources import memory_evidence
    memory=memory_evidence()
    maximum_memory=config.get('maximum_memory_bytes')
    if scientific and maximum_memory and memory['peak_memory_bytes'] is None:
        raise RuntimeError('Scientific memory budget requires an available peak measurement')
    if maximum_memory and memory['peak_memory_bytes'] is not None and memory['peak_memory_bytes']>maximum_memory:
        raise RuntimeError('Frozen memory budget exceeded')
    output.mkdir(parents=True)
    np.savez_compressed(output/'predictions.npz',**predictions,
                        secondary_gene_weights=weights,secondary_gene_panel=panel,
                        **{k:q[k] for k in ('row_ids','pair_ids','culture_ids','genes','scale','units')})
    # Save fitted state for restoration without accessing response banks.
    state={}
    descriptions={}
    for role,model in models.items():
        if model is None:descriptions[role]={'kind':'additive'};continue
        descriptions[role]={'class':type(model).__name__}
        for key,value in vars(model).items():
            if isinstance(value,np.ndarray):state[role+'__'+key]=value
            else:descriptions[role][key]=value
    np.savez_compressed(output/'fitted.npz',**state)
    write_json(output/'fitted.json',descriptions)
    write_json(output/'selection.json',{'aligned':selection,'shuffles':shuffles})
    query_refs=sorted({ref for row_id in q['row_ids']
                       for ids in qualification['matches'].get(str(row_id),{}).values() for ref in ids})
    validate_access(rows,{'prediction':query_refs},q['pair_ids'],q['culture_ids'],aliases)
    write_json(output/'access.json',{'stages':access,'query_reference_rows':query_refs,
                                   'query_feature_rows':q['row_ids'].tolist(),
                                   'reference_lineage_available':bool(query_refs),
                                   'query_pair_outcomes_accessed':False})
    manifest={'schema':'opal-replicated-freeze-v1','created_utc':datetime.now(timezone.utc).isoformat(),
              'inputs':{**{str(Path(p).resolve()):sha(p) for p in (development,query,metadata,protocol)},**statistical_inputs},
              'protocol':config,'files':{p.name:sha(p) for p in output.iterdir()},
              'query_rows':q['row_ids'].tolist(),'genes':q['genes'].tolist(),
              'query_input_sha256':sha(query),
              'scale':str(q['scale']),'units':str(q['units']),'elapsed_seconds':elapsed,
              'fitted_bytes':sum(p.stat().st_size for p in output.iterdir()),
              **memory,'python':sys.version,'numpy':np.__version__,
              'code_sha256':{name:sha(Path(__file__).with_name(name)) for name in
                             ('workflow.py','models.py','contracts.py','evaluation.py','data.py','protocol.py','diagnostics.py','resources.py','sensitivity.py')},
              'secure_custody_claimed':False,'public_registration_established':False,
              'registration_record_supplied':bool(config.get('registration_url'))}
    manifest['processing_repeat_scope']=processing
    manifest['independent_culture_metadata_verified']=qualification['independent_cultures_verified']
    write_json(output/'freeze_manifest.json',manifest)
    return manifest


def restore_predictions(frozen, query):
    from .models import ProfileKernel, ComponentAdditive
    frozen=Path(frozen);q=load_bank(query)
    manifest=json.loads((frozen/'freeze_manifest.json').read_text())
    for name,digest in manifest['code_sha256'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Evaluator code differs from the frozen implementation')
    for name,digest in manifest['files'].items():
        if sha(frozen/name)!=digest:raise ValueError('Changed frozen file')
    if q['genes'].tolist()!=manifest['genes'] or str(q['scale'])!=manifest['scale'] or str(q['units'])!=manifest['units'] or q['row_ids'].tolist()!=manifest['query_rows']:
        raise ValueError('Unsupported restoration query schema')
    with np.load(frozen/'predictions.npz',allow_pickle=False) as f:
        if any(not np.array_equal(q[k],f[k]) for k in ('pair_ids','culture_ids')):
            raise ValueError('Unsupported restoration query identities')
    if sha(query)!=manifest['query_input_sha256']:
        raise ValueError('Changed frozen query references')
    descriptions=json.loads((frozen/'fitted.json').read_text())
    with np.load(frozen/'fitted.npz',allow_pickle=False) as f:state={k:f[k] for k in f.files}
    restored={}
    for role,d in descriptions.items():
        if d.get('kind')=='additive':restored[role]=q['singles'].sum(1);continue
        model=ProfileKernel(d['config']) if d['class']=='ProfileKernel' else ComponentAdditive(d['alpha'])
        for k,v in d.items():
            if k!='class':setattr(model,k,v)
        for k,v in state.items():
            if k.startswith(role+'__'):setattr(model,k.split('__',1)[1],v)
        restored[role]=predict_candidate({'kind':'component_additive' if d['class']=='ComponentAdditive' else d['config']['kind']},model,q)
    return restored


def evaluate(frozen, outcomes, output, evidence=None):
    """Custodian-only endpoint: one immutable record per freeze."""
    frozen=Path(frozen);output=Path(output)
    if output.exists() or (frozen/'evaluation_receipt.json').exists():raise FileExistsError('Freeze has already been scored')
    manifest=json.loads((frozen/'freeze_manifest.json').read_text())
    for name,digest in manifest['code_sha256'].items():
        if sha(Path(__file__).with_name(name))!=digest:raise ValueError('Evaluator code differs from the frozen implementation')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or sha(frozen/name)!=digest:raise ValueError('Frozen file hash mismatch')
    with np.load(frozen/'predictions.npz',allow_pickle=False) as f:pred={k:f[k] for k in f.files}
    with np.load(outcomes,allow_pickle=False) as f:truth={k:f[k] for k in f.files}
    for key in ('row_ids','pair_ids','culture_ids','genes','scale','units'):
        if key not in truth or not np.array_equal(pred[key],truth[key]):raise ValueError('Evaluator schema mismatch: '+key)
    if truth['responses'].shape!=(len(pred['row_ids']),len(pred['genes'])) or not np.isfinite(truth['responses']).all():
        raise ValueError('Invalid sealed outcomes')
    cfg=manifest['protocol'];ev={}
    if evidence:
        ev=json.loads(Path(evidence).read_text())
        if ev.get('freeze_sha256')!=sha(frozen/'freeze_manifest.json'):raise ValueError('Evidence belongs to a different freeze')
    is_scientific=cfg.get('data_role') in {'registered_secondary_analysis','prospective_confirmation'}
    qualified=is_scientific and ev.get('independent_cultures_verified') is True
    qualified=qualified and cfg.get('status')=='registered' and bool(cfg.get('registration_url'))
    qualified=qualified and manifest.get('independent_culture_metadata_verified') is True
    # A URL alone does not establish registration timing; the custodian must
    # review the external record and bind that review to this exact freeze.
    registration=ev.get('registration',{})
    registration_verified=False
    if isinstance(registration,dict) and registration.get('verified') is True:
        try:
            registered=datetime.fromisoformat(registration['registered_utc'])
            frozen_time=datetime.fromisoformat(manifest['created_utc'])
            registration_verified=(registered.tzinfo is not None and registered<=frozen_time and
                registration.get('url')==cfg.get('registration_url') and bool(registration.get('verification_source')))
        except (ValueError,TypeError,KeyError):
            pass
    qualified=qualified and registration_verified
    report=score({k:v for k,v in pred.items() if k in {'candidate','reference','additive','incidence_placebo'} or k.startswith(('shuffle_','family_'))},
                 truth['responses'],pred['culture_ids'],pred['pair_ids'],pred['additive'],cfg['materiality'],
                 qualified and ev.get('measurement_pass') is True,qualified and ev.get('attribution_pass') is True)
    if 'reference_additive' in truth:
        independent=truth['reference_additive']
        if independent.shape!=truth['responses'].shape or not np.isfinite(independent).all():
            raise ValueError('Invalid independent evaluator references')
        report['decomposition']=decomposition(pred['additive'],pred['candidate'],independent,truth['responses'],
            balanced_weights(pred['pair_ids'],pred['culture_ids']))
    report.update({'freeze_sha256':sha(frozen/'freeze_manifest.json'),'outcomes_sha256':sha(outcomes),
                   'evidence_sha256':sha(evidence) if evidence else None,
                   'data_role':cfg['data_role'],'scientific_confirmation_claimed':qualified,
                   'public_registration_review_verified':registration_verified,
                   'access_log':[{'stage':'evaluator','rows':pred['row_ids'].tolist(),'utc':datetime.now(timezone.utc).isoformat()}],
                   'missingness_rule':'balanced complete culture-by-pair support; reject incomplete blocks',
                   'secondary_endpoints_cannot_replace_primary':True})
    from .evaluation import cube
    panel=pred['secondary_gene_panel'];weights=pred['secondary_gene_weights']
    secondary={}
    for role in ('candidate','reference','additive'):
        error,_,_=cube((pred[role]-truth['responses'])**2,pred['culture_ids'],pred['pair_ids'])
        correction=pred[role]-pred['additive'];residual=truth['responses']-pred['additive']
        denominator=np.linalg.norm(correction,axis=1)*np.linalg.norm(residual,axis=1)
        defined=denominator>1e-12
        alignment=np.full(len(defined),np.nan)
        alignment[defined]=np.sum(correction[defined]*residual[defined],axis=1)/denominator[defined]
        alignment_parts,_,_=cube(np.stack([np.nan_to_num(alignment),defined],axis=1),pred['culture_ids'],pred['pair_ids'])
        alignment_blocks=np.full(alignment_parts.shape[:2],np.nan)
        np.divide(alignment_parts[:,:,0],alignment_parts[:,:,1],out=alignment_blocks,where=alignment_parts[:,:,1]>0)
        culture_alignment=[float(row[np.isfinite(row)].mean()) for row in alignment_blocks if np.isfinite(row).any()]
        # Undefined zero corrections remain missing rather than becoming zero cosine.
        norm_blocks,_,_=cube((np.linalg.norm(correction,axis=1)-np.linalg.norm(residual,axis=1))**2,pred['culture_ids'],pred['pair_ids'])
        secondary[role]={'strong_response_panel_mse':float(error[:,:,panel].mean()),
            'signal_weighted_mse':float(np.mean(error@weights)),
            'mean_defined_residual_alignment':float(np.mean(culture_alignment)) if culture_alignment else None,
            'defined_alignment_pairs':int(defined.sum()),
            'residual_norm_error':float(norm_blocks.mean())}
    report['secondary_diagnostics']=secondary
    report['secondary_gene_panel']=pred['genes'][panel].tolist()
    report['secondary_role']='diagnostic only; no replacement of the primary endpoint'
    from .sensitivity import deletion_sensitivity
    report['deletion_sensitivity']=deletion_sensitivity(pred['reference'],pred['candidate'],truth['responses'],
                                                       pred['culture_ids'],pred['pair_ids'])
    if manifest.get('processing_repeat_scope'):
        report['inference_scope']='within-processing-repeat descriptive comparison; no independent-culture inference'
        for interval in list(report['paired_intervals'].values())+[report['aligned_vs_shuffled']]:
            if interval:
                interval['method']='descriptive processing-repeat Student t; biological independence unverified'
                interval['scope']=report['inference_scope']
        report['deletion_sensitivity']['scope']=report['inference_scope']+'; exploratory deletion diagnostics only'
        for group in ('leave_one_culture_out','leave_one_component_out'):
            for result in report['deletion_sensitivity'][group].values():
                if result['interval']:
                    result['interval']['method']='descriptive processing-repeat Student t; biological independence unverified'
                    result['interval']['scope']=report['inference_scope']
    from .diagnostics import interpret
    component=report['paired_intervals'].get('family_component_additive')
    interpretation={'independent_cultures_verified':bool(manifest.get('independent_culture_metadata_verified') and ev.get('independent_cultures_verified')),
        'measurement_pass':report['measurement_pass'],'primary_claim_pass':report['primary_claim_pass'],
        'pair_product_pass':report['primary_claim_pass'],
        'component_additive_pass':bool(component and component['gain']>=cfg['materiality'] and component['difference_interval'][0]>0),
        'gain_upper':report['paired_intervals']['candidate']['gain_interval_scaled_by_observed_reference'][1],
        'aligned_profile_information':bool(report['aligned_vs_shuffled'] and report['aligned_vs_shuffled']['difference_interval'][0]>0)}
    for key in ('release_integrity','precision_adequate','gain_disappears_with_independent_references',
                'state_transfer_pass','registered_secondary_pass'):
        if key in ev:interpretation[key]=ev[key]
    report['section18_interpretation']=interpret(interpretation,cfg['materiality'])
    # Exclusive creation prevents concurrent invocations from publishing two scores.
    # A failed write leaves the attempted score receipt for explicit review.
    with (frozen/'evaluation_receipt.json').open('x',encoding='utf-8') as stream:
        json.dump({'status':'writing_report','output':str(output.resolve()),
                   'outcomes_sha256':sha(outcomes)},stream)
    output.mkdir(parents=True)
    write_json(output/'results.json',report)
    write_json(output/'decision.json',{'decision':report['decision'],'primary_claim_pass':report['primary_claim_pass'],
        'section18_interpretation':report['section18_interpretation'],
        'next_action':{'advance_fixed_state':'Replicate, then register separate crossed-state confirmation',
                       'measurement_unqualified':'Qualify independent culture provenance and incremental reproducibility',
                       'attribution_unqualified':'Complete independent-reference attribution diagnostics',
                       'alignment_unestablished':'Do not claim aligned-profile predictive information',
                       'precise_failure':'Report the useful-gain exclusion within the registered panel',
                       'inconclusive':'Report precision limitation; follow registered stopping policy'}[report['decision']]})
    write_json(frozen/'evaluation_receipt.json',{'status':'complete','report_sha256':sha(output/'results.json'),'output':str(output.resolve()),
                                              'outcomes_sha256':sha(outcomes)})
    return report
