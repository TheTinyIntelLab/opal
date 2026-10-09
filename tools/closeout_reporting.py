"""Reporting-only canonical claims and release inventory, never expression banks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.release_policy import allowed_source, allowed_evidence, public_name
from opal.replicated.contracts import write_json, sha


def claims():
    rows=[]
    def add(path,run,role,pop,candidate,comparator,metric,value,rounding=6,limit='No independent biological replication or universal architecture ranking'):
        rows.append({'claim_id':f'{run}/{pop}/{candidate}/{metric}','source_path':path,'source_sha256':sha(ROOT/path),
            'run_id':run,'run_role':role,'population':pop,'scale':'source-declared effect scale',
            'gene_panel':'source-frozen eligible output universe','weighting':'equal query occurrences under source protocol',
            'candidate':candidate,'comparator':comparator,'metric':metric,'value':float(value),'rounding':rounding,
            'uncertainty_scope':'conditional profile diagnostic','claim_limit':limit})
    for path in sorted(ROOT.glob('results/**/comparisons.csv')):
        relative=path.relative_to(ROOT).as_posix()
        if 'report-v1' in relative and ('stage1' in relative or 'joint' in relative):continue
        role='recheck_sensitivity' if '/recheck-' in relative else 'original_development'
        table=list(csv.DictReader(path.open(encoding='utf-8')))
        def population(r):
            return '/'.join(str(r[k]) for k in ('study','source','panel','subset','held_dose_nM','shift') if r.get(k)) or 'source-declared fixed population'
        refs={population(r):r for r in table if r.get('model')=='selected_simple'}
        for r in table:
            if population(r) not in refs and r.get('relative_gain') and float(r['relative_gain'])==0:
                refs[population(r)]=r
        for i,r in enumerate(table):
            if not r.get('mse') or not r.get('model'):continue
            pop=population(r)
            comparator=refs[pop]['model'] if pop in refs else 'source-declared comparator'
            if r['model']=='baseline_only_mean_plus_offset':comparator='not_comparable_information_contract'
            add(relative,relative.removesuffix('/comparisons.csv'),role,pop,r['model'],comparator,'mse',r['mse'])
            if r.get('relative_gain') and comparator!='not_comparable_information_contract':
                if pop in refs:
                    expected=1-float(r['mse'])/float(refs[pop]['mse'])
                    if abs(expected-float(r['relative_gain']))>1e-9:raise ValueError('Invalid recorded ratio: '+relative)
                add(relative,relative.removesuffix('/comparisons.csv'),role,pop,r['model'],comparator,'gain_percent',100*float(r['relative_gain']),2)
    for name in ('norman-log2','wessels-log2'):
        path=f'results/runs/{name}-primary/test_results.json';result=json.loads((ROOT/path).read_text())
        selected=json.loads((ROOT/f'results/runs/{name}-primary/selection.json').read_text())
        additive=next(r['config']['id'] for r in selected['comparisons'] if r['config']['family']=='additive')
        for label,identifier in [('additive',additive),('selected_neural',result['selected_id']),('selected_simple',result['simpler_id'])]:
            value=result['models'][identifier]['mse']
            add(path,name+'-primary','historical_confirmation',name,identifier,additive,'mse',value)
            if label=='selected_neural':add(path,name+'-primary','historical_confirmation',name,identifier,additive,'gain_percent',100*(1-value/result['models'][additive]['mse']),2)
    path='results/next/dspin-confirmation-v1/results.json';r=json.loads((ROOT/path).read_text());ref=r['development_selected_simple_reference']
    for label in ('selected',ref):add(path,'dspin-confirmation-v1','independent_source_evaluation','held lineages across batches',label,ref,'mse',r['mse'][label])
    # Reconcile the remaining exact resources and count-scale manuscript value.
    path='results/runs/norman-count-sensitivity/test_results.json'
    count=json.loads((ROOT/path).read_text())
    configs=json.loads((ROOT/'results/runs/norman-count-sensitivity/selection.json').read_text())
    additive=next(r['config']['id'] for r in configs['comparisons'] if r['config']['family']=='additive')
    add(path,'norman-count-sensitivity','historical_scale_sensitivity','Norman count test',additive,additive,'mse',count['models'][additive]['mse'],8)
    path='results/panel-scale-20261007/run-v1/results.json';panel=json.loads((ROOT/path).read_text())
    add(path,'panel-scale-resources','original_development','all scale/panel diagnostics','runner','not applicable','seconds',panel['elapsed_seconds'],2)
    add(path,'panel-scale-resources','original_development','all scale/panel diagnostics','runner','not applicable','completion_MB',panel['completion_rss_bytes']/1e6,1)
    executions=[f'results/panel-scale-20261007/run-v1/{name}-count/execution.json' for name in ('norman','wessels')]
    total=sum(json.loads((ROOT/p).read_text())['elapsed_seconds'] for p in executions)
    add(executions[0],'panel-count-resources','original_development','count fitting','runner','not applicable','seconds',total,2)
    rows[-1]['derivation']='sum elapsed_seconds from the two original count executions'
    rows[-1]['source_dependencies']={p:sha(ROOT/p) for p in executions}
    if len({r['claim_id'] for r in rows})!=len(rows):raise ValueError('Duplicate claim IDs')
    return rows


def render_stage1():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records={}
    cycle=ROOT/'results/stage1-20261007'
    for part,pattern in [('run-v1','*/*/results.json'),('main-v1','*/main/results.json')]:
        for p in sorted((cycle/part).glob(pattern)):
            records[p.parent.parent.name+'/'+p.parent.name]=(p,json.loads(p.read_text()))
    if len(records)!=7:raise ValueError('Missing original frozen study records')
    target=ROOT/'paper/evidence/stage1_results.tex'
    text=target.read_text(encoding='utf-8')
    for name,(_,r) in records.items():
        for label,score in r['models'].items():
            display='Selected simple' if label=='selected_simple' else label.replace('_',r'\_')
            prefix=name.replace('_',r'\_')+' & '+display+' & '
            replacement=prefix+f"{score['mse']:.6f} & {100*score.get('relative_gain',0):.2f}"+r'\\'
            text=re.sub(re.escape(prefix)+r'.*?(?=\n)',lambda _:replacement,text)
    text=text.replace('Registered Stage 1 successor development','Locally prespecified Stage 1 development')
    text=text.replace('The local preregistration and evidence archive','The local prespecification and evidence archive')
    text=text.replace('The dashed line is the registered development materiality screen.','The dashed line is the locally prespecified development materiality screen.')
    text=text.replace('Full paired uncertainty and every control are in the saved comparison CSV.',
        'Original frozen runs supply this table and its figures; the changed-environment recheck remains a named sensitivity. Full paired uncertainty and every control are in the saved comparison CSV.')
    target.write_text(text,encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','svg.hashsalt':'opal-closeout-v1','pdf.fonttype':42})
    fig,axes=plt.subplots(3,2,figsize=(12,11),constrained_layout=True)
    for ax,(name,(_,r)) in zip(axes.flat,[(k,v) for k,v in records.items() if not k.endswith('/main')]):
        labels=[k for k in ['context','identity','descriptor','pair_product','mixed','neural_shared','neural_pair'] if k in r['models']]
        vals=[100*r['models'][k]['relative_gain'] for k in labels]
        ax.barh(labels,vals,color=['#397c82' if v>=5 else '#87919a' for v in vals]);ax.axvline(5,color='#b46b49',ls='--');ax.axvline(0,color='black',lw=.5)
        ax.set_title(name);ax.set_xlabel('Relative MSE improvement over selected simple (%)')
    axes.flat[-1].axis('off');axes.flat[-1].text(0,.7,'Original exposed development\nTechnical units do not establish\nindependent culture replication.')
    fig.savefig(ROOT/'images/stage1_results.svg');fig.savefig(ROOT/'paper/evidence/stage1_results.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    for ax,(name,(_,r)) in zip(axes,[(k,v) for k,v in records.items() if k.endswith('/main')]):
        labels=['mean','selected_simple','learned','shuffled_control']
        ax.bar(labels,[r['models'][k]['mse'] for k in labels]);ax.set_title(name);ax.tick_params(axis='x',rotation=20);ax.set_ylabel('Baseline-only single MSE')
    fig.savefig(ROOT/'images/stage1_main.svg');fig.savefig(ROOT/'paper/evidence/stage1_main.pdf');plt.close(fig)
    for name in ('stage1_main.svg','stage1_results.svg'):
        path=ROOT/'images'/name
        path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    write_json(ROOT/'paper/evidence/stage1_manifest.json',{'run_policy':'original_canonical_recheck_sensitivity',
        'sources':{p.relative_to(ROOT).as_posix():sha(p) for p,_ in records.values()},
        'figure_sha256':{n:sha(ROOT/n) for n in ['images/stage1_results.svg','images/stage1_main.svg','paper/evidence/stage1_results.pdf','paper/evidence/stage1_main.pdf']}})


def render_resource_prose(record):
    resources={r['metric']:r['value'] for r in record if r['run_id']=='panel-scale-resources'}
    fitting=next(r['value'] for r in record if r['run_id']=='panel-count-resources')
    path=ROOT/'paper/evidence/panel_scale_results.tex'
    text=path.read_text()
    sentence=f"Count fitting takes {fitting:.2f} elapsed seconds; the complete fitting and diagnostic runner takes {resources['seconds']:.2f} seconds on one CPU thread, with {resources['completion_MB']:.1f} MB completion RSS, not a measured peak."
    text=re.sub(r'Count fitting takes .*?not a measured peak\.',sentence,text)
    path.write_text(text,encoding='utf-8')


def check_claims(record):
    for r in record:
        if sha(ROOT/r['source_path'])!=r['source_sha256']:raise ValueError('Changed claim source hash')
        if not all(r[k] for k in ('candidate','comparator','run_role','population')):raise ValueError('Incomplete claim provenance')
        for path,digest in r.get('source_dependencies',{}).items():
            if sha(ROOT/path)!=digest:raise ValueError('Changed derived claim dependency')
    table=(ROOT/'paper/evidence/stage1_results.tex').read_text(encoding='utf-8')
    for r in record:
        if r['source_path']!='results/stage1-20261007/report-v2/comparisons.csv' or r['metric']!='mse':continue
        model=r['candidate'];prefix=r['population'].replace('_',r'\_')+' & '+('Selected simple' if model=='selected_simple' else model.replace('_',r'\_'))+' & '
        if prefix in table and prefix+f"{r['value']:.6f}" not in table:raise ValueError('Public table/run-role mismatch')
    # Ensure cached source-only plots are backed by the exact compact records.
    plots=json.loads((ROOT/'images/publication_data.json').read_text())
    for p,digest in plots['source_sha256'].items():
        if sha(ROOT/p)!=digest:raise ValueError('Changed publication figure input')
    # Public prose carries generated claim macros for all abstract numerical claims.
    macros=(ROOT/'paper/evidence/canonical_claims.tex').read_text()
    expected=principal_values(record)
    for name,value in expected.items():
        if '\\newcommand{\\'+name+'}{'+value+'}' not in macros:raise ValueError('Abstract claim mismatch')
    main=(ROOT/'paper/main.tex').read_text();abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',main,re.S)[1]
    if any('\\'+name not in abstract for name in expected):raise ValueError('Abstract detached from canonical claim macros')
    count=next(r for r in record if r['run_id']=='norman-count-sensitivity')
    if f"versus additive ${count['value']:.8f}$" not in main:raise ValueError('Count-scale manuscript source mismatch')
    resources={r['metric']:r['value'] for r in record if r['run_id']=='panel-scale-resources'}
    fitting=next(r['value'] for r in record if r['run_id']=='panel-count-resources')
    panel=(ROOT/'paper/evidence/panel_scale_results.tex').read_text()
    sentence=f"Count fitting takes {fitting:.2f} elapsed seconds; the complete fitting and diagnostic runner takes {resources['seconds']:.2f} seconds on one CPU thread, with {resources['completion_MB']:.1f} MB completion RSS"
    if sentence not in panel:raise ValueError('Panel resource prose mixes saved executions')
    return {'status':'passed','claims':len(record),'scope':'source hashes, recorded ratios, Stage 1 table, publication figure inputs, abstract macros',
            'full_narrative_audit':'see numerical surface inventory; editorial figures remain human-reviewed'}


def principal_values(record):
    values={}
    for r in record:
        if r['run_id'] in {'norman-log2-primary','wessels-log2-primary'} and r['metric']=='gain_percent':
            values['NormanGain' if r['run_id'].startswith('norman') else 'WesselsGain']=f"{r['value']:.2f}"
        if r['run_id']=='dspin-confirmation-v1':
            values['DspinCandidate' if r['candidate']=='selected' else 'DspinReference']=f"{r['value']:.6f}"
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--render',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.check:
        saved=json.loads((a.output/'canonical_claims.json').read_text())
        print(json.dumps(check_claims(saved),indent=2));return
    a.output.mkdir(parents=True,exist_ok=False)
    record=claims();write_json(a.output/'canonical_claims.json',record)
    if a.render:
        macros='\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in principal_values(record).items())+'\n'
        (ROOT/'paper/evidence/canonical_claims.tex').write_text(macros,encoding='utf-8')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    inventory=[];private=[]
    for name in tracked:
        path=ROOT/name
        if not name or not path.is_file():continue
        if not public_name(name):private.append(name);continue
        if not (allowed_source(name) or allowed_evidence(name)):continue
        role='recheck_sensitivity' if '/recheck-' in name else 'historical_confirmation' if name.startswith('results/runs/') else 'development_or_release_input'
        inventory.append({'path':name,'sha256':sha(path),'run_id':str(Path(name).parent).replace('\\','/'),
                          'evidence_class':role,'publication_eligible':True,'exclusion_reason':None})
    write_json(a.output/'inventory.json',{'commit':head,'historical_baseline':'84de7e36eae20531e815be9de5a4cb2fa9b94ce9',
        'version':'0.1.2','scientific_records_refitted':False,'files':inventory,
        'git_tree_private_paths':private,'git_tree_publication_ready':not private,'dirty_workspace_expected':True})
    surfaces={}
    for path in [ROOT/'README.md',ROOT/'paper/main.tex',ROOT/'docs/models.md',*sorted((ROOT/'paper/evidence').glob('*.tex'))]:
        text=path.read_text(encoding='utf-8')
        surfaces[path.relative_to(ROOT).as_posix()]={'sha256':sha(path),'numbers':sorted(set(re.findall(r'(?<![A-Za-z])\d+\.\d+(?:\\?%)?',text)))}
    write_json(a.output/'numerical_surface_inventory.json',surfaces)
    if a.render:
        render_stage1();render_resource_prose(record)
    write_json(a.output/'consistency.json',check_claims(record))
    print(json.dumps({'output':str(a.output),'claims':len(record),'private_tracked_paths':len(private)},indent=2))


if __name__=='__main__':main()
