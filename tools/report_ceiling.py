"""Publish verified joint-noise, refitting and subset-retuned evidence."""
from pathlib import Path
import sys,zipfile,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import read_json,write_json,sha256,environment,fingerprint
DATASETS={'norman-log2':'Norman','wessels-log2':'Wessels original','wessels-expanded-protected-log2':'Wessels protected'}


def verify(folder,name,inputs):
    protocol=read_json(folder/'protocol.json')
    for field,path in [('profiles_sha256',ROOT/'data/prepared'/name/'profiles.npz'),('draws_sha256',ROOT/'data/architecture_resources'/('joint-'+name)/'draws.npz')]:
        if protocol[field]!=sha256(path):raise ValueError('Changed study inputs: '+str(folder))
    if protocol['evaluator_access']:raise ValueError('Unexpected evaluator access')
    with zipfile.ZipFile(folder/'source_snapshot.zip') as archive:
        runner=[n for n in archive.namelist() if n.startswith('tools/')]
        if len(runner)!=1 or hashlib.sha256(archive.read(runner[0])).hexdigest()!=protocol['runner_sha256']:raise ValueError('Archived runner fingerprint mismatch')
        sources={n[len('src/opal/'):]:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if n.startswith('src/opal/') and n.endswith('.py')}
        if protocol['source_hash'] not in {fingerprint(sources),fingerprint({k.replace('/',chr(92)):v for k,v in sources.items()})}:raise ValueError('Archived package fingerprint mismatch')

    for file in ['protocol.json','results.json','source_snapshot.zip']:
        path=folder/file;inputs[str(path.relative_to(ROOT))]=sha256(path)
    return read_json(folder/'results.json')


def main():
    inputs={};rows=[];technical=[];resources={};curves={};risk=[]
    for name,label in DATASETS.items():
        resources[name]=read_json(ROOT/'data/architecture_resources'/('joint-'+name)/'manifest.json')
        for version in [1,2,3]:
            result=verify(ROOT/'results/architecture'/f'joint-uncertainty-v{version}'/name,name,inputs)
            for kind in ['random','partner']:
                metrics=result[kind]['mse_per_occurrence'];baseline=metrics['unweighted']
                for model,mse in metrics.items():rows.append({'dataset':name,'protocol':kind,'version':version,'model':model,'mse':mse,'reduction_percent':100*(1-mse/baseline)})
                for record in result[kind]['records']:
                    if set(record['train']) & set(record['validation']):raise ValueError('Overlapping fitting and scoring pairs')
                    if 'technical_refitting' in record:technical.append({'dataset':name,'protocol':kind,'version':version,'fold':record['fold'],**record['technical_refitting']})
            if result['export_verification']['maximum_absolute_difference']>1e-6:raise ValueError('Export mismatch')
        curves[name]=verify(ROOT/'results/architecture/retuned-curves-v2'/name,name,inputs)
        for kind in ['random','partner']:
            # Canonical full-size control must share its inner folds and selected config.
            for r in curves[name][kind]['records']:
                if r['status']=='complete' and r['fraction']==1:
                    if not np.isclose(r['mse']['frozen_full_size_kernel'],r['mse']['retuned_unweighted'],rtol=1e-10,atol=1e-12):raise ValueError('Full-size controls differ')
            diagnostics=curves[name][kind]['error_radius_diagnostics']
            for r in diagnostics:
                if r['status']=='complete':
                    fit,cal,val=map(set,[r['fit'],r['calibration'],r['outer_validation']])
                    if fit&cal or fit&val or cal&val:raise ValueError('Risk split overlap')
                    if r['selection_uses_calibration_responses']:raise ValueError('Calibration selection leakage')
            for coverage in [.8,.9]:
                successful=[r for r in diagnostics if r['status']=='complete']
                finite=[(r,next(x for x in r['radii'] if x['coverage_requested']==coverage)) for r in successful if next(x for x in r['radii'] if x['coverage_requested']==coverage)['finite_radius']]
                risk.append({'dataset':name,'protocol':kind,'nominal_coverage':coverage,'finite_folds':len(finite),'completed_folds':len(successful),'infeasible_folds':len(diagnostics)-len(successful),
                    'observed_coverage':float(np.average([q['observed_outer_coverage'] for r,q in finite],weights=[len(r['outer_validation']) for r,q in finite])) if finite else None,
                    'mean_radius':float(np.mean([q['radius'] for r,q in finite])) if finite else None,
                    'calibration_size_range':[min(len(r['calibration']) for r in successful),max(len(r['calibration']) for r in successful)] if successful else None})
    report={'status':'complete','report_environment':environment(),'dependency_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'tools/improve_residual_model.py',ROOT/'tools/prepare_architecture.py']},'inputs':inputs,'resources':resources,'covariance_scores':rows,'technical_refitting':technical,'risk_diagnostics':risk,
        'learning_curves':{name:{kind:value['learning_curves'] for kind,value in result.items()} for name,result in curves.items()},
        'scope':'Late exploratory dependent development resamplings. No independent confirmation, calibrated abstention or frontier-quality claim. Fixed-QC technical refitting omits model selection and biological variation.',
        'amendments':['retuned-curves-v1 retained; v2 restores identical full-size inner partitions by canonical pair order'],
        'source_reference':'https://jmlr.org/papers/v9/shafer08a.html'}
    evidence=ROOT/'paper/evidence';evidence.mkdir(exist_ok=True)
    table=[r'\begin{tabular}{llrrr}\toprule',r'Dataset & Split & Diagonal & Correlated & Permuted \\ \midrule']
    prose=['# Joint uncertainty and retuned learning curves','','These are late exploratory development studies; their split seeds and shared biological profiles are dependent. Historical query responses were not read. The correlated and permuted searches have 469 candidates, diagonal has 201, and unweighted has 67. Ranges below are extrema over three seeds, not confidence intervals.','', '| Dataset | Split | Diagonal gain (%) | Correlated gain (%) | Permuted gain (%) |','|---|---|---:|---:|---:|']
    for name,label in DATASETS.items():
        for kind in ['random','partner']:
            values=[]
            for family in ['diagonal','correlated','permuted_covariance']:
                group=[r['reduction_percent'] for r in rows if r['dataset']==name and r['protocol']==kind and r['model']==family]
                values.append(f'{min(group):.2f} to {max(group):.2f}')
            prose.append('| '+' | '.join([label,kind,*values])+' |')
            table.append(' & '.join([label,kind,*values])+r' \\')
    table.append(r'\bottomrule\end{tabular}')
    (evidence/'ceiling_covariance.tex').write_text('\n'.join(table)+'\n')
    risk_table=[r'\begin{tabular}{llrrrr}\toprule',r'Dataset & Split & Nominal & Finite folds & Observed & Mean radius \\ \midrule']
    for r in risk:
        obs='--' if r['observed_coverage'] is None else f"{100*r['observed_coverage']:.1f}"
        radius='--' if r['mean_radius'] is None else f"{r['mean_radius']:.4f}"
        risk_table.append(' & '.join([DATASETS[r['dataset']],r['protocol'],str(int(100*r['nominal_coverage'])),f"{r['finite_folds']}/{r['completed_folds']}",obs,radius])+r' \\')
    risk_table.append(r'\bottomrule\end{tabular}')
    (evidence/'ceiling_risk.tex').write_text('\n'.join(risk_table)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(10,5.7),sharex=True)
    for col,(name,label) in enumerate(DATASETS.items()):
        for row,kind in enumerate(['random','partner']):
            ax=axes[row,col];group=curves[name][kind]['learning_curves'];x=[float(k) for k in group]
            for metric,title in [('additive','Additive'),('frozen_full_size_kernel','Fixed kernel'),('retuned_unweighted','Retuned kernel'),('retuned_diagonal','Retuned diagonal'),('retuned_correlated','Retuned covariance')]:
                ax.plot(x,[group[str(int(f)) if f==int(f) else str(f)]['mse_per_occurrence'][metric] for f in x],marker='o',markersize=3,label=title)
            ax.set_title(label+' / '+kind,fontsize=9);ax.grid(alpha=.2)
            if col==0:ax.set_ylabel('Gene MSE')
            if row==1:ax.set_xlabel('Fraction of outer training pairs')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=8)
    fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(evidence/'ceiling-learning-curves.pdf');plt.close(fig)
    prose.extend(['','Each subset retunes its model using fitting-only inner folds. Three nested subset orders share each outer validation set; full size is fitted once. The v2 full-size fixed and retuned kernel controls use identical pair order and folds. The original v1 used a reordered full-size partition and is retained as exploratory evidence.','','Technical sensitivity refits the kernel and coefficients under each of 16 joint cell-bootstrap draws, preserving shared single/control draws. It conditions on the selected hyperparameters, covariance, QC and library normalizer. Its percentiles measure sensitivity to technical fitting/reference noise against the point validation response; they are not confidence intervals for biological generalization.','','Error radii use a separate fitting/calibration split: calibration responses cannot select hyperparameters or fit the model. The radius is the ceil((n+1)*coverage)-th ordered calibration pair RMSE; unavailable finite ranks are recorded instead of silently clipped. Outer coverage is empirical because these biological pair examples are dependent. The diagnostic does not support a deployed abstention guarantee. See [Shafer and Vovk (2008)](https://jmlr.org/papers/v9/shafer08a.html) for the assumptions behind conformal confidence statements.','','Run `.venv/Scripts/python.exe main.py ceiling` to verify inputs, reproduce missing stages, test the code, regenerate tables and build the paper. Full numerical records, technical sensitivity values, calibration errors, fit/calibration/validation identities and source hashes are in `results/architecture/ceiling_report.json` and the per-dataset study folders.'])
    (ROOT/'results/architecture/ceiling_report.md').write_text('\n'.join(prose)+'\n')
    write_json(ROOT/'results/architecture/ceiling_report.json',report)
    print('\n'.join(prose[4:12]))


if __name__=='__main__':main()
