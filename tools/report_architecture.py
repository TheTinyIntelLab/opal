"""Generate architecture evidence from completed development-only artifacts."""
from pathlib import Path
import argparse,sys,platform,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from opal.artifacts import read_json,write_json,load_npz,sha256
from opal.inference import Predictor

LABELS={'norman-log2':'Norman','wessels-log2':'Wessels original','wessels-expanded-log2':'Wessels expanded','wessels-expanded-protected-log2':'Wessels protected'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    # Verify fitting-resource lineage before publishing cached numerical results.
    for name in LABELS:
        resource=ROOT/'data/architecture_resources'/name
        manifest=read_json(resource/'manifest.json')
        if sha256(resource/'resources.npz')!=manifest['resources_sha256'] or sha256(ROOT/'data/prepared'/name/'profiles.npz')!=manifest['profiles_sha256']:
            raise ValueError('Architecture fitting-resource fingerprint changed: '+name)
    evidence=ROOT/'paper/evidence';evidence.mkdir(exist_ok=True)
    study_names=['v1','protected-v1','experts-v1','tabular-v1','prior-kernels-v1','noise-kernels-v1','noise-kernels-v2','noise-kernels-v3','transfer-v3','linear-bounded-v1','common-coverage-held-v1','common-coverage-held-v2','common-coverage-held-v3']
    inputs={};studies={};missing=[]
    for study in study_names:
        path=ROOT/'results/architecture'/study/('results.json' if study.startswith('common') else 'summary.json')
        if not path.exists():missing.append(study);continue
        studies[study]=read_json(path)
        if study in ['v1','protected-v1']:
            studies[study]={name:read_json(path.parent/name/'results.json') for name in studies[study]}
            for name in studies[study]:inputs[str((path.parent/name/'results.json').relative_to(ROOT))]=sha256(path.parent/name/'results.json')
        inputs[str(path.relative_to(ROOT))]=sha256(path)
    if missing and not args.allow_partial:raise ValueError('Unfinished required studies: '+str(missing))
    for track in study_names:
        if track not in studies or track.startswith('common'):continue
        prefix='semantic-only-' if track.startswith('prior') else ('enriched-' if (track.startswith('transfer') or track.startswith('linear-bounded')) else '')
        for name in studies[track]:
            protocol_path=ROOT/'results/architecture'/track/name/'protocol.json'
            protocol=read_json(protocol_path)
            expected=protocol.get('resource_sha256',protocol.get('resources_sha256',protocol.get('source_sha256')))
            if expected and sha256(ROOT/'data/architecture_resources'/(prefix+name)/'resources.npz')!=expected:
                raise ValueError('Study resource lineage changed: '+track+'/'+name)
            inputs[str(protocol_path.relative_to(ROOT))]=sha256(protocol_path)
    rows=[];kernel={};selected=[]
    for track in ['v1','protected-v1']:
        for name,result in studies[track].items():
            for kind in ['random','partner']:
                metrics=result[kind]['metrics'];kernel[name,kind]=metrics['kernel']['mse_per_occurrence']
                for model,metric in metrics.items():rows.append({'dataset':name,'protocol':kind,'study':track,'model':model,'mse':metric['mse_per_occurrence']})
                selected.extend([r['competitive_selection'] for r in result[kind]['records']])
    for track in ['experts-v1','transfer-v3','linear-bounded-v1','prior-kernels-v1','noise-kernels-v1','noise-kernels-v2','noise-kernels-v3']:
        for name,result in studies.get(track,{}).items():
            for kind in ['random','partner']:
                for model,mse in result[kind]['mse_per_occurrence'].items():rows.append({'dataset':name,'protocol':kind,'study':track,'model':model,'mse':mse})
    for name,result in studies.get('tabular-v1',{}).items():
        for kind in ['random','partner']:rows.append({'dataset':name,'protocol':kind,'study':'tabular-v1','model':'TabICLv2','mse':result[kind]['mse_per_occurrence']})
    noise=[]
    for name in ['norman-log2','wessels-log2','wessels-expanded-protected-log2']:
        for kind in ['random','partner']:
            gains=[];placebo=[]
            for track in ['noise-kernels-v1','noise-kernels-v2','noise-kernels-v3']:
                m=studies[track][name][kind]['mse_per_occurrence'];gains.append(100*(1-m['measurement']/m['unweighted']));placebo.append(100*(1-m['permuted_variance']/m['unweighted']))
            noise.append({'dataset':name,'protocol':kind,'gain_percent_by_seed':gains,'placebo_percent_by_seed':placebo})
    coverage=studies['common-coverage-held-v1'];coverage_gains={}
    for kind in ['random','partner']:
        m=coverage[kind]['mse_per_occurrence'];coverage_gains[kind]={k:100*(1-v/m['original']) for k,v in m.items()}
    coverage_sensitivity={}
    for kind in ['random','partner']:
        coverage_sensitivity[kind]={}
        for model in ['extra_pairs','extra_pairs_noise']:
            gains=[]
            for version in [1,2,3]:
                m=studies[f'common-coverage-held-v{version}'][kind]['mse_per_occurrence']
                gains.append(100*(1-m[model]/m['original']))
            coverage_sensitivity[kind][model]=gains
    ids=lambda data,key:set(tuple(sorted(map(str,p))) for p in data['targets'][data[key]])
    old=load_npz(ROOT/'data/prepared/wessels-log2/profiles.npz');broad=load_npz(ROOT/'data/prepared/wessels-expanded-log2/profiles.npz');protected=load_npz(ROOT/'data/prepared/wessels-expanded-protected-log2/profiles.npz')
    exposure={'historical_query_in_broad_development':len(ids(old,'query_pairs')&ids(broad,'pairs')),
        'historical_query_in_protected_development':len(ids(old,'query_pairs')&ids(protected,'pairs')),
        'protected_query_already_in_broad_development':len(ids(protected,'query_pairs')&ids(broad,'pairs')),
        'independent_confirmation':False}
    if exposure['historical_query_in_protected_development']:raise ValueError('Protected identity contract failed')
    timing=[]
    import torch
    torch.set_num_threads(1)
    for name in ['norman-log2','wessels-log2','wessels-expanded-protected-log2']:
        folder=ROOT/'results/architecture/noise-kernels-v1'/name/'checkpoint'
        data=load_npz(ROOT/'data/prepared'/name/'profiles.npz')
        t=time.perf_counter();predictor=Predictor(folder);cold=(time.perf_counter()-t)*1000
        a,b=map(str,data['targets'][data['query_pairs'][0]])
        for _ in range(10):predictor.predict(a,b)
        times=[]
        for _ in range(100):
            t=time.perf_counter();prediction=predictor.predict(a,b);times.append((time.perf_counter()-t)*1000)
        if not np.isfinite(prediction['effect']).all():raise ValueError('Nonfinite exported inference')
        timing.append({'dataset':name,'cold_load_ms':cold,'warm_median_ms':float(np.median(times)),
            'warm_p95_ms':float(np.percentile(times,95)),'artifact_bytes':sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
            'repeats':100,'threads':1,'machine':platform.platform(),'role':'unscored query inference, not held-out evaluation'})
    report={'status':'partial' if missing else 'complete','missing':missing,'input_sha256':inputs,'rows':rows,'noise':noise,
        'common_coverage':coverage,'coverage_gain_percent':coverage_gains,'coverage_gain_percent_by_seed':coverage_sensitivity,'exposure_audit':exposure,'competitive_selections':selected,
        'inference':timing,'scope':'post-primary exploratory development studies; no historical evaluator response access',
        'uncertainty':'dependent resamplings; no confidence intervals or biological replication',
        'resource_terms':'PRESAGE semantic derivatives CC BY-NC 4.0; source code license does not relicense weights'}
    write_json(ROOT/'results/architecture/report.json',report)
    def table(headers,records):
        return '\\begin{tabular}{'+'l'*len(headers)+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(map(str,r))+' \\\\' for r in records)+'\n\\bottomrule\n\\end{tabular}\n'
    # Compact primary comparator table; the full report preserves every ablation.
    table_rows=[]
    for name in LABELS:
        for kind in ['random','partner']:
            def score(study,model):
                found=[r['mse'] for r in rows if r['dataset']==name and r['protocol']==kind and r['study']==study and r['model']==model]
                return f'{found[0]:.6f}' if found else '--'
            table_rows.append([LABELS[name],kind,f'{kernel[name,kind]:.6f}',score('protected-v1' if 'protected' in name else 'v1','adaptive_neural'),score('experts-v1','GP'),score('tabular-v1','TabICLv2')])
    (evidence/'architecture_table.tex').write_text(table(['Dataset','Fold','Kernel','Neural selection','GP','TabICLv2'],table_rows))
    (evidence/'architecture_noise.tex').write_text(table(['Dataset','Fold','Real noise gain (\\%)','Permuted gain (\\%)'],[[LABELS[n['dataset']],n['protocol'],f"{min(n['gain_percent_by_seed']):.2f} to {max(n['gain_percent_by_seed']):.2f}",f"{min(n['placebo_percent_by_seed']):.2f} to {max(n['placebo_percent_by_seed']):.2f}"] for n in noise]))
    (evidence/'architecture_coverage.tex').write_text(table(['Fold','Original','Extra singles','Extra pairs','Pairs + noise'],[[kind]+[f"{coverage[kind]['mse_per_occurrence'][k]:.6f}" for k in ['original','extra_singles','extra_pairs','extra_pairs_noise']] for kind in ['random','partner']]))
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for ax,kind in zip(axes,['random','partner']):
        for i,name in enumerate(['norman-log2','wessels-log2','wessels-expanded-protected-log2']):
            n=next(n for n in noise if n['dataset']==name and n['protocol']==kind)
            ax.scatter(np.full(3,i)-.08,n['gain_percent_by_seed'],label='Measured variance' if i==0 else None,color='#236892')
            ax.scatter(np.full(3,i)+.08,n['placebo_percent_by_seed'],label='Permuted variance' if i==0 else None,color='#bd6b3e',marker='x')
        ax.axhline(0,color='gray',lw=.8);ax.set_title(kind+' folds');ax.set_xticks(range(3),['Norman','Wessels','Protected\nexpanded']);ax.set_ylabel('MSE reduction (%)');ax.legend(fontsize=8)
    for extension in ['pdf','png']:fig.savefig(evidence/('architecture-noise-gains.'+extension),dpi=160)
    plt.close(fig)
    lines=['# Architecture experiments','', 'These are exploratory development studies conducted after the historical outcomes were known. All scores weight validation occurrences equally; partner folds overlap.','', 'Reproduce with `python main.py frontier`. Completed experiment directories are preserved rather than overwritten.','', '## Measured outcomes','',f"Adding eligible protected Wessels training pairs reduces same-population MSE by {coverage_gains['random']['extra_pairs']:.2f}% on random folds and {coverage_gains['partner']['extra_pairs']:.2f}% on partner folds. Adding variance weighting gives {coverage_gains['random']['extra_pairs_noise']:.2f}% and {coverage_gains['partner']['extra_pairs_noise']:.2f}%. This comparison fixes the original validation identities and gene panel, and excludes historical query identities from training.",'', 'The baseline residual kernel is selected in every competitive architecture outer fold. GP, residual retrieval and the fixed four-estimator, eight-output-PC TabICLv2 comparator do not beat it on the original datasets. Semantic priors have small, inconsistent gains; shuffled identity controls must be considered.','']
    for n in noise:lines.append(f"- {LABELS[n['dataset']]} {n['protocol']}: measured-variance gain {min(n['gain_percent_by_seed']):.2f}% to {max(n['gain_percent_by_seed']):.2f}%; permuted control {min(n['placebo_percent_by_seed']):.2f}% to {max(n['placebo_percent_by_seed']):.2f}% across three dependent split seeds.")
    lines+=['','## Exposure and limits','',f"The broad expansion moves {exposure['historical_query_in_broad_development']} original query identities into development. The protected expansion moves zero, but {exposure['protected_query_already_in_broad_development']} of its query identities had already appeared in the broad development track. Neither expansion is independent confirmation.",'', 'Joint technical bootstraps share single and control draws across conditions. The noise likelihood uses diagonal pair variances and does not model their full covariance. GP intervals exclude query measurement variance and selection uncertainty. No calibrated abstention rule is claimed.','', 'External transfer uses independent K562 CRISPRi single effects on the released X scale to initialize a GO encoder and output directions. It is representation transfer, with a separately fitted current-context pair head. The external scale is not equated with Opal log2 pseudobulk effects.','', 'PRESAGE includes semantic priors and DEG rankings, not the required expression matrices; expression was separately downloaded from scPerturb record 7278143 and verified by publisher MD5. Semantic derivatives have CC BY-NC 4.0 terms and are excluded from Git.','', 'CPU-only runs leave native contextual PertAdapt/GEARS/CPA, broad hyperparameter tuning of foundation models, independent confirmation and frontier status unresolved. Every completed score, including unsuccessful additions, is retained in results/architecture/report.json.']
    transfer=studies.get('transfer-v3',{})
    if transfer:
        lines+=['','## Transfer and objective controls','']
        for name,result in transfer.items():
            for kind in ['random','partner']:
                m=result[kind]['mse_per_occurrence'];lines.append(f"- {LABELS[name]} {kind}: "+'; '.join(f'{k}={v:.6f}' for k,v in m.items() if k not in ['GP','retrieval'])+'.')
    lines+=['','## Coverage split sensitivities','']
    for kind,variants in coverage_sensitivity.items():
        for model,gains in variants.items():lines.append(f'- {kind}, {model}: {min(gains):.2f}% to {max(gains):.2f}% MSE reduction across three dependent seeds.')
    coverage_text='Across three dependent split seeds, extra pair labels reduce random-fold error by '+f"{min(coverage_sensitivity['random']['extra_pairs']):.2f}--{max(coverage_sensitivity['random']['extra_pairs']):.2f}\\%"+' and partner error by '+f"{min(coverage_sensitivity['partner']['extra_pairs']):.2f}--{max(coverage_sensitivity['partner']['extra_pairs']):.2f}\\%"+'. With measurement weighting the ranges are '+f"{min(coverage_sensitivity['random']['extra_pairs_noise']):.2f}--{max(coverage_sensitivity['random']['extra_pairs_noise']):.2f}\\%"+' and '+f"{min(coverage_sensitivity['partner']['extra_pairs_noise']):.2f}--{max(coverage_sensitivity['partner']['extra_pairs_noise']):.2f}\\%"+', respectively. These ranges are split sensitivities, not confidence intervals.\n'
    (evidence/'architecture_coverage_ranges.tex').write_text(coverage_text)
    bounded=studies.get('linear-bounded-v1',{})
    if bounded:
        lines+=['','## Bounded-feature sensitivity','', 'A late separate control clips training-standardized features at magnitude eight during fitting and prediction. It reduces the severe expanded-fold extrapolation errors but does not beat the kernel. Both unbounded results and interrupted optimization records remain preserved.','']
        for name,result in bounded.items():
            for kind in ['random','partner']:
                m=result[kind]['mse_per_occurrence']
                lines.append(f"- {LABELS[name]} {kind}: bounded ordinary MSE={m['linear_mse_bounded']:.6f}; bounded weighted MSE={m['linear_weighted_bounded']:.6f}; kernel MSE={m['kernel']:.6f}.")
    scope_update = '''\n## Scope update, 6 October 2026\n\nThe kernel-best conclusion above concerns the historical architecture batch.\nLater native aggregate-input CPA has chemical pilot MSE 0.157098 versus kernel\n0.172218, and adapted native GEARS beats the kernel on Wessels partner folds.\nThe frozen kernel procedure fails independent D-SPIN improvement and uncertainty\ngates. These new comparisons prevent extending the historical ranking to every\nsource or protocol. See docs/evidence.md for exact adaptation and exposure\nlimits, and the paper's context-transfer amendment for saved-score evidence.\n'''
    (ROOT/'results/architecture/report.md').write_text('\n'.join(lines)+'\n'+scope_update)
    if transfer:
        transfer_rows=[]
        for name,result in transfer.items():
            for kind in ['random','partner']:
                m=result[kind]['mse_per_occurrence']
                transfer_rows.append([LABELS[name],kind]+[f'{m[k]:.6f}' for k in ['no_transfer','transfer_encoder','transfer_decoder','transfer_both','linear_mse','linear_weighted']])
        text='\\begin{table}[htbp]\\centering\\normalsize'+table(['Dataset','Fold','No transfer','Encoder','Decoder','Both','Linear MSE','Linear weighted'],transfer_rows)+'\\caption{Corrected transfer and convex objective controls on development folds. Linear heads solve a regularized gene-space quadratic on frozen standardized features; neural heads use joint reference augmentation. These are different fitting procedures, and none uses historical query responses.}\\end{table}\n'
        (evidence/'architecture_transfer.tex').write_text(text)
    bounded=studies.get('linear-bounded-v1',{})
    if bounded:
        bounded_rows=[]
        for name,result in bounded.items():
            for kind in ['random','partner']:
                m=result[kind]['mse_per_occurrence'];old=transfer[name][kind]['mse_per_occurrence'] if name in transfer else {}
                bounded_rows.append([LABELS[name],kind,f"{m['kernel']:.6f}",f"{old.get('linear_mse',float('nan')):.6f}",f"{m['linear_mse_bounded']:.6f}",f"{m['linear_weighted_bounded']:.6f}"])
        text=r'\begin{table}[htbp]\centering\normalsize'+table(['Dataset','Fold','Kernel','Linear MSE','Bounded MSE','Bounded weighted'],bounded_rows)+r'\caption{Late exploratory bounded-feature controls after observing expanded-fold extrapolation. The standardized feature map clips at magnitude eight during training and prediction; the regularized gene-space objectives are solved to convergence. This is a separate amendment, not a replacement of the unfavorable unbounded controls.}\end{table}'+'\n'
        (evidence/'architecture_bounded.tex').write_text(text)
    print('Architecture report',report['status'],'coverage gains',coverage_gains,'exposure',exposure)


if __name__=='__main__':main()
