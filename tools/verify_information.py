"""Replay every saved fold and validate existing hash-bound evaluator refusal."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal import load_model
from opal.artifacts import array_hash,read_json,sha256,write_json
from opal.information import load_head,predict_head
from opal.series import evaluate_predictions,freeze_predictions
from opal.series_bilinear import predict_reference
from prepare_information import LOCAL,OUT,resource_check


def run():
    sources=read_json(OUT/'measurement/summaries.json')
    results=[]
    with threadpool_limits(limits=1):
        for source in sources:
            with np.load(source['bank'],allow_pickle=False) as z:bank={k:z[k] for k in ['singles','responses','pair_ids','culture_ids']}
            with np.load(source['summary'],allow_pickle=False) as z:summary={k:z[k] for k in z.files}
            for path in sorted((OUT/'development/folds').glob(source['source']+'-*.json')):
                row=read_json(path);work=Path(row['checkpoint']);contract=read_json(work/'freeze.json')['contract']
                if sha256(work/'predictions.npz')!=row['prediction_sha256']:raise ValueError('Original predictions changed')
                if sha256(source['bank'])!=source['bank_sha256'] or sha256(source['summary'])!=source['summary_sha256']:raise ValueError('Source changed')
                selected=(bank['culture_ids']==row['stratum'])&np.isin(bank['pair_ids'],row['held_pair_ids'])
                query={k:v[selected] for k,v in bank.items() if k!='responses'}
                if row['precision']:query['singles']=summary['precision_singles'][selected]
                means=summary['precision_projected' if row['precision'] else 'means_projected'][selected]
                base=load_model(work/'base');bp=predict_reference(base.config,base.fitted['reference'],query)
                with np.load(work/'predictions.npz',allow_pickle=False) as z:predictions={k:z[k] for k in ['Base','mean','capacity_mean','variance','shuffle']}
                if not np.allclose(bp,predictions['Base'],atol=1e-10,rtol=0):raise ValueError('Base changed')
                error=[]
                for arm in ['mean','capacity_mean','variance','shuffle']:
                    variance=summary[('precision_' if row['precision'] else '')+('shuffle' if arm=='shuffle' else 'variance')][selected]
                    replay=bp+predict_head(load_head(work/arm,contract),means,variance,512)
                    error.append(float(np.max(np.abs(replay-predictions[arm]))))
                if max(error)>1e-10:raise ValueError('Head changed')
                # Reuse the established evaluator with prediction-only replay. This
                # new verification freeze does not claim to replace original chronology.
                class Frozen:
                    def __init__(self):self.contract=contract;self.access=base.access
                    def predict(self,q,c):
                        if set(q)!={'singles','pair_ids','culture_ids'}:raise ValueError('Outcome-free query required')
                        if c!=contract or array_hash(*(q[k] for k in sorted(q)))!=array_hash(*(query[k] for k in sorted(query))):raise ValueError('Frozen query changed')
                        return dict(base.predict(q,c),predictions=predictions['variance'])
                with tempfile.TemporaryDirectory(dir=LOCAL,prefix='evaluator-replay-') as tmp:
                    dest=Path(tmp)/'freeze';freeze_predictions(dest,Frozen(),query,contract)
                    evaluated=evaluate_predictions(dest,query,contract,bank['responses'][selected])
                    if abs(evaluated['mse']-row['metrics']['variance']['mse'])>1e-12:raise ValueError('Evaluator score changed')
                    refusals=[]
                    bad=dict(query);bad['singles']=query['singles']+1
                    for q,c in [(bad,contract),(query,dict(contract,units='normalized_count_change')),(query,dict(contract,checkpoint_role='all-development deployment refit after scoring'))]:
                        try:evaluate_predictions(dest,q,c,bank['responses'][selected])
                        except ValueError:refusals.append(True)
                        else:raise ValueError('Evaluator accepted mutation')
                    with (dest/'predictions.npz').open('ab') as f:f.write(b'fixture-tamper')
                    try:evaluate_predictions(dest,query,contract,bank['responses'][selected])
                    except ValueError:refusals.append(True)
                    else:raise ValueError('Evaluator accepted changed prediction')
                results.append({'id':row['id'],'head_replay_max_abs':max(error),'existing_evaluator_mse_parity':True,'hash_role_refusals':len(refusals)})
    preservation=read_json(OUT/'preservation.json')
    mismatches=[]
    for p,h in preservation['files'].items():
        if Path(p).is_file() and sha256(p)==h:
            continue
        if Path(p)==Path('paper/main.pdf') and (OUT/'verification/paper_preservation.json').exists():
            saved=read_json(OUT/'verification/paper_preservation.json')
            if sha256(saved['old_pdf'])==h:
                continue
        mismatches.append(p)
    if mismatches:raise ValueError('Historical evidence changed: '+str(mismatches))
    write_json(OUT/'verification/replay.json',{'records':results,'historical_files_unchanged':len(preservation['files']),
        'historical_mismatches':mismatches,'replay_atol':1e-10,'verification_freezes':'temporary verification only; original prediction hashes and timestamps preserved'})
    print(len(results),'folds replay; evaluator input/unit/deployment/prediction refusals pass; historical files unchanged')


if __name__=='__main__':run()
