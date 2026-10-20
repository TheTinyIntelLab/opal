"""Export the fixed validation-selected references without refitting factors."""
import argparse
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import load_npz,read_json,sha256,write_json
from opal.models.fitness import GrowthReference
from opal.models.fitness_inference import FitnessPredictor


def export(run,output):
    if output.exists():raise FileExistsError('Preserve previous exported models')
    protocol=read_json(run/'protocol.json');result=read_json(run/'results.json')
    data=Path(protocol['data_root'])
    if sha256(data/'profiles.npz')!=protocol['profiles_sha256']:raise ValueError('Data drift')
    d=load_npz(data/'profiles.npz');splits=load_npz(run/'assignment.npz')['split'];train=np.flatnonzero(splits==0)
    singles=d['singles'].mean(1);output.mkdir(parents=True)
    arrays={'genes':d['genes'],'singles':singles};contexts={}
    expected=load_npz(run/'reference_and_schema.npz')['reference_matrix']
    for c,name in enumerate(d['contexts']):
        chosen=result['selection']['reference'][c]
        reference=GrowthReference(chosen['kind'],chosen['alpha']).fit(singles,d['pairs'][train],d['responses'][:,train])
        if not np.array_equal(reference.predict(d['pairs'])[c],expected[c]):raise RuntimeError('Reference serialization refit differs')
        if reference.fits:
            center,scale,coefficient,mean=reference.fits[c]
            arrays.update({f'c{c}_center':center,f'c{c}_scale':scale,f'c{c}_coefficient':coefficient,f'c{c}_mean':mean})
        selection=result['selection']['baseline_per_context'][c]
        if 'checkpoint' not in selection:raise ValueError('This export requires a selected completion reference')
        source=run/selection['checkpoint']
        fit=next(r for r in result['all_fits'] if r['checkpoint']==selection['checkpoint'])
        if sha256(source)!=fit['checkpoint_sha256']:raise ValueError('Checkpoint drift')
        shutil.copyfile(source,output/source.name)
        contexts[str(name)]={'context_index':c,'checkpoint':source.name,'reference_kind':chosen['kind'],'reference_alpha':chosen['alpha']}
    np.savez_compressed(output/'reference.npz',**arrays)
    manifest={'contexts':contexts,'files':{p.name:sha256(p) for p in output.iterdir() if p.is_file()},
        'source_protocol_sha256':sha256(run/'protocol.json'),'source_result_sha256':sha256(run/'results.json'),
        'scope':'held pairs of supported genes; measured growth gamma, not expression or new contexts',
        'attribution':'Horlbeck et al Cell 2018; GSE116198; author dataset DOI 10.17632/rdzk59n6j4.1; weights/data local only',
        'refit_scope':'only exact closed-form training reference refit for serialization; no factor retraining or test-driven selection'}
    write_json(output/'manifest.json',manifest)
    predictor=FitnessPredictor(output);test=np.flatnonzero(splits==2);held=load_npz(run/'predictions.npz')['baseline']
    named=[tuple(d['genes'][p]) for p in d['pairs'][test]]
    differences={str(context):float(np.max(np.abs(predictor.predict(named,str(context))-held[c]))) for c,context in enumerate(d['contexts'])}
    if max(differences.values())>1e-12:raise RuntimeError('Composite prediction replay differs')
    receipt={'status':'passed','max_abs_test_replay':differences,'files':manifest['files'],'exporter_sha256':sha256(__file__)}
    write_json(output/'verification.json',receipt);print(receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.run,a.output)
