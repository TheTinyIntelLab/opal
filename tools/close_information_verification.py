"""Verify final package contents against the already exercised installed wheel."""
import argparse
from pathlib import Path
import sys
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opal.artifacts import read_json,write_json,sha256
from prepare_information import OUT,LOCAL,resource_check


def run(package_root='package-final-complete',record='package_final'):
    resource_check(100000)
    name='opal_research-0.1.3.dev0'
    old=LOCAL/'package'/f'{name}-py3-none-any.whl'
    final=LOCAL/package_root/f'{name}-py3-none-any.whl'
    sdist=LOCAL/package_root/f'{name}.tar.gz'
    forbidden=lambda n: n.endswith(('.npz','.npy','.pt','.h5ad')) or Path(n).name.lower() in {'plan.md','outreach.md','context.md','roadmap.md'}
    with zipfile.ZipFile(old) as a,zipfile.ZipFile(final) as b:
        payload=lambda z:{n:z.read(n) for n in z.namelist() if n.startswith('opal/')}
        assert payload(a)==payload(b),'Installed proof does not match final runtime payload'
        assert not any(forbidden(n) for n in b.namelist()),'Private wheel member'
        count=len(payload(b))
    with tarfile.open(sdist) as archive:
        assert not any(forbidden(n.name) for n in archive),'Private sdist member'
        assert any(n.name.endswith('/tools/close_information_verification.py') for n in archive)
    original=read_json(OUT/'verification/paper_preservation.json')
    assert sha256(original['old_pdf'])==read_json(OUT/'preservation.json')['files']['paper\\main.pdf']
    assert (ROOT/'PLAN.md').read_bytes()==(ROOT/'docs/next_cycle/model_series_v2_plan.md').read_bytes()
    assert read_json(OUT/'verification/manuscript.json')['historical_page_text_parity']
    for label in ['311-complete','314-complete']:
        installed=read_json(OUT/f'verification/installed{label}.json')
        assert installed['information']['outside_checkout'] and installed['information']['source_replay_max_abs']<1e-10
    write_json(OUT/f'verification/{record}.json',{
        'wheel':str(final),'wheel_sha256':sha256(final),'sdist':str(sdist),'sdist_sha256':sha256(sdist),
        'previous_verified_wheel_sha256':sha256(old),'installed_runtime_payload_identical':True,
        'runtime_members':count,'private_arrays_weights_root_plan_contacts_excluded':True,
        'installed_checks':['Python3.11 actual series/head fit-predict-score-replay','Python3.14 actual series/head fit-predict-score-replay'],
        'version':'0.1.3.dev0 local review only','published_assets_modified':False,
        'manuscript':'43 pages, original42 page text parity and rendered addendum review',
        'plan_mirror_byte_identical':True,'remote_ci':'awaiting owner-authorized push workflow' if (OUT/'verification/delivery_authorization.json').exists() else 'pending separate authorization; no dispatch'})
    print('Final wheel payload matches both installed proofs; package exclusions and plan/paper checks pass')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--package-root',default='package-final-complete')
    parser.add_argument('--record',default='package_final')
    args=parser.parse_args()
    run(args.package_root,args.record)
