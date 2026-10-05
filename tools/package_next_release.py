"""Create a local research bundle with source, notices and safe pilot exports."""
import argparse
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, write_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    validation = read_json(ROOT / 'results/next/release-smoke-v6/installed_validation.json')
    if validation['torch_imported'] or validation['estimated_maximum_absolute_error'] > 1e-8:
        raise ValueError('Installed research export validation is incomplete')
    args.output.mkdir(parents=True)
    paths = set((ROOT / 'src/opal').rglob('*.py')) | set((ROOT / 'tests').glob('test_*.py'))
    paths |= {ROOT / name for name in ['main.py', 'pyproject.toml', 'README.md', 'LICENSE',
              'docs/next_cycle.md', 'docs/model_card_next.md', 'docs/decisions.md',
              'data/public_method_registry.json', 'paper/main.tex', 'paper/references.bib', 'paper/main.pdf']}
    paths |= set((ROOT / 'configs/environments').glob('*.txt'))
    paths |= {ROOT / 'configs/datasets' / name for name in ['emeraldbay-pilot-v1.json',
              'dspin-confirmation-v1.json', 'context-source-candidates.json', 'crossed-context-screen-spec.json']}
    paths |= {ROOT / 'configs/experiments' / name for name in ['kernel-attribution-v1.json',
              'acquisition-v1.json', 'native-cpa-aggregate-v1.json']}
    names = ['package_next_release.py', 'run_next.py', 'next_audit.py', 'download.py', 'download_dspin.py',
             'prepare_emeraldbay.py', 'prepare_dspin.py', 'evaluate_confirmation.py', 'study_main_effects.py',
             'study_native_cpa.py', 'study_native_gears.py', 'probe_native_pertadapt.py', 'report_next.py',
             'repackage_contextual.py', 'package_native_cpa.py', 'predict_native_cpa.py', 'report_next_paper.py',
             'study_kernel_attribution.py', 'study_acquisition.py', 'improve_residual_model.py',
             'report_ceiling.py', 'fetch_public_sources.py', 'profile_context_checkpoint.py']
    paths |= {ROOT / 'tools' / name for name in names}
    for name in ['CPA', 'GEARS']:
        paths |= {ROOT / 'tools/external' / name / 'LICENSE'}
        paths |= set((ROOT / 'tools/external' / name).rglob('*.py'))
        paths |= {path for path in (ROOT / 'tools/external' / name).iterdir()
                  if path.is_file() and (path.suffix == '.toml' or path.name in {'README.md', 'requirements.txt'})}
    paths |= {ROOT / 'data/public_methods' / (name + '_tree.json') for name in ['cpa', 'gears', 'pertadapt']}
    paths |= {ROOT / 'results/next/release-smoke-v6/opal_research-0.1.0-py3-none-any.whl',
              ROOT / 'results/next/release-smoke-v6/installed_validation.json'}
    for directory in ['results/next/portable-checkpoints-v1/emeraldbay',
                      'results/next/missing-single-pilot-v3/checkpoint',
                      'results/next/native-cpa-release-v1/checkpoint',
                      'results/next/research-ledger-v2', 'paper/evidence']:
        paths |= {path for path in (ROOT / directory).rglob('*') if path.is_file()}
    paths = sorted(paths)
    if any(not path.is_file() or path.name == 'opal_next_master_plan.md' for path in paths):
        raise ValueError('Unexpected local release file')
    with zipfile.ZipFile(args.output / 'opal-local-research-preview.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(args.output / 'opal-local-research-preview.zip') as archive:
        if archive.testzip() is not None or any('opal_next_master_plan' in name for name in archive.namelist()):
            raise ValueError('Local research bundle integrity failure')
    write_json(args.output / 'manifest.json', {'scope': 'Local research preview; no public upload',
        'zip_sha256': sha256(args.output / 'opal-local-research-preview.zip'),
        'files': {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths},
        'master_plan_included': False, 'historical_evaluator_outcomes_included': False,
        'raw_datasets_included': False, 'dspin_weights_included': False,
        'pertadapt_backbone_and_graph_included': False,
        'dataset_terms': 'EmeraldBay CC BY 4.0 with linked source attribution',
        'native_notices': {'CPA': 'BSD 3-Clause', 'GEARS': 'MIT'},
        'scientific_gates': 'Failed independent material skill and transferable uncertainty; PertAdapt incomplete'})
    print('Local research bundle verified:', args.output / 'opal-local-research-preview.zip')


if __name__ == '__main__':
    main()
