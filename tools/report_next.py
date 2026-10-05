"""Build a recoverable next-cycle evidence ledger from recorded artifacts only."""
import argparse
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    paths = {
        'pilot': 'emeraldbay-pilot-v1/results.json',
        'confirmation': 'dspin-confirmation-v1/results.json',
        'confirmation_review': 'dspin-confirmation-v1/confirmation_review.json',
        'cpa': 'native-cpa-aggregate-v1/results.json',
        'gears_norman': 'native-gears-norman-v1/results.json',
        'gears_wessels': 'native-gears-wessels-v1/results.json',
        'estimated_singles': 'missing-single-pilot-v3/results.json',
        'pertadapt': 'pertadapt-native-probe-v1/results.json'}
    ledger = {name: {'path': str(ROOT / 'results/next' / relative),
                    'sha256': sha256(ROOT / 'results/next' / relative),
                    'result': read_json(ROOT / 'results/next' / relative)} for name, relative in paths.items()}
    write_json(args.output / 'ledger.json', {'evaluator_outcomes_reopened': False, 'experiments': ledger,
        'scientific_release': 'not established', 'engineering_release': 'local research preview',
        'master_plan_included': False})
    r = {name: entry['result'] for name, entry in ledger.items()}
    table = '\n'.join([
        '| Registered experiment | Metric | Result and interpretation |', '| --- | --- | --- |',
        f"| EmeraldBay additive | MSE | {r['pilot']['mse']['additive']:.6f} |",
        f"| EmeraldBay fitting-selected kernel | MSE | {r['pilot']['mse']['selected']:.6f}; no gain over its selected simple reference |",
        f"| EmeraldBay native CPA aggregate adaptation | MSE | {r['cpa']['mse']:.6f}; retrospective pilot, not canonical cell-level CPA |",
        f"| Independent D-SPIN frozen procedure | MSE | {r['confirmation']['mse']['selected']:.6f} |",
        f"| D-SPIN source-selected reference | MSE | {r['confirmation']['mse'][r['confirmation']['development_selected_simple_reference']]:.6f}; frozen procedure is worse |",
        f"| D-SPIN empirical 90% source radius | Held-row coverage | {r['confirmation']['empirical_radius_coverage']:.1%}; does not transfer as a calibrated interval |",
        f"| Native GEARS aggregate adaptation, Norman | Random / partner MSE | {r['gears_norman']['mse']['random']:.6f} / {r['gears_norman']['mse']['partner']:.6f} |",
        f"| Native GEARS aggregate adaptation, Wessels | Random / partner MSE | {r['gears_wessels']['mse']['random']:.6f} / {r['gears_wessels']['mse']['partner']:.6f} |",
        f"| Missing-single source validation | Gain over target mean | {r['estimated_singles']['skill_over_source_target_mean']:.1%}; complexity gate remains closed |",
        f"| Missing-single pilot | Total MSE | {r['estimated_singles']['mse']:.6f}; main/residual error cancellation prevents a transferable capability claim |",
        f"| Native PertAdapt adapter feasibility | Outcome | {r['pertadapt']['adapter_probe']}; no biological score |"])
    report = '# Opal next-cycle research preview\n\nThe local software runs, but positive biological context-transfer confirmation and transferable reliability are not established. No historical evaluator was reopened, and the local master plan is excluded from archives.\n\n' + table
    report += '\n\nThe two native adaptations use the original model and loss implementations with aggregate observations. GEARS uses graphs derived only from permitted single profiles rather than external GO. These results do not support a canonical state-of-the-art comparison. CPA has a reproducible weight/reference export with an exact no-label restore.\n\n'
    report += 'D-SPIN supplies a second study with matched doses and within-batch vehicle references. Whole Myeloid and NK lineages are held out across both batches. Published lineage labels were assigned after treatment, dosage pairs share four compounds, and batches are not biological replicates. The frozen procedure fails the 5% skill gate.\n\n'
    report += 'The estimated-single route uses a separate schema, never marks estimates as measured, reports the source of each single, and abstains when an exact intervention/dose has no source reference. Observed-single intervals never apply to it.\n\n'
    report += 'The next scientific step is a replicated crossed-context screen or an independently qualified corpus, followed by a fresh sealed confirmation. Conditional learned encoders, higher-order interventions, sequential treatment and mechanistic claims remain outside the demonstrated release.\n'
    (args.output / 'report.md').write_text(report, encoding='utf-8')
    with zipfile.ZipFile(args.output / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        paths = list((ROOT / 'src/opal').rglob('*.py'))
        paths += [ROOT / 'tools' / name for name in ['report_next.py', 'evaluate_confirmation.py',
                    'study_main_effects.py', 'study_native_cpa.py', 'study_native_gears.py',
                    'probe_native_pertadapt.py', 'package_native_cpa.py', 'predict_native_cpa.py']]
        paths += [ROOT / 'tools/external/CPA/LICENSE', ROOT / 'tools/external/GEARS/LICENSE']
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
    write_json(args.output / 'manifest.json', {name: sha256(args.output / name)
               for name in ['ledger.json', 'report.md', 'source_snapshot.zip']})


if __name__ == '__main__':
    main()
