"""Generate manuscript evidence from saved next-cycle scores, without evaluator access."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = {}

    def read(name):
        path = ROOT / 'results/next' / name / 'results.json'
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding='utf-8'))

    pilot = read('emeraldbay-pilot-v1')
    cpa = read('native-cpa-aggregate-v1')
    independent = read('dspin-confirmation-v1')
    missing = read('missing-single-pilot-v3')
    norman = read('native-gears-norman-v1')
    wessels = read('native-gears-wessels-v1')
    rows = [
        ('EmeraldBay', 'Observed additive', pilot['mse']['additive']),
        ('EmeraldBay', 'Selected pooled-sum kernel', pilot['mse']['selected']),
        ('EmeraldBay', 'Native CPA, aggregate inputs', cpa['mse']),
        ('D-SPIN', 'Observed additive', independent['mse']['additive']),
        ('D-SPIN', 'Source-selected pooled-pair kernel', independent['mse'][independent['development_selected_simple_reference']]),
        ('D-SPIN', 'Frozen kernel procedure', independent['mse']['selected']),
        ('EmeraldBay, masked singles', 'Estimated additive', missing['additive_estimated_single_mse']),
        ('EmeraldBay, masked singles', 'Estimated singles plus residual', missing['mse']),
        ('Norman development', 'Native GEARS, random', norman['mse']['random']),
        ('Norman development', 'Native GEARS, partner', norman['mse']['partner']),
        ('Wessels development', 'Native GEARS, random', wessels['mse']['random']),
        ('Wessels development', 'Native GEARS, partner', wessels['mse']['partner']),
    ]
    lines = [r'\begin{tabular}{lll}', r'\toprule', r'Population / contract & Method & MSE \\', r'\midrule']
    lines.extend(f'{population} & {method} & {mse:.8f} ' + r'\\' for population, method, mse in rows)
    lines += [r'\bottomrule', r'\end{tabular}']
    output = ROOT / 'paper/evidence/next_cycle_results.tex'
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    manifest = {'sources': sources, 'evaluator_outcomes_reopened': False,
                'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    (output.parent / 'next_cycle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
