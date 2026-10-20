"""Post-primary descriptive assay-shift checks, with no fitting or score replacement."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import load_npz, write_json
from opal.models.fitness_inference import FitnessPredictor

MATCHED = {'AURKA', 'CENPM', 'COG3', 'CTNNBL1', 'DAP3', 'MAX', 'MTOR',
           'NDUFA8', 'NDUFA9', 'PDSS2', 'ROMO1', 'SEC22B'}


def diagnose(original, prepared, evaluation, output):
    if output.exists():
        raise FileExistsError('Preserve prior diagnostic outputs')
    model = FitnessPredictor(original/'fitness-reference-model')
    old = load_npz(original/'prepared/profiles.npz')
    lookup = {tuple(sorted((str(old['genes'][a]), str(old['genes'][b])))): i
              for i, (a, b) in enumerate(old['pairs'])}
    receipt = {'label': 'post-primary descriptive diagnosis only; no fitting, tuning or primary-result replacement',
               'cohorts': {}}
    for cohort in ('environment1', 'environment2', 'parp2022', 'parp2023'):
        d = load_npz(prepared/f'{cohort}.npz')
        f = load_npz(evaluation/f'{cohort}_predictions.npz')
        y = d['gamma'][f['rows']].mean(1)
        singles = {g: s.mean() for g, s in zip(d['single_genes'], d['singles'], strict=True)}
        genes = [g for g in singles if g in model.index and model.models['K562'].support[model.index[g]]]
        a = np.array([model.arrays['singles'][0, model.index[g]] for g in genes])
        b = np.array([singles[g] for g in genes])
        score = {'supported_singles': len(genes), 'original_single_mean': float(a.mean()),
                 'external_single_mean': float(b.mean()), 'single_difference_mse': float(np.mean((a-b)**2)),
                 'single_pearson': float(np.corrcoef(a, b)[0, 1]),
                 'single_median_absolute_ratio': float(np.median(abs(b[a < -.05])/abs(a[a < -.05])))}
        mask = np.array([tuple(p) in lookup for p in f['pair_names']])
        ids = [lookup[tuple(p)] for p in f['pair_names'][mask]]
        source = old['responses'][0, ids]
        score['remeasured_old_pairs'] = {'pairs': len(ids), 'old_mean': float(source.mean()),
            'new_mean': float(y[mask].mean()), 'mse': float(np.mean((source-y[mask])**2)),
            'pearson': float(np.corrcoef(source, y[mask])[0, 1])}
        if cohort.startswith('environment'):
            mask = np.array([a in MATCHED and b in MATCHED for a, b in f['pair_names']])
            score['both_reagents_matched_genes'] = {'pairs': int(mask.sum()),
                'novel_pairs': int(np.sum(mask & (f['strata'] == 'absent_original_eligible_bank'))),
                **{f'{m}_mse': float(np.mean((f[m][mask]-y[mask])**2))
                   for m in ('completion', 'main_effect', 'minimum')}}
        score['replicate_disagreement_mse'] = float(np.mean(np.diff(d['gamma'][f['rows']], axis=1)**2))
        receipt['cohorts'][cohort] = score
    write_json(output, receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'prepared', 'evaluation', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.original, args.prepared, args.evaluation, args.output)
