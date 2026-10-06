"""Compare repeat-well sensitivity with main-model differences, without culture claims."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import load_npz, write_json, sha256
from opal.contextual.contracts import ReferenceBank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True); p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--main-run', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    import anndata as ad
    data = ad.read_h5ad(args.raw, backed='r'); obs = data.obs
    eligible = obs.loc[obs.time == 24., ['cell_line', 'plate', 'replicate']].drop_duplicates()
    mapping = {}
    for context, rows in eligible.groupby(['cell_line', 'plate'], observed=True):
        labels = rows.replicate.astype(str).unique()
        if len(labels) != 1:
            raise ValueError('Ambiguous replicate mapping')
        mapping[str(context[0]) + '@' + str(context[1])] = str(labels[0])
    data.file.close()
    bank = ReferenceBank.load(args.dataset); a = bank.arrays
    records = []
    for line in ['A549', 'K562', 'MCF7']:
        contexts = [i for i, c in enumerate(a['contexts']) if bank.metadata['context_records'][str(c)]['biological_state'] == line]
        for ti, target in enumerate(a['targets']):
            rows = [c for c in contexts if a['available'][c, ti]]
            by_label = {label: [c for c in rows if mapping[str(a['contexts'][c])] == label] for label in ['rep1', 'rep2']}
            if any(len(v) != 1 for v in by_label.values()):
                raise ValueError('This sensitivity requires exactly one source profile per replicate label')
            difference = a['singles'][by_label['rep1'][0], ti] - a['singles'][by_label['rep2'][0], ti]
            records.append({'biological_state': line, 'intervention': str(target),
                'replicate_difference_mse': float(np.mean(difference ** 2)),
                'half_difference_squared': float(np.mean(difference ** 2) / 4)})
    saved = load_npz(args.main_run / 'predictions.npz'); ci, ti = saved['contexts'], saved['targets']
    y = a['singles'][ci, ti]
    difference = np.mean((saved['mean'] - y) ** 2, 1) - np.mean((saved['selected_simple'] - y) ** 2, 1)
    write_json(args.output / 'results.json', {'status': 'complete_development_sensitivity',
        'source_reference_hash': bank.fingerprint, 'prediction_sha256': sha256(args.main_run / 'predictions.npz'),
        'replicate_records': records, 'mean_replicate_difference_mse': float(np.mean([r['replicate_difference_mse'] for r in records])),
        'mean_half_difference_squared': float(np.mean([r['half_difference_squared'] for r in records])),
        'mean_target_average_minus_selected_simple_mse': float(difference.mean()),
        'interpretation': 'Nominal duplicate-well contrasts include measurement, plate and biological variation. Half-difference squared is a sensitivity under independent equal-variance replicates, not an identified measurement variance or biological accuracy ceiling.',
        'independent_cultures_verified': False, 'independent_studies': 1, 'biological_states': 3})


if __name__ == '__main__':
    main()
