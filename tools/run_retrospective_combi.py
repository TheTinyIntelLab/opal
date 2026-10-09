"""Run the separately amended exposed-source nominal Combi benchmark."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.replicated.contracts import sha, write_json
from opal.replicated.public_diagnostic import nominal_panel
from opal.replicated.retrospective import nested_pair_benchmark, reference_gene_panel
from diagnose_public_combi import load_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    menu_path = ROOT / 'configs/experiments/replicated_fixed_state_v1.json'
    menu = json.loads(menu_path.read_text())['menu']
    scope = {'fixed_utc': datetime.now(timezone.utc).isoformat(),
             'role': 'exposed_source_retrospective_nominal_pair_prediction',
             'source': 'GSE174695', 'outcomes_previously_inspected': True,
             'first_repository_expression_access': '2026-10-08',
             'metadata_sha256': sha(args.metadata), 'matrix_sha256': sha(args.matrix),
             'menu_source_sha256': sha(menu_path), 'menu': menu,
             'code_sha256': {str(path.relative_to(ROOT)): sha(path) for path in
                             [Path(__file__), ROOT / 'src/replicated/retrospective.py',
                              ROOT / 'src/replicated/public_diagnostic.py', ROOT / 'src/replicated/models.py',
                              ROOT / 'tools/diagnose_public_combi.py']},
             'seed': 20261009, 'shuffle_seeds': [11, 29, 47],
             'endpoint': '512 highest-variance unique per-run observed-single profiles; lexical gene tie break',
             'endpoint_conditioning': 'All query single references permitted; no pair responses select genes',
             'normalization': 'library total 10000; log2(1+x); mean controls/singles before log; mean pair logs',
             'folds': 'three seeded identity-balanced outer folds shared across runs; rotating inner holdout',
             'fit': 'separate per-run fits; select existing 16-entry menu on inner validation then refit outer training',
             'summary': 'MSE equal genes and canonical pairs within run, then equal runs; all fold records retained',
             'limits': 'single BLAS thread; existing compressed file <=80MB; uncompressed CSV <=600MB; no search expansion',
             'unverified': ['physical dose/vehicle reference compatibility', 'independent culture initiation'],
             'primary_claim_pass': False, 'new_confirmation_performed': False,
             'public_registration': False, 'confidence_intervals_computed': False}
    write_json(args.output / 'scope_before_fit.json', scope)
    for name in scope['code_sha256']:
        target = args.output / 'code_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    if args.matrix.stat().st_size > 80_000_000:
        raise ValueError('Compressed file exceeds fixed budget')
    started = time.monotonic()
    with gzip.open(args.metadata, 'rt', encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    counts, genes = load_matrix(args.matrix, [r[''] for r in rows])
    panel = nominal_panel(rows, counts, genes)
    del counts
    indices = reference_gene_panel(panel['singles'], genes)
    pair_ids = [json.dumps(pair, separators=(',', ':')) for pair in panel['pairs']]
    runs, pairs = len(panel['runs']), len(pair_ids)
    bank = {'singles': panel['singles'][:, :, :, indices].reshape(runs * pairs, 2, len(indices)),
            'responses': panel['responses'][:, :, indices].reshape(runs * pairs, len(indices)),
            'pair_ids': np.tile(pair_ids, runs),
            'culture_ids': np.repeat(panel['runs'], pairs)}
    with threadpool_limits(limits=1):
        report, predictions = nested_pair_benchmark(bank, menu)
    report.update({'scope_sha256': sha(args.output / 'scope_before_fit.json'),
                   'source_rows': len(rows), 'source_genes': len(genes),
                   'endpoint_genes': [genes[i] for i in indices],
                   'nominal_pair_count': pairs, 'reference_lineage': panel['lineage'],
                   'excluded_pairs': [list(p) for p in sorted(panel['all_pairs'] - set(panel['pairs']))],
                   'zero_library_ids': panel['zero_library_ids'],
                   'elapsed_seconds': time.monotonic() - started})
    # Local predictions support replay; expression arrays stay out of Git.
    np.savez_compressed(args.output / 'predictions.npz', pair_ids=bank['pair_ids'],
                        run_ids=bank['culture_ids'], responses=bank['responses'],
                        **{f'{label}_{role}': value for label, values in predictions.items()
                           for role, value in values.items()})
    report['prediction_archive_sha256'] = sha(args.output / 'predictions.npz')
    write_json(args.output / 'report.json', report)
    print(json.dumps({'pair_count': pairs, 'endpoint_genes': len(indices),
                      'equal_run_mean_mse': report['equal_run_mean_mse'],
                      'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
