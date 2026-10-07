"""Inventory declared intervention order from metadata, without scoring outcomes."""
import argparse
from collections import Counter
from pathlib import Path
import pandas as pd
from opal.artifacts import write_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import anndata as ad
    norman = args.data_root / 'raw/NormanWeissman2019_filtered.h5ad'
    n = ad.read_h5ad(norman, backed='r'); orders = Counter(map(int, n.obs.nperts)); n.file.close()
    wessels = args.data_root / 'raw/source_metadata/wessels_metadata.tsv.gz'
    w = pd.read_csv(wessels, sep='\t', usecols=['CRISPR.Array', 'Guide.Class'])
    slots = Counter(w['CRISPR.Array'].astype(str).str.count(':').add(1).astype(int))
    write_json(args.output / 'inventory.json', dict(status='completed_bounded_metadata_inventory',
        response_access=False, sources=dict(
            norman=dict(sha256=sha256(norman), declared_nperts_cell_counts=dict(orders), genuine_order_three_or_more=any(k >= 3 for k in orders)),
            wessels=dict(sha256=sha256(wessels), array_slot_cell_counts=dict(slots), guide_classes=w['Guide.Class'].value_counts().to_dict(),
                         genuine_order_three_or_more=False, reason='Declared source constructs have two guide slots; guide capture/doublets cannot create a genuine registered triple intervention'),
            sciplex=dict(scope='Qualified publisher single-compound four-dose screen, not combinations'),
            emeraldbay=dict(scope='Qualified prepared pilot contains registered pair conditions, no qualified higher-order corpus')),
        higher_order_experiment='not registered or fitted: genuine triple corpus not qualified',
        needed=['Real simultaneous triple or higher construct identities', 'Exact matched constituent singles and declared lower-order measurements',
                'Reagent/dose/time/control alignment', 'Independent biological-unit coverage and source license/provenance'],
        limitation='This local inventory does not exhaust all public data or assert biological impossibility of higher-order prediction'))


if __name__ == '__main__':
    main()
