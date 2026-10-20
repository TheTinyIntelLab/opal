"""Stream official Horlbeck guide-pair growth measurements into gene-pair means."""
import argparse
import csv
import gzip
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import sha256, write_json

SOURCE_HASH = 'ba9772f3b0e929bcf5b23bebf17d4caab86c7f2ccd50dbd62fbc47e40c2aac21'


def prepare(source, output):
    if output.exists():
        raise FileExistsError('Preserve previous preparations')
    if sha256(source) != SOURCE_HASH:
        raise ValueError('Unexpected official source bytes')
    started = time.perf_counter()
    # First pass reads identities only; eligibility never uses growth magnitude.
    genes = set()
    with gzip.open(source, 'rt') as stream:
        rows = csv.reader(stream, delimiter='\t')
        headers = [next(rows) for _ in range(3)]
        columns = [[13, 14], [4, 5]]
        for context, indexes in zip(['K562', 'Jurkat'], columns):
            for rep, col in enumerate(indexes, 1):
                if [h[col] for h in headers] != [context, 'Triple Sequencing', f'Replicate {rep}']:
                    raise ValueError('Source schema differs')
        for row in rows:
            if not row or '++' not in row[0]:
                continue
            genes.update(g.split('_')[0] for g in row[0].split('++'))
    genes = sorted(genes - {'negative'})
    index = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    sums = np.zeros((2, 2, n, n)); counts = np.zeros_like(sums, dtype=np.int32)
    single_sums = np.zeros((2, 2, n)); single_counts = np.zeros_like(single_sums, dtype=np.int32)
    source_rows = 0
    with gzip.open(source, 'rt') as stream:
        rows = csv.reader(stream, delimiter='\t')
        for _ in range(3): next(rows)
        for row in rows:
            if not row or '++' not in row[0]: continue
            a, b = [g.split('_')[0] for g in row[0].split('++')]
            source_rows += 1
            for c, indexes in enumerate(columns):
                for r, col in enumerate(indexes):
                    try: value = float(row[col])
                    except ValueError: continue
                    if not np.isfinite(value): continue
                    if (a == 'negative') != (b == 'negative'):
                        i = index[b if a == 'negative' else a]
                        single_sums[c,r,i] += value; single_counts[c,r,i] += 1
                    elif a != 'negative' and b != 'negative' and a != b:
                        i, j = sorted((index[a], index[b]))
                        sums[c,r,i,j] += value; counts[c,r,i,j] += 1
    keep = np.all(single_counts >= 20, axis=(0,1))
    # At least four ordered guide-pair observations in EACH infection replicate
    # and context. This is a coverage rule, not an effect-size filter.
    supported = np.all(counts >= 4, axis=(0,1)) & keep[:,None] & keep[None,:]
    a, b = np.where(np.triu(supported, 1))
    if len(a) < 10000: raise ValueError('Insufficient compatible pair supervision')
    singles = single_sums / np.maximum(single_counts, 1)
    reps = sums[:,:,a,b] / counts[:,:,a,b]
    output.mkdir(parents=True)
    np.savez_compressed(output/'profiles.npz', genes=np.asarray(genes), pairs=np.stack([a,b],1),
        singles=singles, responses=reps.mean(1), replicates=reps,
        counts=counts[:,:,a,b], single_counts=single_counts, contexts=np.array(['K562','Jurkat']))
    receipt = {'source_sha256':SOURCE_HASH, 'source_url':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116198/suppl/GSE116198_sgRNA_pair_phenotypes.txt.gz',
        'source_rows':source_rows, 'genes':n,'eligible_single_genes':int(keep.sum()),'pairs':len(a),
        'contexts':['K562','Jurkat'], 'target':'mean measured gamma over guide pairs/orientations and two infection replicates',
        'input':'measured gene-negative-control growth means; no published GI scores or correlations',
        'eligibility':'single counts >=20 and pair counts >=4 in every context and replicate; no magnitude filter',
        'profiles_sha256':sha256(output/'profiles.npz'),'seconds':time.perf_counter()-started,
        'preparer_sha256':sha256(__file__), 'source_licensing':'Author dataset CC BY 4.0; GEO mirror exact identity not verified; public research reuse with attribution, no raw redistribution'}
    write_json(output/'preparation.json', receipt)
    print(receipt, flush=True)
    return receipt


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.output)
