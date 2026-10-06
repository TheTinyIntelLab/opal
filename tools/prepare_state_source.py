"""Qualify a corrected sci-Plex count release for bounded single-response transfer."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, write_json, sha256
from opal.contextual.contracts import ReferenceBank, SCHEMA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=ROOT / 'data/raw/state')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--maximum-interventions', type=int, default=24)
    p.add_argument('--genes', type=int, default=512)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Prepared data are immutable')
    config = read_json(ROOT / 'configs/datasets/sciplex-state.json')
    verified = read_json(args.raw / 'verified.json')
    raw = args.raw / config['filename']
    if sha256(raw) != verified['source_sha256']:
        raise ValueError('Verified source changed')
    import anndata as ad
    data = ad.read_h5ad(raw, backed='r')
    obs = data.obs
    required = {'cell_line', 'perturbation', 'dose_value', 'dose_unit', 'time', 'plate', 'well'}
    if not required <= set(obs):
        raise ValueError('Missing exact conditions or matched-control strata: ' + str(required - set(obs)))
    if set(obs['dose_unit'].astype(str)) != {'nM'}:
        raise ValueError('Unqualified dose units')
    biological_states = ['A549', 'K562', 'MCF7']
    doses = pd.to_numeric(obs['dose_value'], errors='raise').to_numpy()
    times = obs['time'].astype(str).to_numpy()
    line = obs['cell_line'].astype(str).to_numpy(); drug = obs['perturbation'].astype(str).to_numpy()
    plate = obs['plate'].astype(str).to_numpy(); well = obs['well'].astype(str).to_numpy()
    # The corrected release also contains unassigned cells and a 72-hour A549
    # follow-up. Neither enters this registered 24-hour cell-line experiment.
    eligible_cells = np.isin(line, biological_states) & (pd.to_numeric(obs['time'], errors='coerce').to_numpy() == 24.)
    control = (drug == 'control') & eligible_cells
    # One endpoint and exact 1 uM conditions; selection uses identities and coverage.
    valid_times = sorted(set(times[(doses == 1000.) & eligible_cells & ~control]))
    if len(valid_times) != 1:
        raise ValueError('Need one documented shared endpoint for this registered pilot')
    endpoint = valid_times[0]
    treatment = (doses == 1000.) & (times == endpoint) & eligible_cells & ~control
    control_strata = set(zip(line[control], plate[control]))
    eligible = {c: set(drug[treatment & (line == c)]) for c in biological_states}
    shared = sorted(set.intersection(*eligible.values()))[:args.maximum_interventions]
    if len(shared) < 4:
        raise ValueError('Insufficient exact intervention overlap')
    selected = np.isin(drug, shared) & treatment
    if any((c, p) not in control_strata for c, p in zip(line[selected], plate[selected])):
        raise ValueError('Treatment lacks line/plate-matched controls')
    context_keys = sorted(set(zip(line[selected], plate[selected])))
    contexts = [c + '@' + p for c, p in context_keys]
    control &= np.array([(c, pp) in set(context_keys) for c, pp in zip(line, plate)])
    gene_ids = data.var['ensembl_id'].astype(str).to_numpy() if 'ensembl_id' in data.var else data.var_names.astype(str).to_numpy()
    human = np.flatnonzero(np.char.startswith(np.asarray(gene_ids, dtype=str), 'ENSG'))
    if not len(human):
        raise ValueError('Human gene identifiers are required for this assay contract')
    # Query controls may choose a gene panel, but query treatment outcomes cannot.
    # This adaptation is declared; subsequent PCA and all scales are source-fold fitted.
    score = np.zeros(data.n_vars)
    retained_counts = Counter()
    for start in range(0, data.n_obs, 2048):
        end = min(start + 2048, data.n_obs)
        local = np.flatnonzero(control[start:end])
        if not len(local):
            continue
        block = data.X[start:end].tocsr()[local].astype(float)
        total = np.asarray(block[:, human].sum(1)).ravel()
        if np.any(total <= 0) or not np.isfinite(block.data).all() or np.any(block.data < 0) or np.any(block.data != np.floor(block.data)):
            raise ValueError('Control matrix is not nonnegative raw UMI counts')
        block = block.multiply(10000. / total[:, None]).tocsr()
        block.data = np.log2(1 + block.data)
        score += np.asarray(block.sum(0)).ravel()
    unique = np.flatnonzero(~pd.Index(gene_ids).duplicated() & np.char.startswith(np.asarray(gene_ids, dtype=str), 'ENSG'))
    columns = np.sort(unique[np.argsort(-score[unique], kind='stable')[:args.genes]])
    genes = np.asarray(gene_ids[columns].tolist(), dtype=str)
    keys = [(line[i], plate[i], well[i], 'control' if control[i] else drug[i])
            if control[i] or selected[i] else None for i in range(data.n_obs)]
    unique_keys = sorted(set(k for k in keys if k is not None)); mapping = {k: i for i, k in enumerate(unique_keys)}
    sums = np.zeros((len(mapping), len(columns))); sumsquares = sums.copy(); used = np.zeros(len(mapping), dtype=int)
    for start in range(0, data.n_obs, 2048):
        end = min(start + 2048, data.n_obs)
        active = [(i, mapping[k]) for i, k in enumerate(keys[start:end]) if k is not None]
        if not active:
            continue
        rows, indices = np.asarray(active).T
        block = data.X[start:end].tocsr()[rows].astype(float)
        total = np.asarray(block[:, human].sum(1)).ravel()
        if np.any(total <= 0) or np.any(block.data < 0) or not np.isfinite(block.data).all() or np.any(block.data != np.floor(block.data)):
            raise ValueError('Treatment matrix is not nonnegative raw UMI counts')
        values = np.log2(1 + block[:, columns].toarray() * (10000. / total[:, None]))
        np.add.at(sums, indices, values); np.add.at(sumsquares, indices, values ** 2); np.add.at(used, indices, 1)
    data.file.close()
    if np.any(used == 0):
        raise ValueError('Empty declared condition')
    means = sums / used[:, None]
    controls = np.zeros((len(contexts), len(columns)))
    effects = np.zeros((len(contexts), len(shared), len(columns)))
    available = np.zeros(effects.shape[:2], bool)
    precision_records = []
    for ci, (c, pp) in enumerate(context_keys):
        control_rows = [i for i, k in enumerate(unique_keys) if k[0] == c and k[1] == pp and k[3] == 'control']
        controls[ci] = means[control_rows].mean(0)
        for ti, d in enumerate(shared):
            treatment_rows = [i for i, k in enumerate(unique_keys) if k[0] == c and k[1] == pp and k[3] == d]
            if not treatment_rows:
                continue
            well_effects = means[treatment_rows] - controls[ci]
            effects[ci, ti] = well_effects.mean(0); available[ci, ti] = True
            precision_records.append({'context': c + '@' + pp, 'intervention': d, 'wells': len(treatment_rows),
                'cells': int(used[treatment_rows].sum()),
                'well_effect_dispersion': float(np.mean(np.var(well_effects, axis=0, ddof=1))) if len(well_effects) > 1 else None,
                'scope': 'Replicate-well sensitivity; culture/sample mapping removed by harmonization, not independent-culture uncertainty'})
    targets = np.array([d + '@1000nM:' + endpoint for d in shared])
    metadata = {'schema': SCHEMA, 'scale': 'log2', 'synthetic': False,
        'reference_access': 'controls_and_measured_singles_all_declared_contexts',
        'context_features_source': 'controls_and_metadata', 'feature_names': genes.tolist(),
        'gene_panel_uses_query_controls': True,
        'context_records': {c + '@' + pp: {'biological_state': c, 'selection_group': c, 'batch': pp, 'modality': 'chemical',
            'assay': 'sciPlex3_scRNAseq', 'study': 'Srivatsan2020', 'source': config['url']} for c, pp in context_keys},
        'intervention_records': {str(t): {'canonical_target': d, 'reagent': d, 'modality': 'chemical',
            'dose': '1000 nM', 'time': '24 hours'} for t, d in zip(targets, shared)},
        'normalization': 'Raw per-cell human-gene UMI total scaled to 10000, log2(1+x); equal well means minus line/plate-matched vehicle',
        'replication': 'Primary study randomizes replicate culture wells; harmonized release lacks culture/sample mapping. No independent culture or study confirmation.',
        'source_sha256': verified['source_sha256'], 'publisher_md5': config['publisher_md5'],
        'license': config['license'], 'prior_exposure': config['prior_exposure']}
    bank = ReferenceBank({'controls': controls, 'features': controls.copy(), 'singles': effects,
        'available': available, 'genes': genes, 'targets': targets, 'contexts': np.array(contexts)}, metadata)
    bank.save(args.output)
    write_json(args.output / 'source_audit.json', {'status': 'accepted_development_main_effect_only',
        'information_contract': 'All query single outcomes withheld per fold; baseline controls permitted; no combination data',
        'metadata_columns': list(obs.columns), 'source_cells': len(obs), 'contexts': contexts,
        'excluded_metadata_cells': int((~eligible_cells).sum()), 'biological_groups': biological_states,
        'human_gene_identifiers': len(human), 'retained_single_conditions': int(available.sum()),
        'retained_interventions': shared, 'endpoint': endpoint, 'dose': '1000 nM', 'genes': len(genes),
        'controls_matching': 'same cell line and plate; equal well means', 'query_control_gene_selection': True,
        'selection': 'first 24 exact-dose shared identities in lexicographic order, never selected by response magnitude',
        'precision': precision_records, 'qualification_limits': ['three independent line identities only',
            'catalog reagent and original culture IDs removed by harmonization', 'no combinations', 'public benchmark already widely exposed'],
        'primary_publication': config['primary_publication'], 'processing_sha256': verified['processing_sha256'],
        'source_sha256': verified['source_sha256'], 'publisher_md5_verified': config['publisher_md5']})
    print('Qualified main-effect development source:', args.output, flush=True)


if __name__ == '__main__':
    main()
