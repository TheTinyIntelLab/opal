"""Recover primary condition provenance without accessing expression outcomes."""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from opal.artifacts import read_json, write_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary', type=Path, required=True); p.add_argument('--harmonized', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    import anndata as ad
    f = ad.read_h5ad(args.harmonized, backed='r'); obs = f.obs; cells = pd.Index(obs.index)
    required = ['cell', 'sample', 'culture_plate', 'plate_oligo', 'well_oligo', 'cell_type', 'replicate',
                'time_point', 'dose', 'catalog_number', 'vehicle', 'product_name', 'target', 'pathway_level_1', 'pathway_level_2']
    samples, aliases, primary_seen, matched, mismatches, annotations = set(), 0, 0, 0, {}, 0
    seen = set()
    eligible_cells, assigned_cells, assigned_aliases = 0, 0, 0
    plate_mapping = set()
    conditions = set(); replica = set(); controls = set(); reagent_conditions = set()
    for chunk in pd.read_csv(args.primary, sep=' ', usecols=required, chunksize=50000):
        if time.perf_counter() - started > 180:
            raise RuntimeError('Primary metadata audit budget exceeded')
        primary_seen += len(chunk); chunk = chunk.loc[chunk.cell.isin(cells)]
        if not len(chunk):
            continue
        identities = set(chunk.cell)
        if chunk.cell.duplicated().any() or identities & seen:
            raise ValueError('Ambiguous primary cell identity')
        seen.update(identities)
        h = obs.loc[chunk.cell.to_numpy()]
        matched += len(chunk); samples.update(chunk['sample'].astype(str))
        aliases += int((chunk.culture_plate.astype(str) == chunk.plate_oligo.astype(str)).sum())
        for primary, harmonized in [('plate_oligo', 'plate'), ('well_oligo', 'well'), ('cell_type', 'cell_line'), ('replicate', 'replicate')]:
            a = chunk[primary].fillna('unassigned').astype(str).to_numpy()
            b = h[harmonized].astype(object).fillna('unassigned').astype(str).to_numpy()
            mismatches[primary] = mismatches.get(primary, 0) + int((a != b).sum())
        for primary, harmonized in [('time_point', 'time'), ('dose', 'dose_value')]:
            a = pd.to_numeric(chunk[primary], errors='coerce').to_numpy(); b = pd.to_numeric(h[harmonized], errors='coerce').to_numpy()
            mismatches[primary] = mismatches.get(primary, 0) + int((~np.isclose(a, b, equal_nan=True)).sum())
        for field in ['target', 'pathway_level_1', 'pathway_level_2']:
            a = chunk[field].fillna('nan').astype(str).to_numpy(); b = h[field].astype(object).fillna('nan').astype(str).to_numpy()
            annotations += int((a != b).sum())
        eligible = chunk[chunk.cell_type.isin(['A549', 'K562', 'MCF7']) & (chunk.time_point == 24)]
        eligible_cells += len(eligible)
        assigned = eligible[eligible.plate_oligo.notna()]
        assigned_cells += len(assigned)
        assigned_aliases += int((assigned.culture_plate == assigned.plate_oligo).sum())
        plate_mapping.update(map(tuple, assigned[['cell_type', 'culture_plate', 'plate_oligo']].to_numpy()))
        for row in eligible.itertuples(index=False):
            conditions.add((str(row.cell_type), str(row.culture_plate), str(row.well_oligo), str(row.catalog_number), float(row.dose)))
            replica.add((str(row.cell_type), str(row.replicate), str(row.sample)))
            reagent_conditions.add((str(row.product_name), str(row.catalog_number)))
            if str(row.vehicle).lower() == 'true':
                controls.add((str(row.cell_type), str(row.culture_plate)))
    f.file.close()
    if matched != len(obs) or any(mismatches.values()) or annotations:
        raise ValueError('Primary/harmonized metadata differ: ' + str(dict(matched=matched, expected=len(obs), mismatches=mismatches, annotations=annotations)))
    if assigned_aliases != assigned_cells:
        raise ValueError('Culture provenance does not alias the existing plate strata')
    write_json(args.output / 'assigned-plate-summary.json', dict(eligible_24h_cells=eligible_cells,
        assigned_plate_cells=assigned_cells, culture_plate_alias_cells=assigned_aliases,
        unique_line_plate_mapping=len(plate_mapping), response_access=False,
        interpretation='Every assigned 24h cell in the three qualified lines has culture_plate equal to existing plate_oligo. No additional biological unit is recovered.'))
    write_json(args.output / 'audit.json', dict(status='completed_metadata_provenance_only',
        primary_sha256=sha256(args.primary), harmonized_sha256=sha256(args.harmonized), source_cells=len(obs),
        primary_rows=primary_seen, matched_cells=matched, condition_mismatches=mismatches, annotation_mismatches=annotations,
        original_samples=sorted(samples), culture_plate_equals_existing_plate_oligo_cells=aliases,
        eligible_24h_line_culture_well_catalog_dose_conditions=len(conditions), line_culture_control_strata=len(controls),
        line_replicate_sample_records=[list(x) for x in sorted(replica)], reagent_mapping_count=len(reagent_conditions),
        expression_matrix_access=False, elapsed_seconds=time.perf_counter() - started,
        provenance='Primary GEO pData restores sample/catalog/culture nomenclature. Primary metadata not redistributed.',
        interpretation='Culture plate aliases existing plate strata; sample/run and rep1/rep2 identifiers alone do not establish independently initiated cultures. Randomized independently treated wells are documented, but no pair responses or sealed states are supplied.',
        missing=['Independently initiated culture provenance for the claimed biological sampling model', 'Matched pair responses', 'Sealed independent states and global pairs']))
    print('Matched primary metadata:', matched, 'cells; culture/plate aliases:', aliases, flush=True)


if __name__ == '__main__':
    import sys
    try:
        main()
    except Exception as error:
        if '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output') + 1])
            if output.exists():
                write_json(output / 'execution.json', dict(status='failed', exception=repr(error)))
        raise
