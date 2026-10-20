"""Validate external gamma preprocessing without reading biological outcomes."""
import csv
import gzip
import importlib.util
from pathlib import Path
import zipfile

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('external_preparation',
    Path(__file__).resolve().parents[1] / 'tools/prepare_external_fitness.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def screen():
    genes = np.array(['A', 'A', 'B', 'non-targeting'])
    first, second = np.indices((4, 4)).reshape(2, -1)
    counts = np.full((16, 4), 100.)
    counts[:, 2] = 210
    counts[:, 3] = 45
    pair = ((first < 2) & (second == 2)) | ((second < 2) & (first == 2))
    counts[pair, 2] = 45
    counts[pair, 3] = 100
    return genes, first, second, counts


def test_native_gamma_controls_doublings_and_gene_aggregation():
    genes, first, second, counts = screen()
    a, b, gamma, metadata = module.normalize_reference(genes, first, second, counts, [2., 4.])
    arrays = module.aggregate_genes(genes, a, b, gamma)
    expected = np.array([-1., np.log2(2.) / 4.])
    assert np.allclose(arrays['gamma'][0], expected)
    assert arrays['pair_names'].tolist() == [['A', 'B']]
    assert arrays['counts'].tolist() == [[4, 4]]
    assert metadata['double_nt_controls'] == 1
    assert np.allclose(gamma[(a == 3) & (b == 3)], 0.)
    assert arrays['single_genes'].tolist() == ['A', 'B']
    assert np.allclose(arrays['singles'], 0.)
    # Arbitrarily rescale total endpoint libraries: center cancels library size,
    # while the explicit pseudocount remains part of the measurement definition.
    direct = np.log2((counts[:, 2:] + 10) / (counts[:, :2] + 10))
    direct = (direct - direct[-1]) / [2., 4.]
    assert np.allclose(gamma, direct)


def test_t0_filter_boundary_and_endpoint_independence():
    genes, first, second, counts = screen()
    counts[(first == 0) & (second == 2), 0] = 50
    a, b, _, metadata = module.normalize_reference(genes, first, second, counts, [2., 4.])
    assert metadata['retained_constructs'] == 15
    assert not np.any((a == 0) & (b == 2))
    changed = counts.copy()
    changed[:, 2:] = 0  # endpoint dropout must not change coverage eligibility
    a2, b2, _, metadata2 = module.normalize_reference(genes, first, second, changed, [2., 4.])
    assert np.array_equal(a, a2) and np.array_equal(b, b2)
    assert metadata2['retained_constructs'] == 15
    # Guide QC averages replicate-specific medians; it does not threshold each
    # replicate median separately. Construct QC remains independently applied.
    counts = np.full((16, 4), 100.)
    counts[first == 0, 0] = 20
    counts[first == 0, 1] = 60
    _, _, _, metadata = module.normalize_reference(genes, first, second, counts, [1., 1.])
    assert metadata['excluded_guides'] == 0
    counts[first == 0, 1] = 50  # mean median is exactly35, excluded
    _, _, _, metadata = module.normalize_reference(genes, first, second, counts, [1., 1.])
    assert metadata['excluded_guides'] == 1


def test_requires_controls_replication_and_valid_measurements():
    genes, a, b, counts = screen()
    with pytest.raises(ValueError, match='controls'):
        module.normalize_reference(np.array(['A', 'A', 'B', 'C']), a, b, counts, [1., 1.])
    with pytest.raises(ValueError, match='doublings'):
        module.normalize_reference(genes, a, b, counts, [0., 1.])
    counts[0, 0] = np.inf
    with pytest.raises(ValueError, match='count'):
        module.normalize_reference(genes, a, b, counts, [1., 1.])
    # A single guide construct cannot become a gene-pair endpoint by duplicating
    # its two biological measurements or by including same-gene observations.
    with pytest.raises(ValueError, match='gene-pair'):
        module.aggregate_genes(genes, np.array([0, 0]), np.array([2, 1]), np.zeros((2, 2)))


@pytest.mark.parametrize('delimiter', ['\t', ','])
def test_environment_schema_delimiters_and_duplicate_guard(tmp_path, delimiter):
    path = tmp_path / 'counts.txt.gz'
    with gzip.open(path, 'wt', newline='') as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(['double', 'a', 'b', 'c', 'd'])
        writer.writerow(['A_guide:B_guide', 101, 102, 103, 104])
        writer.writerow(['B_guide:A_guide', 105, 106, 107, 108])
    names, genes, first, second, counts = module.read_counts(path, ['a', 'b', 'c', 'd'], 'environment')
    assert genes.tolist() == ['A', 'B']
    assert first.tolist() == [0, 1] and second.tolist() == [1, 0]
    assert counts.tolist() == [[101, 102, 103, 104], [105, 106, 107, 108]]
    with gzip.open(path, 'at', newline='') as stream:
        csv.writer(stream, delimiter=delimiter).writerow(['A_guide:B_guide', 101, 102, 103, 104])
    with pytest.raises(ValueError, match='duplicate'):
        module.read_counts(path, ['a', 'b', 'c', 'd'], 'environment')


def test_zip_schema_safe_arrays_and_source_integrity(tmp_path):
    source = tmp_path / 'counts.zip'
    text = 'FirstPosition\tFirstGene\tSecondPosition\tSecondGene\ta\tb\tc\td\nA_x\tA\tB_x\tB\t100\t100\t100\t100\n'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('sample_raw_counts.tsv', text)
    names, genes, a, b, counts = module.read_counts(source, ['a', 'b', 'c', 'd'], 'parp')
    assert names.tolist() == ['A_x', 'B_x'] and genes.tolist() == ['A', 'B']
    output = tmp_path / 'output'
    with pytest.raises(FileNotFoundError):
        module.prepare(tmp_path, output)
    assert not output.exists()
