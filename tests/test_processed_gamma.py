import csv
import gzip
import numpy as np
import pytest
from opal.processed_gamma import library, rows, collect_singles, aggregate, panel_predictions
from opal.scalar_screen import block_assignments


def guides():
    return {name: dict(index=i, gene=name.split('_')[0], sequence=seq,
        control=name.startswith('negative_control-'), transcript=None, mismatch=None)
        for i, (name, seq) in enumerate([('A_one','A'*20), ('A_two','C'*20),
            ('B_one','G'*20), ('negative_control-1','T'*20), ('negative_control-2','AC'*10)])}


def measurement(g, a, b, value):
    values = np.broadcast_to(np.asarray(value, dtype=float), (2, 2)).copy()
    return g[a], g[b], values


def controls(g):
    result = []
    for name in ('A_one', 'A_two', 'B_one'):
        for nt in ('negative_control-1', 'negative_control-2'):
            result += [measurement(g, name, nt, 1 if name == 'A_one' else 3),
                       measurement(g, nt, name, 2 if name == 'A_one' else 4)]
    return result


def write_source(path, records):
    header = [['']*16 for _ in range(3)]
    for context, cols in [('K562',(13,14)), ('Jurkat',(4,5))]:
        for rep, col in enumerate(cols,1):
            header[0][col], header[1][col], header[2][col] = context, 'Triple Sequencing', f'Replicate {rep}'
    with gzip.open(path, 'wt', newline='') as stream:
        csv.writer(stream, delimiter='\t').writerows(header+records)


def test_exact_library_and_missing_annotations(tmp_path):
    path = tmp_path/'library.gz'
    rows_ = [[name,name,name,'Y','unused',v['sequence'],'up'+name,'down'+name,'pLG18'] for name,v in guides().items()]
    with gzip.open(path,'wt',newline='') as f: csv.writer(f,delimiter='\t').writerows(rows_)
    parsed = library(path)
    assert len(parsed) == 5 and parsed['A_one']['transcript'] is None
    assert parsed['A_one']['mismatch'] is None
    assert parsed['negative_control-1']['control']
    with gzip.open(path,'at',newline='') as f: csv.writer(f,delimiter='\t').writerow(rows_[0])
    with pytest.raises(ValueError, match='Unique'): library(path)


def test_duplicate_and_unmatched_construct_rejected(tmp_path):
    path = tmp_path/'source.gz'
    record = ['A_one++B_one']+['']*15
    write_source(path,[record,record])
    with pytest.raises(ValueError,match='Duplicate'): list(rows(path,guides()))
    record[0] = 'missing++B_one'
    write_source(path,[record])
    with pytest.raises(ValueError,match='Unmatched'): list(rows(path,guides()))


def test_context_infection_columns_and_blanks(tmp_path):
    path = tmp_path/'source.gz'
    record = ['A_one++B_one']+['']*15
    record[13],record[14],record[4],record[5] = '0','2','3','nan'
    write_source(path,[record])
    value = list(rows(path,guides()))[0][2]
    assert value[0,0] == 0 and value[0,1] == 2 and value[1,0] == 3
    assert np.isnan(value[1,1])
    record[0] = 'A_one++B_one'
    write_source(path,[record[:10]])
    with pytest.raises(ValueError,match='Truncated'): list(rows(path,guides()))


def test_arm_specific_controls_only_and_context_infection():
    g = guides(); records = controls(g)
    records += [measurement(g,'A_one','B_one',999), measurement(g,'negative_control-1','negative_control-2',999)]
    records[0][2][0,1] = np.nan
    means,counts,tally = collect_singles(records,len(g))
    assert means[0,0,0,0] == 1 and means[0,0,0,1] == 2
    assert np.isnan(means[0,1,0,0]) and counts[0,1,0,0] == 1
    assert means[1,1,0,0] == 1 and tally['target_control_rows'] == 12
    assert np.isnan(means[:,:,3:]).all()


def test_sum_commutes_but_min_does_not():
    p = panel_predictions([0,4],[4,0])
    np.testing.assert_array_equal(p,[2,2,0,2,4])
    assert p[4] == p[0]+p[1]
    with pytest.raises(ValueError): panel_predictions([np.nan],[1])


def test_full_roster_exclusion_and_exact_matched_weights():
    g=guides(); single,_,_=collect_singles(controls(g),len(g))
    records=[measurement(g,'A_one','B_one',10),measurement(g,'B_one','A_two',20)]
    genes=np.array(['A','B','C']); pairs=np.array([[0,1],[0,2]])
    arr,tally=aggregate(records,genes,pairs,single)
    assert arr['means'].shape == (2,2,2,6)
    assert np.isnan(arr['means'][:,:,1]).all()
    np.testing.assert_array_equal(arr['counts'][:,:,0],2)
    np.testing.assert_allclose(arr['means'][:,:,0,0],15)
    np.testing.assert_allclose(arr['means'][...,4],arr['means'][...,1]+arr['means'][...,2],equal_nan=True)
    assert arr['orientation_counts'][0,0,0].tolist() == [1,1]
    single[0,0,g['A_one']['index'],0]=np.nan
    arr,_=aggregate(records,genes,pairs,single)
    assert arr['raw_counts'][0,0,0] == 2 and arr['counts'][0,0,0] == 1
    assert arr['missing_matched_single_counts'][0,0,0] == 1
    assert np.isnan(arr['means'][0,0,0]).all()


def test_canonical_roster_and_grouped_splits():
    g=guides(); single,_,_=collect_singles(controls(g),len(g))
    with pytest.raises(ValueError,match='Unique'):
        aggregate([],np.array(['A','B']),np.array([[0,1],[0,1]]),single)
    screen=dict(genes=np.array(['A','B','C']),pairs=np.array([[0,1],[0,2]]),accession='GSE116198')
    first=block_assignments(screen)
    for split,repeat in zip(first,block_assignments(screen)):
        np.testing.assert_array_equal(split,repeat)
        assert np.isin(split,[0,1,2]).all()
    # All ordered guides aggregate into the same single canonical-pair row.
    arr,_=aggregate([measurement(g,'A_one','B_one',1),measurement(g,'B_one','A_two',2)],
                    screen['genes'],screen['pairs'],single)
    assert arr['counts'][0,0].tolist() == [2,0]

def test_header_swap_rejected_before_values(tmp_path):
    path=tmp_path/'source.gz'
    write_source(path,[['A_one++B_one']+['0']*15])
    with gzip.open(path,'rt') as stream: content=stream.read()
    with gzip.open(path,'wt',newline='') as stream: stream.write(content.replace('Replicate 1','Replicate 9'))
    with pytest.raises(ValueError,match='header'):list(rows(path,guides()))


def test_double_labels_cannot_change_singles():
    g=guides()
    first=collect_singles(controls(g)+[measurement(g,'A_one','B_one',-999)],len(g))
    second=collect_singles(controls(g)+[measurement(g,'A_one','B_one',999)],len(g))
    np.testing.assert_array_equal(first[0],second[0])
    np.testing.assert_array_equal(first[1],second[1])
