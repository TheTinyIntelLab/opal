"""Consequential raw-identity and source/query gene-alignment guards."""
import gzip
import numpy as np
import pytest
from tools.run_adamson_transfer import canonical_construct, gene_alignment, joined_metadata


def test_exact_factorial_constructs_do_not_create_false_controls_or_triples():
    assert canonical_construct('3x_neg_ctrl_pMJ144-1')==()
    assert canonical_construct('ATF6_PERK_IRE1_pMJ158')==('ATF6','EIF2AK3','ERN1')
    with pytest.raises(ValueError):canonical_construct('ATF6_PSEUDO_NegCtrl')
    si,qi=gene_alignment(np.array(['g1','g2','g3']),np.array(['g3','g1','g4']))
    np.testing.assert_array_equal(si,[0,2]);np.testing.assert_array_equal(qi,[1,0])
    with pytest.raises(ValueError):gene_alignment(['g1'],['g1','g1'])


def test_full_barcode_suffix_preserves_distinct_drug_backgrounds(tmp_path):
    prefix='GSM2406677_10X005_'
    with gzip.open(tmp_path/(prefix+'barcodes.tsv.gz'),'wt') as f:f.write('same-1\nsame-2\nsame-3\n')
    with gzip.open(tmp_path/(prefix+'cell_identities.csv.gz'),'wt') as f:
        f.write('cell BC,guide identity,good coverage,number of cells\nsame-1,ATF6_only_pMJ145,True,1\nsame-2,PERK_only_pMJ146,True,1\nsame-3,3x_neg_ctrl_pMJ144-1,True,1\n')
    obs,good=joined_metadata(tmp_path)
    assert good.all() and len(obs)==3
    assert obs['gemgroup'].tolist()==['1','2','3']
