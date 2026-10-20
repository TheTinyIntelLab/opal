"""The source projector must isolate metadata from activity annotations."""
import csv
import io

import pytest

from tools.qualify_spidr_metadata import project_line


def test_projection_discards_outcome_and_matches_csv_for_selected_identity_fields():
    values = ['guide,a', 'GENE1', 'DO_NOT_RETAIN_OUTCOME', 'ACGT"TT', '', 'FALSE']
    buffer = io.StringIO()
    csv.writer(buffer).writerow(values)
    result = project_line(buffer.getvalue(), (0, 1, 3, 4, 5), 6)
    assert result == {0: 'guide,a', 1: 'GENE1', 3: 'ACGT"TT', 4: '', 5: 'FALSE'}
    assert 'DO_NOT_RETAIN_OUTCOME' not in repr(result)


def test_projection_refuses_width_changes_and_multiline_records():
    with pytest.raises(ValueError, match='width'):
        project_line('guide,GENE,ACGT\n', (0, 1), 4)
    with pytest.raises(ValueError, match='Multiline'):
        project_line('guide,"unfinished\n', (0,), 2)


def test_unselected_quoted_activity_field_cannot_shift_identity_columns():
    result = project_line('guide,"discard,this,""too""",GENE,ACGT\r\n', (0, 2, 3), 4)
    assert result == {0: 'guide', 2: 'GENE', 3: 'ACGT'}
