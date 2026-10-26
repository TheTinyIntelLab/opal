import pytest

from tools.colo1_frontier_processing_v9 import project


def test_only_individual_operation_lines_are_projected():
    text = 'hidden <- secret\nx <- rowSums(train)\nnext <- secret\ny <- log2(x)\n'
    rows, rendered = project(text)
    assert rows == [(2, 'x <- rowSums(train)'), (4, 'y <- log2(x)')]
    assert 'secret' not in rendered


@pytest.mark.parametrize('line', ['# count 17', '[1] count 17', '2 count 17',
    'plot(counts)', 'print(counts)', 'counts <- 13', 'fitness = 0.3 # mean',
    'counts <- c(14, 18)', 'x <- mean(c(0.1, 0.4))',
    'p <- ggplot(counts)', 'write.table(counts, output)'])
def test_comments_outputs_rendering_and_numeric_literals_are_refused(line):
    assert project(line) == ([], '')


def test_caps_refuse_before_returning_a_partial_projection():
    with pytest.raises(ValueError, match='cap'):
        project('\n'.join(['x <- mean(train)'] * 129))
    with pytest.raises(ValueError, match='cap'):
        project('x <- mean(' + 'a' * 17000 + ')')
