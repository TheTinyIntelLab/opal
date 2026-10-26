import pytest

from tools.colo1_frontier_processing_v10 import retain
from tools.colo1_frontier_processing_v9 import project


def test_v9_renderer_gap_is_preserved_as_a_known_diagnostic():
    assert project('p <- barplot(counts)')[0]
    assert not retain('p <- barplot(counts)')


@pytest.mark.parametrize('line', ['text(x,count,round(count,2))',
    "legend('topleft',mean(count))", "pdf('mean.pdf')", "main='raw counts'",
    "xlab='log2FC'", 'p <- ggplot(counts)', 'write.table(counts, path)'])
def test_renderers_and_quoted_only_matches_are_removed(line):
    assert not retain(line)


@pytest.mark.parametrize('line', ["x <- readRDS('processed.rds')",
    "m <- mean(x, na.rm=TRUE)", "counts <- apply(input,2,as.numeric)"])
def test_executable_operation_tokens_remain(line):
    assert retain(line)
