"""Public source/evidence membership must not depend on private workspace files."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('release_packaging', Path(__file__).parents[1] / 'tools/package_research_release.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_source_rejects_private_and_external_inputs():
    for name in ['plan.md', 'PLAN.md', 'docs/Plan.MD', 'docs/OUTREACH.md', 'docs/outreach.md', 'context.md', 'roadmap.md', 'docs/opal_next_master_plan.md', 'web/index.html',
                 'tools/external/CPA/model.py', 'tools/vendor/key.json',
                 'data/raw/source.json', 'results/state/model.npz', '../README.md', 'C:/README.md', 'src/../README.md', 'src\\cli.py']:
        assert not module.allowed_source(name), name
    assert not module.allowed_source('docs/frontier/historical/plan-20261015-before-confirmation-phase.md.txt')
    assert not module.allowed_source('docs/frontier/historical/plan-20261015-before-evidence-audit.md')
    assert not module.allowed_source('docs/frontier/outreach_register.json')
    for name in ['README.md', 'src/cli.py', 'configs/releases/v0.1.0.json', 'paper/main.pdf']:
        assert module.allowed_source(name), name


def test_evidence_keeps_negative_reports_but_excludes_banks():
    for name in ['results/next/dspin-confirmation-v1/results.json', 'results/state/capability_ledger.json',
                 'data/source_registry.json', 'results/stage1-20261007/report-v2/comparisons.csv']:
        assert module.allowed_evidence(name), name
    for name in ['data/raw/measurement.json', 'data/prepared/source/profiles.npz',
                 'results/state/resume/optimizer.pt', 'results/next/demo-v1/summary.json',
                 '../results/test.json', 'results/PLAN.md', 'results/raw/summary.json','results/measurements.csv']:
        assert not module.allowed_evidence(name), name


def test_v1_keeps_weights_separate_from_source_release():
    import json
    root = Path(__file__).parents[1]
    recipe = json.loads((root / 'configs/releases/v1.0.0.json').read_text())
    assert recipe['version'] == '1.0.0' and recipe['models'] == {}
    for name in ['RELEASE_CLEANUP_PLAN.md', 'RELEASE_DELETIONS.json', 'OPAL_REPOSITORY_AUDIT.md',
                 'src/checkpoint.npz', 'paper/model.pt', 'results/growth/checkpoint.npz']:
        assert not module.allowed_source(name), name
        assert not module.allowed_evidence(name), name
    assert module.allowed_source('docs/archive/pre-v1-documentation.zip')
    assert not module.allowed_source('docs/archive/unreviewed-models.zip')
