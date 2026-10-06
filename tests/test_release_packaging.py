"""Public source/evidence membership must not depend on private workspace files."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('release_packaging', Path(__file__).parents[1] / 'tools/package_research_release.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_source_rejects_private_and_external_inputs():
    for name in ['plan.md', 'docs/outreach.md', 'context.md', 'roadmap.md', 'docs/opal_next_master_plan.md', 'web/index.html',
                 'tools/external/CPA/model.py', 'tools/vendor/key.json',
                 'data/raw/source.json', 'results/state/model.npz', '../README.md']:
        assert not module.allowed_source(name), name
    for name in ['README.md', 'src/cli.py', 'configs/releases/v0.1.0.json', 'paper/main.pdf']:
        assert module.allowed_source(name), name


def test_evidence_keeps_negative_reports_but_excludes_banks():
    for name in ['results/next/dspin-confirmation-v1/results.json', 'results/state/capability_ledger.json',
                 'data/source_registry.json']:
        assert module.allowed_evidence(name), name
    for name in ['data/raw/measurement.json', 'data/prepared/source/profiles.npz',
                 'results/state/resume/optimizer.pt', 'results/next/demo-v1/summary.json',
                 '../results/test.json']:
        assert not module.allowed_evidence(name), name
