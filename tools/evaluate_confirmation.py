"""Evaluate the EmeraldBay-frozen procedure on the isolated D-SPIN source."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, write_json, sha256
from opal.contextual.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config_path = ROOT / 'configs/datasets/dspin-confirmation-v1.json'
    config = read_json(config_path)
    result = run_benchmark(ROOT / 'data/prepared/dspin-confirmation-v1', args.output,
                           frozen_config=config['frozen_predictor'])
    skill = result['skill_over_development_selected_reference']
    write_json(args.output / 'confirmation_review.json', {
        'config_sha256': sha256(config_path), 'independent_source': config['source'],
        'frozen_procedure': config['frozen_predictor'], 'skill': skill,
        'material_skill_gate_passed': skill is not None and skill >= .05,
        'scope': config['scope'], 'biological_replication_established': False,
        'external_validation': 'independent study, conditional on published post-treatment lineage labels',
        'deployment_claim': 'none; the assay and endpoint differ from EmeraldBay',
        'next_gate': 'A crossed-context screen with independent biological replicates is still required.'})
    print(result['mse'], flush=True)


if __name__ == '__main__':
    main()
