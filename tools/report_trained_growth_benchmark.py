"""Project the published frozen benchmark; never load outcomes or fit a model."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_RESULT = '55884ef35d4d558125058cd17624c6e1fb186dd77ba92a5b74da7aa1c6f88b36'


def project(source, output):
    source, output = Path(source), Path(output)
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != EXPECTED_RESULT:
        raise ValueError('This report requires the unchanged published result ledger')
    original = json.loads(payload)
    report = dict(
        contract='opal-trained-growth-benchmark-v1',
        source_result_sha256=EXPECTED_RESULT,
        population='original five disjoint test blocks pooled per task',
        deployment_relation='training-procedure benchmark; not a fold-0-only score',
        new_fits=0, new_scores=0, biological_confirmation=False,
        uncertainty='conditional gene sensitivity; not independent-study confidence',
        per_query_prediction_intervals='unavailable',
        program=original['program'], tasks={})
    for task, block in original['tasks'].items():
        models = block['models']
        simple = models['selected_simple']['common_support']['mean']['mse']
        policy = models['selected_policy']['common_support']['mean']['mse']
        report['tasks'][task] = dict(
            pairs=block['full_test_pairs'], genes=block['supported_genes'],
            coverage=block['coverage'], relative_mse_gain=1-policy/simple,
            # Preserve every comparator, infection target and population.
            models=models,
            # Preserve remaining gate and conditional-sensitivity fields verbatim.
            evidence={k: v for k, v in block.items() if k != 'models'})
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), tasks=len(report['tasks']), new_fits=0,
                          new_scores=0, program=report['program'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project(args.source, args.output)


if __name__ == '__main__':
    main()
