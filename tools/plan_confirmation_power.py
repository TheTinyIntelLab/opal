"""Generate a small analytic sensitivity table, without experimental measurements."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal.artifacts import write_json, sha256
from opal.confirmation_planning import sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = sensitivity()
    result['helper_sha256'] = sha256(ROOT/'src/confirmation_planning.py')
    write_json(args.output, result)
    for row in result['rows']:
        print(row['standardized_excess_over_margin'], row['per_context_target_power'], row['independent_blocks_per_context'])


if __name__ == '__main__':
    main()
