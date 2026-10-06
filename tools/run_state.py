"""Run registered baseline-only main and source-development combination studies."""
import argparse
from pathlib import Path
import sys
import traceback
import shutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, write_json, load_npz, sha256
from opal.contextual.contracts import ReferenceBank
from opal.state.benchmark import run_main_benchmark
from opal.state.combinations import run_combinations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/experiments/state-development.json')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--combinations', action='store_true')
    p.add_argument('--only-combinations', action='store_true', help='Run corrected pair diagnostics without rerunning completed main experiments')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve old runs; use a new output directory')
    args.output.mkdir(parents=True)
    write_json(args.output / 'capacity.json', {'free_bytes': shutil.disk_usage(args.output).free,
        'expected_incremental_bytes': 100_000_000, 'prior_artifacts_deleted': False})
    if shutil.disk_usage(args.output).free < 250_000_000:
        raise OSError('Insufficient capacity for bounded training and checkpoint reserve')
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA environment is unavailable')
        torch.cuda.reset_peak_memory_stats()
    write_json(args.output / 'status.json', {'status': 'running', 'config_sha256': sha256(args.config)})
    try:
        bank = ReferenceBank.load(args.dataset)
        config = read_json(args.config)
        with threadpool_limits(limits=1):
            if not args.only_combinations:
                run_main_benchmark(bank, args.output / 'main', config, device=args.device)
            if args.combinations or args.only_combinations:
                # This reads source-development outcomes only, never evaluator labels.
                fitting = load_npz(args.dataset / 'profiles.npz')
                run_combinations(bank, fitting, args.output / 'combinations', config, device=args.device)
        write_json(args.output / 'status.json', {'status': 'complete_development', 'config_sha256': sha256(args.config),
            'historical_pair_evaluator_access': False, 'independent_confirmation': False})
        write_json(args.output / 'resources.json', {'device': args.device,
            'cuda_device': torch.cuda.get_device_name() if args.device == 'cuda' else None,
            'peak_cuda_tensor_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
            'output_bytes': sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()),
            'resource_scope': 'bounded state experiments; CUDA driver and CPU memory excluded'})
    except Exception as error:
        write_json(args.output / 'status.json', {'status': 'failed_preserved', 'error': str(error),
                                                'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
