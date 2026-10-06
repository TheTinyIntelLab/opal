"""Bounded native precision probe and explicit full-gene resource requirements."""
import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, write_json, sha256, environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch.distributed as dist
    code = ROOT / 'tools/external/PertAdapt/scFoundation/PertAdapter/gears'
    package = ModuleType('opal_state_pertadapt_probe'); package.__path__ = [str(code)]
    sys.modules[package.__name__] = package
    for name in ['utils', 'model_new']:
        spec = importlib.util.spec_from_file_location(package.__name__ + '.' + name, code / (name + '.py'))
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    graph = args.output / 'synthetic-256-token-mask.npz'
    sparse.save_npz(graph, sparse.eye(256, format='csr'))
    records = []
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for batch in [1, 2]:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.manual_seed(41)
            try:
                adapter = module.PertAdapterNew(512, 8, graph_npz_file=str(graph)).cuda()
                x = torch.randn(batch, 256, 512, device='cuda', requires_grad=True)
                p0 = torch.randn(batch, 1, 512, device='cuda', requires_grad=True)
                with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
                    output = adapter(x, p0); loss = output.float().square().mean()
                loss.backward()
                records.append({'precision': str(dtype), 'batch': batch, 'forward_backward': 'passed',
                    'finite': bool(torch.isfinite(output).all() and torch.isfinite(x.grad).all()),
                    'peak_cuda_tensor_bytes': torch.cuda.max_memory_allocated()})
                del adapter, x, p0, output, loss
            except Exception as error:
                records.append({'precision': str(dtype), 'batch': batch, 'forward_backward': 'failed', 'error': str(error)[:800]})
    prior = read_json(ROOT / 'results/next/pertadapt-native-probe-v1/results.json')
    write_json(args.output / 'results.json', {'identity': 'pertadapt-synthetic-256-token-resource-probe',
        'canonical_score': None, 'native_method_changed': 'Synthetic 256-token mask for capacity probing only; no biological model or score',
        'upstream_module_sha256': sha256(code / 'model_new.py'), 'environment': environment(),
        'device': torch.cuda.get_device_name(), 'nccl_available': dist.is_nccl_available(),
        'precision_and_batching': records, 'full_gene_float32_prior': prior,
        'full_gene_attention_elements_per_batch': 8 * 19264 ** 2,
        'single_float32_attention_tensor_bytes': 4 * 8 * 19264 ** 2,
        'single_float16_attention_tensor_bytes': 2 * 8 * 19264 ** 2,
        'full_gene_execution': 'not attempted again; prior failure preserved',
        'alternative_compute': 'Linux CUDA/NCCL, full aligned 19264-gene input, at least 48 GB GPU as an initial float32 profiling target. Final training budget must be measured including backbone and optimizer.',
        'reduced_precision_limit': 'Microprobe success does not prove full-gene fit, training stability or canonical method equivalence',
        'main_context_comparator_contract': {
            'CPA': 'Existing known-category aggregate pair task retained. Native categorical context embeddings do not compute an unseen context from a baseline-only input; a declared adaptation is needed.',
            'GEARS': 'Existing aggregate genetic source development retained; the new chemical single-response task does not share its genetic intervention/graph contract.',
            'PertAdapt': 'Full foundation alignment, pretraining exposure and execution blockers remain. No canonical score supplied.'}})


if __name__ == '__main__':
    main()
