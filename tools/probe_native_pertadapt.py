"""Bounded upstream adapter feasibility probe, never a biological score."""
import argparse
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opal.artifacts import read_json, write_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import torch
    import torch.distributed as dist
    resources = ROOT / 'OneDrive_1_10-5-2026'
    code = ROOT / 'tools/external/PertAdapt/scFoundation/PertAdapter/gears'
    audit = read_json(ROOT / 'data/public_methods/pertadapt_resource_audit.json')
    for name in ['go_mask_19264.npz', 'scfoundation_genes.txt', 'models.ckpt']:
        if sha256(resources / name) != audit['files'][name]['sha256']:
            raise ValueError('Previously audited resource fingerprint changed')
    result = {'native_score': None, 'biological_inputs_used': False,
              'nccl_available': dist.is_nccl_available(), 'cuda_available': torch.cuda.is_available(),
              'model_sha256': sha256(code / 'model_new.py'), 'adapter_gene_tokens': 19264,
              'backbone_gene_order_verified_against_prepared_expression': False,
              'scope': 'Only a default float32 native adapter forward/backward feasibility probe',
              'remaining': ['Native distributed launcher requires NCCL, unavailable on this Windows build',
                            'Foundation gene symbols must be matched to full measured expression before a scored run',
                            'Backbone pretraining exposure and external graph access require a matched protocol']}
    write_json(args.output / 'protocol.json', result)
    package = ModuleType('opal_pinned_pertadapt')
    package.__path__ = [str(code)]
    sys.modules[package.__name__] = package
    try:
        for name in ['utils', 'model_new']:
            spec = importlib.util.spec_from_file_location(package.__name__ + '.' + name, code / (name + '.py'))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
        torch.set_num_threads(1); torch.manual_seed(20261005)
        adapter = mod.PertAdapterNew(512, 8, graph_npz_file=str(resources / 'go_mask_19264.npz')).cuda()
        expression = torch.randn(1, 19264, 512, device='cuda', requires_grad=True)
        perturbation = torch.randn(1, 1, 512, device='cuda', requires_grad=True)
        output = adapter(expression, perturbation)
        output.square().mean().backward()
        result.update({'adapter_probe': 'forward_backward_passed', 'finite': bool(torch.isfinite(output).all()),
                       'output_shape': list(output.shape), 'backbone_run': False})
    except Exception as error:
        result.update({'adapter_probe': 'failed', 'error_type': type(error).__name__, 'error': str(error)[:2500]})
    result['peak_cuda_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    result['cuda_total_bytes'] = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    write_json(args.output / 'results.json', result)
    print(result['adapter_probe'], result.get('error_type'), flush=True)


if __name__ == '__main__':
    main()
