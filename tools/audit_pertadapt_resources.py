"""Inspect supplied resources without importing upstream code or unsafe pickle."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
import csv
import numpy as np
import torch
from scipy import sparse
from opal.artifacts import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resources', type=Path, default=Path('data/architecture_resources/OneDrive_1_10-5-2026'))
    args = parser.parse_args()
    root = args.resources
    names = ['models.ckpt', 'go.csv', 'go_mask_19264.npz', 'scfoundation_genes.txt']
    record = {'files': {name: {'bytes': (root / name).stat().st_size,
               'sha256': sha256(root / name)} for name in names},
              'upstream_compatibility': 'not yet established', 'cuda_available': torch.cuda.is_available()}
    genes = (root / names[3]).read_text().splitlines()
    record['genes'] = {'count': len(genes), 'unique': len(set(genes)), 'empty': sum(not x for x in genes)}
    mask = sparse.load_npz(root / names[2])
    record['mask'] = {'shape': list(mask.shape), 'nonzeros': int(mask.nnz),
                      'finite': bool(np.isfinite(mask.data).all()), 'dtype': str(mask.dtype)}
    with (root / 'go.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        record['go'] = {'columns': reader.fieldnames, 'first_rows': [next(reader) for _ in range(3)]}
    # Refuse fallback to unrestricted pickle if this cannot load safely.
    try:
        checkpoint = torch.load(root / names[0], map_location='cpu', weights_only=True, mmap=True)
        def describe(value, depth=0):
            if isinstance(value, torch.Tensor):
                return {'shape': list(value.shape), 'dtype': str(value.dtype)}
            if isinstance(value, dict):
                if depth >= 2:
                    return {'type': 'dict', 'entries': len(value), 'first_keys': list(map(str, list(value)[:12]))}
                return {str(k): describe(v, depth + 1) for k, v in value.items()}
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            return {'type': type(value).__name__}
        record['checkpoint'] = describe(checkpoint)
        record['backbones'] = {}
        for name, entry in checkpoint.items():
            if isinstance(entry, dict) and 'state_dict' in entry:
                state = entry['state_dict']
                record['backbones'][name] = {
                    'tensor_elements': sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor)),
                    'position_embedding_shape': list(state['model.pos_emb.weight'].shape),
                    'model': entry['config']['model'],
                    'model_config': entry['config']['model_config'][entry['config']['model']]}
        record['safe_load'] = 'passed'
    except Exception as error:
        record['safe_load'] = 'failed'
        record['error'] = str(error)
    write_json('data/public_methods/pertadapt_resource_audit.json', record)
    print(record['safe_load'], record['genes'], record['mask'], flush=True)


if __name__ == '__main__':
    main()
