"""Frozen positional-gene-table study, separate from native PertAdapt."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import zipfile
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from opal.artifacts import sha256, write_json, load_npz
from opal.models.kernel import single_kernel, symmetric_kernel, global_kernel
import improve_residual_model as base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", type=Path, default=Path("data/architecture_resources/OneDrive_1_10-5-2026"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shuffle-seed", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    checkpoint = torch.load(args.resources / "models.ckpt", map_location="cpu", weights_only=True, mmap=True)
    genes = (args.resources / "scfoundation_genes.txt").read_text().splitlines()
    weights = checkpoint["gene"]["state_dict"]["model.pos_emb.weight"].numpy()
    if len(set(genes)) != len(genes) or weights.shape[0] < len(genes):
        raise ValueError("Invalid positional gene table")
    index = {g: i for i, g in enumerate(genes)}
    provenance = {"checkpoint_sha256": sha256(args.resources / "models.ckpt"),
        "gene_order_sha256": sha256(args.resources / "scfoundation_genes.txt"),
        "wrapper_sha256": sha256(__file__), "shuffle_seed": args.shuffle_seed,
        "tensor": "gene/state_dict/model.pos_emb.weight", "seed": 20261005,
        "interpretation": "Frozen gene positions, not contextual embeddings or native PertAdapt",
        "pretraining_overlap": "Unknown", "candidate_count": 112, "evaluator_access": False,
        "fallback": "Unmatched targets have zero external kernel rows and columns",
        "mapping": "Exact symbols only, zero-based positions; no guessed aliases"}
    write_json(args.output / "foundation_protocol.json", provenance)
    original = base.candidates
    summary = {}
    with threadpool_limits(limits=1):
        for name in ["norman-log2", "wessels-log2"]:
            data = load_npz(Path("data/prepared") / name / "profiles.npz")
            available = np.array([t in index for t in data["targets"]], dtype=float)
            vectors = np.zeros((len(available), weights.shape[1]), np.float64)
            for i, target in enumerate(data["targets"]):
                if target in index:
                    vectors[i] = weights[index[target]]
            if args.shuffle_seed is not None:
                matched = np.flatnonzero(available)
                vectors[matched] = vectors[np.random.default_rng(args.shuffle_seed).permutation(matched)]
            write_json(args.output / (name + "_mapping.json"), {
                "targets": data["targets"].tolist(), "matched": int(available.sum()),
                "unmatched": data["targets"][available == 0].tolist()})
            # Supply the same extra kernels for development and unscored query reconstruction.
            def with_positions(singles, pairs):
                configs = original(singles, pairs)
                global_gram = global_kernel(singles, pairs)
                for bandwidth in [.25, 1, 4]:
                    gram = single_kernel(vectors, bandwidth=bandwidth) * np.outer(available, available)
                    summed = symmetric_kernel(gram, pairs, pairs)
                    product = symmetric_kernel(gram, pairs, pairs, interaction=True)
                    for kind, kernel in [("foundation_sum", summed), ("foundation_pair", product),
                                         ("global_plus_foundation_pair", global_gram + product)]:
                        for alpha in [.001, .01, .1, 1, 10]:
                            configs.append(({"kind": kind, "bandwidth": bandwidth, "alpha": alpha}, kernel))
                return configs
            base.candidates = with_positions
            summary[name] = base.study(Path("data/prepared") / name, args.output / name)
            with zipfile.ZipFile(args.output / name / "source_snapshot.zip", "a") as archive:
                archive.write(__file__, "tools/study_foundation_positions.py")
    base.candidates = original
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
