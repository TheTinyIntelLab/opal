"""Late, explicitly adapted SLIM references; never changes Opal selection.

Uses pinned upstream fitting code on signed condition-effect pseudoprofiles.
The upstream expression nonnegativity clip is omitted for this signed endpoint.
"""
import argparse
import contextlib
import io
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from opal.artifacts import read_json, write_json, load_npz, sha256, fingerprint


def score_existing(dataset, output):
    if (output / "evaluation.json").exists():
        raise FileExistsError("Reference evaluation already exists")
    fixed = read_json(output / "protocol.json")
    if sha256(dataset / "evaluator/profiles.npz") != fixed["evaluator_sha256"]:
        raise ValueError("Evaluator changed after registration")
    data = load_npz(dataset / "profiles.npz")
    query, evaluated = load_npz(output / "predictions.npz"), load_npz(dataset / "evaluator/profiles.npz")
    if not np.array_equal(query["pair_indices"], evaluated["pair_indices"]):
        raise ValueError("Evaluator pair order differs from frozen queries")
    from opal.evaluation import metrics
    pairs = data["query_pairs"]
    additive = data["singles"][pairs[:, 0]] + data["singles"][pairs[:, 1]]
    results = {track: metrics(query[track], evaluated["doubles"], additive)[0] for track in ["STRING64", "observed_single_PCA"]}
    write_json(output / "evaluation.json", {"results": results, "protocol_hash": fingerprint(fixed),
        "predictions_sha256": sha256(output / "predictions.npz"),
        "claim": "adapted reference, not native published-protocol replication"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["norman-log2", "wessels-log2", "norman-count", "wessels-count"])
    parser.add_argument("--score-existing", action="store_true", help="Recover scoring of frozen predictions; never refits or selects")
    args = parser.parse_args()
    dataset = ROOT / "data/prepared" / args.dataset
    output = ROOT / "results/public_references" / args.dataset
    if args.score_existing:
        score_existing(dataset, output)
        return
    if (output / "selection.json").exists():
        raise FileExistsError("Frozen reference selection exists; do not reopen its test")
    output.mkdir(parents=True, exist_ok=True)
    upstream = ROOT / "tools/external/SLIM"
    manifest = read_json(ROOT / "data/public_methods/slim_local_manifest.json")
    for entry in manifest["files"]:
        if sha256(upstream / entry["path"]) != entry["sha256"]:
            raise ValueError("Pinned upstream source changed")
    # Import only the native model modules; package __init__ eagerly imports
    # unused GEARS/Scanpy adapters. No model source is modified or stubbed.
    package = types.ModuleType("slim")
    package.__path__ = [str(upstream / "src/slim")]
    sys.modules["slim"] = package
    from slim.model_combo import BilinearComboModel
    from slim.config import SlimConfig
    from slim.embeddings import load_h5_embeddings
    data = load_npz(dataset / "profiles.npz")
    splits = read_json(dataset / "splits.json")
    folds = np.array([splits["fold"][str(i)] for i in data["pair_indices"]])
    targets = data["targets"].tolist()
    singles, doubles, pairs = data["singles"], data["doubles"], data["pairs"]
    evaluator_sha = sha256(dataset / "evaluator/profiles.npz")
    fixed = {"ranks": [2, 8, 16], "ridge": [0.1, 1, 10, 100], "folds": 5,
             "upstream": manifest["revision"], "adapter_sha256": sha256(Path(__file__)),
             "endpoint": "signed effects, same frozen genes and condition weighting",
             "adaptations": ["one condition-effect pseudoprofile per condition", "omit upstream nonnegative expression clipping", "two biases for singles and doubles"],
             "timing": "late supplementary reference; original Opal selections unchanged",
             "evaluator_sha256": evaluator_sha, "development_sha256": sha256(dataset / "profiles.npz")}
    write_json(output / "protocol.json", fixed)
    string_map, string_emb = load_h5_embeddings(upstream / "data/gene_string_embeddings.v0.3.h5", 64)
    embeddings = {"STRING64": (string_map, string_emb),
                  "observed_single_PCA": ({t: i for i, t in enumerate(targets)},
                                          PCA(n_components=min(16, len(targets) - 1), svd_solver="full").fit_transform(singles))}
    single_conditions = [t + "+ctrl" for t in targets]

    def fitted(train, rank, alpha, info):
        condition = single_conditions + [targets[a] + "+" + targets[b] for a, b in pairs[train]]
        matrix = np.vstack([singles, doubles[train]])
        adapter = types.SimpleNamespace(adata=ad.AnnData(matrix, obs=pd.DataFrame({"condition": condition}, index=[str(i) for i in range(len(condition))]), var=pd.DataFrame(index=data["genes"].tolist())))
        model = BilinearComboModel(config=SlimConfig(K=rank, lambda_reg=alpha, bias_method="two_bias"), genes2idx=info[0], embeddings=info[1])
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(adapter, condition)
        return model

    def prediction(model, query):
        feature = np.stack([model._mean_embedding([targets[a], targets[b]]) for a, b in query])
        return feature @ model.W.T @ model.G.T + model.b_double.ravel()

    winners = {}
    with threadpool_limits(limits=1):
        for track, info in embeddings.items():
            records = []
            for rank in fixed["ranks"]:
                for alpha in fixed["ridge"]:
                    oof = np.zeros_like(doubles)
                    for fold in sorted(set(folds)):
                        train, val = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                        np.random.seed(0)
                        model = fitted(train, rank, alpha, info)
                        oof[val] = prediction(model, pairs[val])
                    score = float(np.mean((oof - doubles) ** 2))
                    records.append({"rank": rank, "alpha": alpha, "cv_mse": score})
                    print(args.dataset, track, rank, alpha, score, flush=True)
            best = min(records, key=lambda r: (r["cv_mse"], r["rank"], r["alpha"]))
            winners[track] = {"selected": best, "search": records,
                              "missing_embedding_targets": [t for t in targets if t not in info[0]]}
        write_json(output / "selection.json", winners)
        query = {"pairs": data["query_pairs"], "pair_indices": data["query_indices"]}
        predictions = {}
        for track, info in embeddings.items():
            best = winners[track]["selected"]
            np.random.seed(0)
            model = fitted(np.arange(len(pairs)), best["rank"], best["alpha"], info)
            predictions[track] = prediction(model, query["pairs"])
        np.savez_compressed(output / "predictions.npz", pair_indices=query["pair_indices"], **predictions)
    if sha256(dataset / "evaluator/profiles.npz") != evaluator_sha:
        raise ValueError("Evaluator changed after registration")
    evaluated = load_npz(dataset / "evaluator/profiles.npz")
    if not np.array_equal(query["pair_indices"], evaluated["pair_indices"]):
        raise ValueError("Evaluator pair order differs from frozen queries")
    from opal.evaluation import metrics
    additive = singles[query["pairs"][:, 0]] + singles[query["pairs"][:, 1]]
    results = {track: metrics(pred, evaluated["doubles"], additive)[0] for track, pred in predictions.items()}
    write_json(output / "evaluation.json", {"results": results, "protocol_hash": fingerprint(fixed), "test_accesses": 1,
                                           "claim": "adapted reference, not native published-protocol replication"})


if __name__ == "__main__":
    main()
