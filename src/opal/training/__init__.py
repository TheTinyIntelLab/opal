"""Full-batch condition-balanced fitting; evaluators are never imported here."""
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from opal.representations import fit_representation
from opal.models import ResidualNetwork
from opal.baselines import fit_baseline, predict_baseline

NEURAL = {"coordinate", "bilinear", "independent", "network", "sum_network"}


def fit(config, singles, pairs, doubles, seed=0, validation=None, single_partitions=None):
    if not np.isfinite(singles).all() or not np.isfinite(doubles).all():
        raise ValueError("Nonfinite fitting profiles")
    if len(pairs) != len(doubles) or len(pairs) == 0:
        raise ValueError("Empty or mismatched fitting pairs")
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if config['family'] == 'kernel':
        from opal.models.kernel import KernelResidual
        model = KernelResidual(singles, config).fit(pairs, doubles)
        return {'type': 'kernel', 'model': model, 'singles': singles, 'config': config,
                'epochs': 0, 'seed': seed,
                'parameters': model.coefficients.size + model.mean.size + model.column_mean.size}
    representation = fit_representation(singles, config.get("embedding", 16), config.get("representation", "pca"), seed, partitions=single_partitions)
    embedding = representation["embedding"]
    if config["family"] not in NEURAL:
        return {"type": "baseline", "model": fit_baseline(config, embedding, pairs, doubles, singles),
                "epochs": 0, "representation": representation, "seed": seed}
    additive = singles[pairs[:, 0]] + singles[pairs[:, 1]]
    target = doubles if config.get("direct", False) else doubles - additive
    basis = None
    if config.get("fixed_basis", False):
        _, _, vt = np.linalg.svd(target - target.mean(0), full_matrices=False)
        basis = vt.T
        if config.get("rank", 4) > basis.shape[1]:
            config = {**config, "requested_rank": config.get("rank", 4), "rank": basis.shape[1]}
    model = ResidualNetwork(embedding, singles.shape[1], config, basis)
    opt = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 0.01), weight_decay=config.get("weight_decay", 0.01))
    tx, ty = torch.as_tensor(pairs, dtype=torch.long), torch.as_tensor(target, dtype=torch.float64)
    original_embedding = torch.as_tensor(embedding, dtype=torch.float64)
    decoder = torch.as_tensor(np.linalg.pinv(embedding) @ (singles - representation["mean"]), dtype=torch.float64)
    best_state, best_error, best_epoch = None, float("inf"), 0
    epochs = config.get("epochs", 120)
    curve = []
    for epoch in range(epochs):
        model.train()
        prediction = model(tx)
        loss = (prediction - ty).square().mean()
        reconstruction = torch.zeros((), dtype=torch.float64)
        if config.get("finetune", False):
            reconstruction = (model.embedding @ decoder - torch.as_tensor(singles - representation["mean"], dtype=torch.float64)).square().mean()
            loss = loss + config.get("reconstruction_weight", 0.1) * reconstruction
        opt.zero_grad()
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite gradient")
        opt.step()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss")
        if validation is not None:
            vp, vy = validation
            with torch.no_grad():
                pred = model(torch.as_tensor(vp, dtype=torch.long)).numpy()
            if not config.get("direct", False):
                pred += singles[vp[:, 0]] + singles[vp[:, 1]]
            error = float(np.mean((pred - vy) ** 2))
            if error < best_error:
                best_error, best_epoch = error, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            best_epoch = epoch + 1
        if epoch % 10 == 0 or epoch == epochs - 1:
            curve.append({"epoch": epoch + 1, "loss": float(loss.detach()), "single_reconstruction": float(reconstruction.detach())})
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"type": "neural", "model": model, "singles": singles, "config": config, "seed": seed,
            "epochs": best_epoch, "curve": curve, "representation": representation,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def predict(fitted, pairs):
    if fitted['type'] == 'noise_kernel':
        return fitted['model'].predict(pairs)
    if fitted['type'] == 'prior_kernel':
        return fitted['model'].predict(pairs)
    if fitted['type'] == 'feature_ridge':
        return fitted['model'].predict(pairs)
    if fitted['type'] == 'advanced':
        return fitted['model'].predict(pairs, fitted['config'].get('mixture', 1))
    if fitted['type'] == 'kernel':
        return fitted['model'].predict(pairs)
    if fitted["type"] == "baseline":
        return predict_baseline(fitted["model"], pairs)
    with torch.no_grad():
        pred = fitted["model"](torch.as_tensor(pairs, dtype=torch.long)).numpy()
    if not fitted["config"].get("direct", False):
        pred += fitted["singles"][pairs[:, 0]] + fitted["singles"][pairs[:, 1]]
    return pred
