"""Splits consume identities only, never expression."""
import numpy as np


def canonical_pair(a, b):
    a, b = str(a), str(b)
    if not a or not b or a == b:
        raise ValueError("Pairs require two distinct nonempty canonical target IDs")
    return tuple(sorted((a, b)))


def random_pair_split(pairs, seed=1729, test_fraction=0.3, folds=5):
    keys = [canonical_pair(*p) for p in pairs]
    unique = sorted(set(keys))
    if len(unique) < folds + 2:
        raise ValueError("Too few independent pairs for registered folds")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    ntest = max(1, int(np.ceil(len(unique) * test_fraction)))
    test_keys = {unique[i] for i in order[:ntest]}
    dev_keys = [unique[i] for i in order[ntest:]]
    if len(dev_keys) < folds:
        raise ValueError("Development split cannot support requested folds")
    fold_map = {p: j % folds for j, p in enumerate(dev_keys)}
    return {"seed": seed, "folds": folds,
            "development": [i for i, p in enumerate(keys) if p not in test_keys],
            "test": [i for i, p in enumerate(keys) if p in test_keys],
            "fold": {str(i): fold_map[p] for i, p in enumerate(keys) if p not in test_keys}}


def partner_splits(pairs, seed=2718, folds=5, minimum_development=12, minimum_test=4):
    keys = [canonical_pair(*p) for p in pairs]
    targets = sorted({t for p in keys for t in p})
    degree = {t: sum(t in p for p in set(keys)) for t in targets}
    rng = np.random.default_rng(seed)
    tie = dict(zip(targets, rng.random(len(targets))))
    groups, loads = [[] for _ in range(folds)], np.zeros(folds, int)
    for t in sorted(targets, key=lambda t: (-degree[t], tie[t])):
        f = int(np.argmin(loads))
        groups[f].append(t)
        loads[f] += degree[t]
    output = []
    for f, group in enumerate(groups):
        held = set(group)
        dev = [i for i, p in enumerate(keys) if not held.intersection(p)]
        test = [i for i, p in enumerate(keys) if held.intersection(p)]
        output.append({"fold": f, "seed": seed, "held_targets": group, "development": dev,
                       "test": test, "feasible": len(dev) >= minimum_development and len(test) >= minimum_test,
                       "two_unseen": sum(set(keys[i]).issubset(held) for i in test)})
    return output


def preserved_pair_split(pairs,development_pairs,protected_pairs,seed=1729,folds=5):
    """Retain old pair roles and split only newly eligible identities."""
    keys=[canonical_pair(*p) for p in pairs]
    old_dev={canonical_pair(*p) for p in development_pairs}
    protected={canonical_pair(*p) for p in protected_pairs}
    if old_dev&protected or not (old_dev|protected)<=set(keys):
        raise ValueError('Preserved pair roles overlap or reference missing identities')
    new=sorted(set(keys)-old_dev-protected)
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(new));ntest=int(np.ceil(.3*len(new)))
    test_keys=protected|{new[i] for i in order[:ntest]}
    dev_keys=sorted(set(keys)-test_keys)
    if len(dev_keys)<folds:raise ValueError('Too few development pairs')
    shuffled=[dev_keys[i] for i in rng.permutation(len(dev_keys))]
    fold_map={p:i%folds for i,p in enumerate(shuffled)}
    return {'seed':seed,'folds':folds,'development':[i for i,p in enumerate(keys) if p not in test_keys],
        'test':[i for i,p in enumerate(keys) if p in test_keys],
        'fold':{str(i):fold_map[p] for i,p in enumerate(keys) if p not in test_keys},
        'preserved_development_identities':len(old_dev),'protected_historical_holdout_identities':len(protected),
        'new_identity_holdouts':ntest,'policy':'old roles retained; only new pair identities randomly assigned'}
