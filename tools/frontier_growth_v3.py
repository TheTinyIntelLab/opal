"""Exposed-growth adapter v3. The scientific menu is the frozen v1 menu.

Checkpoints for completion models use the separate lossless packing contract.
Every other checkpoint stays on the original predictor contract. Historical
block assignment, labels, gate, and caps are unchanged. This module does not
modify those implementations, open source banks at import, or impute singles.

Genes absent from the eligible pair roster are collapsed before fitting.
Every eligible pair row is kept. A nonfinite single on a gene that occurs in
any eligible pair fails qualification.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tools.frontier_tournament as tournament_tool
from opal.evaluation_precision import metrics_v3
from opal.frontier_packing_v2 import PACKING_CONTRACT, MixedLoader, load_checkpoint, save_checkpoint
from opal.frontier_tournament import (CONTRACT, Predictor, fit_models, make_world, pair_order, select_records)
from opal.label_budget import budget_split
from opal.scalar_screen import (EXTERNAL_HASHES, EXTERNAL_PREPARATION_HASH, ORIGINAL_HASH,
    ORIGINAL_SOURCE_HASH, TASKS, block_assignments, paired_gene_uncertainty)

torch.set_num_threads(1)

_HELPER = (ROOT / 'tools' / 'frontier_tournament.py').resolve()
if Path(tournament_tool.__file__).resolve() != _HELPER:
    raise ImportError('Frontier helper must be the checkout tools module')

PROTOCOL_ID = 'opal-frontier-growth-v3'
REGISTERED_PROCEDURES = 30
TRAIN_LABELS = 16000
VALIDATION_LABELS = 4000
FIT_SEED = 261401
PROFILE_SEED = 261501
PROFILE_GENES = (467, 600)
PROFILE_WORLDS = ('additive', 'lowrank')
PROFILE_QUERY_COUNT = 10000
COMPLETION_CHECKPOINTS_PER_PROCEDURE = 4
PROFILE_BOUND_GENES = 600
# Aligned completion plus three shuffled completions, float64 n-by-n, 30 procedures.
# 4 * 600 * 600 * 8 * 30 = 345_600_000 raw bytes, before predictions or the profile tree.
RAW_COMPLETION_BYTES = (COMPLETION_CHECKPOINTS_PER_PROCEDURE * PROFILE_BOUND_GENES
    * PROFILE_BOUND_GENES * 8 * REGISTERED_PROCEDURES)
EVALUATION_SEEDS = frozenset((261101, 261102, 261103, 261001))
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261022
PROGRAM_CAP = dict(wall_seconds=7200, cpu_seconds=7200, rss_bytes=2*1024**3,
    artifact_bytes=512*1024**2, reserve_bytes=1024**3+32*1024**2)
PROFILE_CAP = dict(wall_seconds=600, cpu_seconds=600, rss_bytes=2*1024**3,
    artifact_bytes=64*1024**2, reserve_bytes=1024**3+32*1024**2)
if RAW_COMPLETION_BYTES != 345_600_000 or RAW_COMPLETION_BYTES <= 256*1024**2:
    raise RuntimeError('Completion storage arithmetic drifted')
if PROGRAM_CAP['artifact_bytes'] < RAW_COMPLETION_BYTES:
    raise RuntimeError('Program artifact cap does not cover the raw completion matrices')
if PROFILE_SEED in EVALUATION_SEEDS or FIT_SEED in EVALUATION_SEEDS:
    raise RuntimeError('Profile and fit seeds overlap the evaluation namespace')

_BOUND = (
    'tools/frontier_growth.py',
    'tools/frontier_growth_v2.py',
    'tools/frontier_growth_v3.py',
    'src/frontier_packing_v2.py',
    'tests/test_frontier_packing_v2.py',
    'docs/frontier_execution/growth_shuffle_registration.md',
    'results/frontier-execution-20261026/packed-qualification-v1.json',
    'tests/test_frontier_growth.py',
    'tests/test_frontier_growth_v2.py',
    'tests/test_frontier_packing.py',
    'src/frontier_packing.py',
    'src/frontier_tournament.py',
    'tools/frontier_tournament.py',
    'docs/frontier_execution/Grok_packing_v1.md',
    'docs/frontier_execution/growth_packing_registration.md',
    'src/block_completion.py',
    'src/models/fitness.py',
    'src/evaluation_precision.py',
    'src/evaluation_contracts.py',
    'src/scalar_screen.py',
    'src/label_budget.py',
    'src/artifacts.py',
    'docs/frontier_execution/growth_proposal.md',
    'docs/frontier_execution/experiment_matrix.md',
    'docs/frontier_execution/Grok_growth_critique.md',
    'docs/frontier_execution/growth_registration.md',
    'docs/model_development_20261020.md',
    'docs/research_reset/label_budget_protocol.md',
    'docs/research_reset/block_protocol.md',
)
_COMMAND = re.compile(
    r'python tools/study_block_completion\.py freeze --original ([^\s`]+) '
    r'--prepared ([^\s`]+) --output ([^\s`]+)')


class QualificationError(ValueError):
    """Stop before fitting. Do not impute singles or delete pairs."""


def digest(path):
    return tournament_tool.digest(path)


def read(path):
    return tournament_tool.read(path)


def write(path, obj):
    tournament_tool.write(path, _plain(obj))


def save_arrays(path, **arrays):
    tournament_tool.save_arrays(path, **arrays)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _now():
    return datetime.now(timezone.utc).isoformat()


def _bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def protocol_data_paths():
    """Paths from the one freeze command in the historical block protocol."""
    text = (ROOT / 'docs/research_reset/block_protocol.md').read_text(encoding='utf-8')
    found = _COMMAND.findall(text)
    if len(found) != 1:
        raise ValueError('Block protocol data paths are not uniquely declared')
    original, prepared, historical_output = found[0]
    return dict(original=original, prepared=prepared, historical_block_output=historical_output)


def bound_files():
    notes = sorted((ROOT / 'docs/frontier_execution').glob('Grok_growth*.md'))
    packing_notes = sorted((ROOT / 'docs/frontier_execution').glob('Grok_packing*.md'))
    if not notes:
        raise ValueError('Growth critique note is missing')
    if not packing_notes:
        raise ValueError('Packing review note is missing')
    files = [ROOT / name for name in _BOUND]
    for note in [*notes, *packing_notes]:
        if note not in files:
            files.append(note)
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        raise ValueError('Bound file missing: ' + ', '.join(missing))
    return files


def bank_expectations(paths):
    return dict(original_root=paths['original'], prepared_root=paths['prepared'],
        historical_block_output=paths['historical_block_output'],
        original_hash=ORIGINAL_HASH, original_source_hash=ORIGINAL_SOURCE_HASH,
        external_preparation_hash=EXTERNAL_PREPARATION_HASH,
        external_hashes=dict(EXTERNAL_HASHES), banks_opened=False)


def _binding_record(protocol_path):
    files = bound_files()
    return dict(protocol_sha256=digest(protocol_path),
        files={path.relative_to(ROOT).as_posix(): digest(path) for path in files},
        helper=str(_HELPER), model_contract=CONTRACT,
        environment=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
            threads=torch.get_num_threads()),
        utc=_now())


def verify_binding(root):
    root = Path(root)
    binding = read(root / 'code-binding.json')
    if digest(root / 'protocol.json') != binding['protocol_sha256']:
        raise ValueError('Protocol drift')
    for name, expected in binding['files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Code binding drift: ' + name)
    env = binding['environment']
    if env['numpy'] != np.__version__ or env['torch'] != torch.__version__ or env['python'] != sys.version:
        raise ValueError('Numerical environment drift')
    if env['threads'] != 1 or torch.get_num_threads() != 1:
        raise ValueError('Numerical thread drift')
    if binding.get('model_contract') != CONTRACT:
        raise ValueError('Model contract drift')
    return read(root / 'protocol.json')


def _refuse_failures(root):
    if any(Path(root).glob('failure-*.json')):
        raise FileExistsError('Preserve failed experiment')


def _refuse_markers(root, names):
    _refuse_failures(root)
    for name in names:
        if (Path(root) / name).exists():
            raise FileExistsError('Preserve prior stage output: ' + name)


def collapse_eligible_genes(genes, pairs, singles):
    """Reindex genes that occur in the eligible pair roster.

    Responses are not an argument. Unused genes are dropped, including when
    their singles are nonfinite. Eligible pair rows stay in order.
    """
    names = np.asarray([str(gene) for gene in np.asarray(genes)])
    pairs = np.asarray(pairs)
    singles = np.asarray(singles)
    if names.ndim != 1 or len(set(names)) != len(names):
        raise QualificationError('Unique gene names required for qualification')
    if any((not name) or name != name.strip() or '|' in name for name in names):
        raise QualificationError('Canonical gene names required for qualification')
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise QualificationError('Integer eligible pairs required for qualification')
    if not len(pairs) or np.any(pairs < 0) or np.any(pairs >= len(names)):
        raise QualificationError('Eligible pair endpoint is outside the gene roster')
    if singles.shape != (len(names),) or singles.dtype.kind not in 'iuf':
        raise QualificationError('Single vector does not match the gene roster')
    values = np.array(singles, dtype=float, copy=True)
    present = np.zeros(len(names), dtype=bool)
    present[pairs[:, 0]] = True
    present[pairs[:, 1]] = True
    kept = np.flatnonzero(present)
    if not np.isfinite(values[kept]).all():
        raise QualificationError(
            'Eligible-pair gene has a nonfinite single; qualification failed '
            'without imputation or pair deletion')
    mapping = np.full(len(names), -1, dtype=np.int64)
    mapping[kept] = np.arange(len(kept), dtype=np.int64)
    new_pairs = mapping[np.asarray(pairs, dtype=np.int64)]
    if np.any(new_pairs < 0) or len(new_pairs) != len(pairs):
        raise QualificationError('Collapse dropped eligible pairs')
    if np.any(new_pairs[:, 0] >= new_pairs[:, 1]):
        raise QualificationError('Reindex broke canonical endpoint order')
    kept_names = names[kept]
    before = names[np.asarray(pairs, dtype=np.int64)]
    after = kept_names[new_pairs]
    if not np.array_equal(before, after):
        raise QualificationError('Collapse changed eligible pair identity')
    return dict(genes=kept_names, pairs=new_pairs, singles=values[kept].copy(),
        mapping=mapping, dropped_names=[str(names[i]) for i in np.flatnonzero(~present)],
        pair_names_before=before, pair_names_after=after)


def _budget_hashes(screen):
    genes, pairs = screen['genes'], screen['pairs']
    accession = screen['accession']
    return [hashlib.sha256(
        f'opal-budget-v1|{accession}|{genes[a]}|{genes[b]}'.encode()).hexdigest()
        for a, b in pairs]


def prefix_budget(screen, original, train_labels=TRAIN_LABELS, validation_labels=VALIDATION_LABELS):
    """Charge the historical hash prefix. Do not replace a charged row."""
    original = np.asarray(original)
    membership = budget_split(screen, original, train_labels, validation_labels)
    hashes = _budget_hashes(screen)
    selected = {}
    for part, count, key in ((0, train_labels, 'train'), (1, validation_labels, 'validation')):
        rows = np.flatnonzero(original == part)
        order = sorted(rows.tolist(), key=lambda row: hashes[row])
        prefix = np.asarray(order[:count], dtype=np.int64)
        got = np.flatnonzero(membership == part)
        if not np.array_equal(np.sort(got), np.sort(prefix)) or len(got) != count:
            raise QualificationError('Budget prefix was repaired or reordered')
        if any(int(membership[row]) == part for row in order[count:]):
            raise QualificationError('Budget prefix included an uncharged row')
        selected[key] = prefix.copy()
    test = np.flatnonzero(membership == 2)
    if not np.array_equal(test, np.flatnonzero(original == 2)):
        raise QualificationError('Budget changed the original test block')
    pairs = np.asarray(screen['pairs'])
    degree = np.bincount(pairs[selected['train']].ravel(), minlength=len(screen['genes']))
    two = degree >= 2
    val_supported = two[pairs[selected['validation']]].all(1)
    test_supported = two[pairs[test]].all(1) if len(test) else np.zeros(0, dtype=bool)
    accounting = dict(nominal_train=int(train_labels), nominal_validation=int(validation_labels),
        charged_train=int(len(selected['train'])), charged_validation=int(len(selected['validation'])),
        charged_pair_union=int(len(selected['train']) + len(selected['validation'])),
        unsupported_validation_charged=int((~val_supported).sum()),
        two_partner_validation=int(val_supported.sum()), full_test=int(len(test)),
        two_partner_test=int(test_supported.sum()), replaced_rows=0,
        selection_includes_fallback=True)
    return dict(train=selected['train'], validation=selected['validation'], test=test,
        test_supported=test_supported, accounting=accounting)


def build_procedure_arrays(screen, original, train_labels=TRAIN_LABELS,
                           validation_labels=VALIDATION_LABELS):
    """Budget on original rows, then collapse unused genes. Test rows stay."""
    budget = prefix_budget(screen, original, train_labels, validation_labels)
    collapsed = collapse_eligible_genes(screen['genes'], screen['pairs'], screen['singles'])
    responses = np.asarray(screen['responses'], dtype=float)
    replicates = np.asarray(screen['replicates'], dtype=float)
    n_pairs = len(collapsed['pairs'])
    if responses.shape != (n_pairs,) or replicates.shape != (n_pairs, 2):
        raise QualificationError('Eligible pair targets do not match the roster')
    if not np.isfinite(responses).all() or not np.isfinite(replicates).all():
        raise QualificationError('Nonfinite eligible pair target; pairs are not dropped')
    test = budget['test']
    return dict(pairs=collapsed['pairs'], singles=collapsed['singles'], genes=collapsed['genes'],
        dropped_names=np.asarray(collapsed['dropped_names'], dtype='U'),
        train_index=budget['train'], validation_index=budget['validation'], test_index=test,
        train_targets=responses[budget['train']].copy(),
        validation_targets=responses[budget['validation']].copy(),
        pair_names=collapsed['pair_names_after'],
        sealed_mean=responses[test].copy(),
        sealed_infection0=replicates[test, 0].copy(),
        sealed_infection1=replicates[test, 1].copy(),
        accounting=dict(budget['accounting'], collapsed_genes=int(len(collapsed['genes'])),
            dropped_unused_genes=int(len(collapsed['dropped_names'])),
            test_pairs=int(len(test))))


def write_procedure(root, task, fold, arrays):
    root = Path(root)
    proc = root / 'procedures' / task / f'fold-{fold}'
    sealed = root / 'sealed-test' / f'{task}__fold-{fold}.npz'
    proc.mkdir(parents=True, exist_ok=False)
    sealed.parent.mkdir(parents=True, exist_ok=True)
    if sealed.exists():
        raise FileExistsError(sealed)
    save_arrays(proc / 'fit-args.npz', pairs=arrays['pairs'], singles=arrays['singles'],
        gene_names=np.asarray(arrays['genes']), dropped_names=arrays['dropped_names'],
        train_index=np.asarray(arrays['train_index'], dtype=np.int64),
        validation_index=np.asarray(arrays['validation_index'], dtype=np.int64),
        test_index=np.asarray(arrays['test_index'], dtype=np.int64),
        train_targets=arrays['train_targets'], validation_targets=arrays['validation_targets'])
    save_arrays(sealed, test_index=np.asarray(arrays['test_index'], dtype=np.int64),
        pair_names=np.asarray(arrays['pair_names'])[arrays['test_index']],
        mean=arrays['sealed_mean'], infection0=arrays['sealed_infection0'],
        infection1=arrays['sealed_infection1'])
    write(proc / 'accounting.json', arrays['accounting'])
    with np.load(proc / 'fit-args.npz', allow_pickle=False) as loaded:
        forbidden = {'mean', 'infection0', 'infection1', 'responses', 'observed', 'test_targets'}
        if forbidden & set(loaded.files):
            raise QualificationError('Fit arguments contain sealed target payloads')
        if not np.isfinite(loaded['singles']).all():
            raise QualificationError('Fit singles are nonfinite; qualification failed without imputation')


def snapshot_files(root, directory):
    base = Path(root) / directory
    out = {}
    if not base.exists():
        return out
    for path in sorted(p for p in base.rglob('*') if p.is_file()):
        out[path.relative_to(root).as_posix()] = digest(path)
    return out


def assert_snapshot(root, expected):
    for name, digest_value in expected.items():
        path = Path(root) / name
        if not path.is_file() or digest(path) != digest_value:
            raise ValueError('Sealed input drift: ' + name)


def expected_persisted(controls):
    return 28 if controls else 19


def _label(record):
    return record['family'] + '--' + record['config']


def kept_models(records):
    selected, simple, policy = select_records(records)
    kept = [record for record in records if str(record['family']).startswith('B')]
    kept += [selected[name] for name in selected if str(name).startswith('I')]
    labels = [_label(record) for record in kept]
    if len(labels) != len(set(labels)):
        raise ValueError('Duplicate persisted model label')
    return kept, simple, policy, selected


def shuffled_policies(records):
    chosen = {}
    for index in range(3):
        group = [record for record in records if str(record['family']).endswith(f'-shuffle{index}')]
        if len(group) != 5:
            raise ValueError('Incomplete shuffled family')
        winner = min(group, key=lambda record: record['validation_mse'])
        chosen[f'shuffle{index}'] = _label(winner)
    return chosen


def _permutable(records):
    found = {}
    for record in records:
        family = str(record['family'])
        if not family.startswith('I0'):
            continue
        key = 'aligned' if 'shuffle' not in family else family.split('-', 1)[1]
        found[key] = float(record['permutable'])
    return found


class Guard:
    def __init__(self, root, cap):
        self.root, self.cap = Path(root), cap
        self.t0, self.c0 = time.perf_counter(), time.process_time()
        self.peak = 0
        self.prior_wall = self.prior_cpu = 0.
        qualification = self.root / 'packed-qualification.json'
        self.retained_bytes = int(read(qualification).get('retained_profile_bytes', 0)) if qualification.is_file() else 0

    def add_prior(self, names):
        for name in names:
            path = self.root / name
            if path.exists():
                record = read(path)
                self.prior_wall += float(record['wall_seconds'])
                self.prior_cpu += float(record['cpu_seconds'])
                self.peak = max(self.peak, int(record.get('peak_rss_bytes', 0)))

    def check(self):
        wall = self.prior_wall + time.perf_counter() - self.t0
        cpu = self.prior_cpu + time.process_time() - self.c0
        info = psutil.Process().memory_info()
        self.peak = max(self.peak, int(getattr(info, 'peak_wset', info.rss)))
        if wall > self.cap['wall_seconds'] or cpu > self.cap['cpu_seconds']:
            raise RuntimeError(f'Cumulative scientific wall/CPU cap: wall={wall:.3f} cpu={cpu:.3f}')
        if self.peak > self.cap['rss_bytes']:
            raise RuntimeError(f'Process memory cap: peak={self.peak}')
        if shutil.disk_usage(self.root).free < self.cap['reserve_bytes']:
            raise RuntimeError('Disk reserve plus working allowance')
        size = _bytes(self.root) + self.retained_bytes
        if size > self.cap['artifact_bytes']:
            raise RuntimeError(f'Artifact cap: bytes={size}')
        return dict(wall_seconds=wall, cpu_seconds=cpu, peak_rss_bytes=self.peak, artifact_bytes=size)

    def finish(self, stage):
        observed = self.check()
        write(self.root / (stage + '-resource.json'), dict(
            wall_seconds=time.perf_counter() - self.t0, cpu_seconds=time.process_time() - self.c0,
            peak_rss_bytes=self.peak, artifact_bytes=observed['artifact_bytes'],
            cumulative_wall_seconds=observed['wall_seconds'],
            cumulative_cpu_seconds=observed['cpu_seconds'], threads=torch.get_num_threads()))


def fit_procedure(proc, *, controls=True, completion_steps=500, neural_steps=300,
                  seed=FIT_SEED, check=None):
    """Fit from fit-args only. Sealed test targets are not a parameter."""
    if check is not None:
        check()
    proc = Path(proc)
    with np.load(proc / 'fit-args.npz', allow_pickle=False) as loaded:
        forbidden = {'mean', 'infection0', 'infection1', 'responses', 'observed', 'test_targets'}
        if forbidden & set(loaded.files):
            raise QualificationError('Fit arguments contain sealed target payloads')
        pairs = loaded['pairs'].copy()
        singles = loaded['singles'].copy()
        gene_names = loaded['gene_names'].copy()
        train = loaded['train_index'].copy()
        val = loaded['validation_index'].copy()
        test = loaded['test_index'].copy()
        train_targets = loaded['train_targets'].copy()
        validation_targets = loaded['validation_targets'].copy()
    if not np.isfinite(singles).all():
        raise QualificationError('Fit singles are nonfinite; qualification failed without imputation')
    require_schema_bounds(len(singles), len(test), where=str(proc))
    if set(map(int, train)) & set(map(int, test)) or set(map(int, val)) & set(map(int, test)):
        raise QualificationError('Test rows entered the fit split')
    records = fit_models(singles, singles[:, None], pairs[train], train_targets, pairs[val],
        validation_targets, seed, controls=controls, completion_steps=completion_steps,
        neural_steps=neural_steps)
    kept, simple, policy, selected = kept_models(records)
    if len(kept) != expected_persisted(controls):
        raise RuntimeError('Persisted roster length drifted')
    if controls:
        shuffle_names = shuffled_policies(records)
        permutable = _permutable(records)
        if any(permutable.get(f'shuffle{i}', 0.) < .8 for i in range(3)):
            raise QualificationError('Uninformative shuffle; no stratum rescue')
    else:
        shuffle_names, permutable = {}, {}
    checkpoint_dir = proc / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=False)
    outputs, models = {}, []
    query = pairs[test]
    for index, record in enumerate(kept):
        name = _label(record)
        path = checkpoint_dir / f'model-{index:02d}.npz'
        save_checkpoint(record['model'], path)
        prediction, supported = tournament_tool.predictions(record['model'], query, singles)
        outputs[f'prediction_{index}'] = prediction
        models.append(dict(name=name, family=record['family'], checkpoint=path.name,
            checkpoint_sha256=digest(path), prediction_key=f'prediction_{index}'))
    save_arrays(proc / 'queries.npz', pairs=query, singles=singles, gene_names=gene_names,
        test_index=np.asarray(test, dtype=np.int64), supported=supported)
    save_arrays(proc / 'predictions.npz', **outputs)
    write(proc / 'selection.json', dict(selected_simple=_label(simple), selected_policy=_label(policy),
        family_selections={name: _label(record) for name, record in selected.items()},
        shuffled_policies=shuffle_names, permutable=permutable, models=models,
        configurations=[{key: value for key, value in record.items() if key != 'model'}
            for record in records],
        controls=bool(controls), seed=int(seed)))
    if check is not None:
        check()
    return proc


def take_charged_prefix(data, n_train, n_val):
    """Synthetic profile prefix. Unavailable charged rows are not replaced."""
    charged, used = {}, {}
    for role, count in (('train', n_train), ('validation', n_val)):
        rows = np.asarray(data[role], dtype=np.int64)
        ordered = rows[pair_order(data['pairs'][rows], 'frontier-budget-' + role)]
        if len(ordered) < count:
            raise ValueError(f'{role} partition cannot supply the profile budget')
        chosen = np.asarray(ordered[:count], dtype=np.int64)
        if len(np.intersect1d(chosen, ordered[count:])):
            raise ValueError('Profile prefix replaced a charged row')
        charged[role] = chosen
        available = np.asarray(data['available'], dtype=bool)
        used[role] = chosen[available[chosen]]
    return charged, used


def synthetic_holdout_count(genes):
    """Held-out size from the frozen make_world 50/20/30 split. Labels are not used."""
    count = int(genes) * (int(genes) - 1) // 2
    return count - (count // 2 + count // 5)


def profile_test_queries(data, count=PROFILE_QUERY_COUNT):
    """Fixed metadata-only hash prefix of the synthetic TEST partition.

    The count is the registered profile budget. It is not replaced by a
    biological fold size, and rows after the prefix are not appended.
    """
    if count != PROFILE_QUERY_COUNT:
        raise ValueError('Profile query count is fixed at 10000')
    rows = np.asarray(data['test'], dtype=np.int64)
    ordered = rows[pair_order(data['pairs'][rows], 'frontier-budget-test')]
    if len(ordered) < count:
        raise ValueError('Synthetic test partition cannot supply 10000 profile queries')
    chosen = np.asarray(ordered[:count], dtype=np.int64)
    if len(chosen) != count or len(np.intersect1d(chosen, ordered[count:])):
        raise ValueError('Profile query prefix was extended or replaced')
    return chosen


def require_schema_bounds(n_genes, n_test, *, where):
    """Stop before a fit when the schema exceeds the profiled ceiling.

    Genes are not dropped and the test is not truncated. The profile budget
    is not raised.
    """
    genes, tests = int(n_genes), int(n_test)
    if genes > PROFILE_BOUND_GENES:
        raise QualificationError(
            'Eligible gene count exceeds the fixed profile schema bound of '
            f'{PROFILE_BOUND_GENES} at {where}: {genes}; genes are not dropped '
            'and the profile is not extended')
    if tests > PROFILE_QUERY_COUNT:
        raise QualificationError(
            'Test count exceeds the fixed profile query budget of '
            f'{PROFILE_QUERY_COUNT} at {where}: {tests}; the test is not '
            'truncated and the profile is not extended')


def require_biological_test_bound(counts):
    """Refuse a fold whose full test exceeds the profile query budget.

    The biological test is not truncated, and the profile budget is not raised.
    """
    over = [(task, int(fold), int(n)) for task, fold, n in counts if int(n) > PROFILE_QUERY_COUNT]
    if over:
        raise QualificationError(
            'Biological test count exceeds the fixed profile query budget of '
            f'{PROFILE_QUERY_COUNT}; the test is not truncated and the profile is not extended: {over}')
    return counts


def _write_synthetic_procedure(root, name, data, train, val, test):
    proc = Path(root) / 'profile' / name
    sealed = Path(root) / 'profile-sealed' / f'{name}.npz'
    proc.mkdir(parents=True, exist_ok=False)
    sealed.parent.mkdir(parents=True, exist_ok=True)
    pairs, singles = data['pairs'], data['singles']
    observed = data['observed']
    gene_names = np.asarray([f's{i:04d}' for i in range(len(singles))])
    save_arrays(proc / 'fit-args.npz', pairs=pairs, singles=singles, gene_names=gene_names,
        dropped_names=np.asarray([], dtype='U'),
        train_index=np.asarray(train, dtype=np.int64),
        validation_index=np.asarray(val, dtype=np.int64),
        test_index=np.asarray(test, dtype=np.int64),
        train_targets=np.asarray(observed[0, train], dtype=float),
        validation_targets=np.asarray(observed[0, val], dtype=float))
    save_arrays(sealed, test_index=np.asarray(test, dtype=np.int64),
        pair_names=gene_names[pairs[test]],
        mean=np.asarray(observed[0, test], dtype=float),
        infection0=np.asarray(observed[0, test], dtype=float),
        infection1=np.asarray(observed[1, test], dtype=float))


def forecast_program(measurements, profile_bytes):
    needed = {(world, genes) for world in PROFILE_WORLDS for genes in PROFILE_GENES}
    seen = {(record['world'], int(record['genes'])) for record in measurements}
    result = dict(margin=1.25, procedures=REGISTERED_PROCEDURES, profile_bytes=int(profile_bytes),
        measurements=measurements, biological_fits=0)
    if seen != needed:
        result.update(qualified=False, blocker='profile roster incomplete')
        return result
    if any(int(record['persisted_models']) != 28 or int(record['threads']) != 1 for record in measurements):
        result.update(qualified=False, blocker='profile did not persist 28 single-thread models')
        return result
    wall = max(float(record['wall_seconds']) for record in measurements) * REGISTERED_PROCEDURES * 1.25
    cpu = max(float(record['cpu_seconds']) for record in measurements) * REGISTERED_PROCEDURES * 1.25
    artifacts = int(profile_bytes) + max(int(record['artifact_bytes']) for record in measurements) * REGISTERED_PROCEDURES * 1.25
    rss = max(int(record['peak_rss_bytes']) for record in measurements)
    ok = wall <= 7200 and cpu <= 7200 and artifacts <= PROGRAM_CAP['artifact_bytes'] and rss <= PROGRAM_CAP['rss_bytes']
    result.update(qualified=bool(ok), blocker=None if ok else 'forecast exceeds the registered program cap',
        forecast_wall_seconds=wall, forecast_cpu_seconds=cpu, forecast_artifact_bytes=artifacts,
        forecast_peak_rss_bytes=rss, artifact_cap_bytes=PROGRAM_CAP['artifact_bytes'],
        raw_completion_bytes=RAW_COMPLETION_BYTES, profile_query_count=PROFILE_QUERY_COUNT)
    return result


def require_qualified_forecast(record):
    if not record.get('qualified'):
        raise RuntimeError('Forecast failed; biological fitting is stopped. ' + str(record.get('blocker')))


def freeze(root):
    root = Path(root)
    if root.exists():
        raise FileExistsError('Preserve prior growth registrations')
    root.mkdir(parents=True, exist_ok=False)
    (root / '.gitignore').write_text(
        '*.npz\n**/*.npz\nprocedures/\nsealed-test/\nprofile/\nprofile-sealed/\nprofile-packed/\n',
        encoding='utf-8')
    paths = protocol_data_paths()
    protocol = dict(id=PROTOCOL_ID, model_contract=CONTRACT, synthetic_only=False,
        biological_access='fit stage only, after qualified forecast',
        tasks=list(TASKS), folds=5, train_labels=TRAIN_LABELS, validation_labels=VALIDATION_LABELS,
        budget='opal-budget-v1|accession|geneA|geneB full sha256 prefix inside original block split',
        support='two distinct TRAIN partners; fixed measured single sum fallback',
        collapse='drop genes absent from the eligible pair roster; fail if an eligible gene single is nonfinite',
        fit_seed=FIT_SEED, profile_seed=PROFILE_SEED, profile_genes=list(PROFILE_GENES),
        profile_worlds=list(PROFILE_WORLDS), profile_queries=PROFILE_QUERY_COUNT,
        max_eligible_genes=PROFILE_BOUND_GENES,
        packing_contract=PACKING_CONTRACT,
        completion_storage=('exact float64 bitwise-symmetric upper triangle with reversible byte planes; '
            'non-completion checkpoints remain on the original predictor contract'),
        profile_reuse='qualify-packed reads a finished v1 synthetic profile and does not refit',
        raw_completion_bytes=RAW_COMPLETION_BYTES, bootstrap_draws=BOOTSTRAP_DRAWS,
        bootstrap_seed=BOOTSTRAP_SEED, program_cap=PROGRAM_CAP, profile_cap=PROFILE_CAP,
        primary='validation-selected full policy on the equal-infection mean; infections are score-only',
        gate='all six tasks; one failure fails the program; no rescue',
        data_paths=paths, utc=_now())
    write(root / 'protocol.json', protocol)
    write(root / 'bank-expectations.json', bank_expectations(paths))
    write(root / 'code-binding.json', _binding_record(root / 'protocol.json'))
    print(dict(frozen=True, output=str(root), banks_opened=False))


def profile(root):
    raise RuntimeError(
        'Growth v2 does not refit the synthetic profile; '
        'use qualify-packed --prior-profile on the measured v1 profile')


def _replay_directory(proc, *, refit, predictor=None):
    """Replay saved checkpoints. The predictor class is the imported model, not a refit.

    A process whose PYTHONPATH imports opal from an external byte-identical install
    uses that install. The checkout tool is not required to be an installed package.
    """
    if refit:
        raise RuntimeError('Replay refit is refused')
    predictor = active_predictor(historical_roots=(Path(proc).resolve(),)) if predictor is None else predictor
    choice = read(proc / 'selection.json')
    with np.load(proc / 'queries.npz', allow_pickle=False) as queries:
        pairs, singles = queries['pairs'], queries['singles']
    with np.load(proc / 'predictions.npz', allow_pickle=False) as saved:
        for record in choice['models']:
            model = predictor.load(proc / 'checkpoints' / record['checkpoint'], record['checkpoint_sha256'])
            prediction, _ = tournament_tool.predictions(model, pairs, singles)
            if not np.array_equal(prediction, saved[record['prediction_key']]):
                raise ValueError('Prediction replay mismatch')
    return len(choice['models'])


def fit(root):
    root = Path(root)
    verify_binding(root)
    _refuse_markers(root, ('fit-started.json', 'inputs-sealed.json', 'all-predictions-sealed.json',
        'results.json', 'scoring-start.json'))
    if not (root / 'profile-qualification.json').exists():
        raise ValueError('Qualified profile forecast is required before biological fitting')
    qualification = read(root / 'profile-qualification.json')
    if qualification.get('protocol_sha256') != digest(root / 'protocol.json'):
        raise ValueError('Qualification protocol drift')
    require_qualified_forecast(qualification)
    expectations = read(root / 'bank-expectations.json')
    # Source loading, validation and input sealing consume the same scientific
    # budget as fitting, rather than disappearing before the clock starts.
    guard = Guard(root, PROGRAM_CAP)
    write(root / 'fit-started.json', dict(utc=_now(), protocol_sha256=digest(root / 'protocol.json'),
        qualification_sha256=digest(root / 'profile-qualification.json')))
    from opal.scalar_screen import load_screen_roster
    screens = load_screen_roster(expectations['original_root'], expectations['prepared_root'])
    if set(screens) != set(TASKS):
        raise QualificationError('Roster tasks differ from the historical six')
    jobs = []
    try:
        assignments, counts = [], []
        for task in TASKS:
            guard.check()
            screen = screens[task]
            expected_hash = ORIGINAL_HASH if task.startswith('horlbeck') else expectations['external_hashes'][task]
            if screen['bank_sha256'] != expected_hash:
                raise QualificationError('Loaded bank hash differs from the frozen expectation')
            splits = block_assignments(screen)
            if len(splits) != 5:
                raise QualificationError('Five historical blocks required')
            assignments.append(splits)
            counts.extend((task, fold, int((np.asarray(split) == 2).sum())) for fold, split in enumerate(splits))
        require_biological_test_bound(counts)
        counted = {(task, fold): n for task, fold, n in counts}
        for task, splits in zip(TASKS, assignments, strict=True):
            for fold, split in enumerate(splits):
                guard.check()
                arrays = build_procedure_arrays(screens[task], split, TRAIN_LABELS, VALIDATION_LABELS)
                require_schema_bounds(len(arrays['genes']), len(arrays['test_index']),
                    where=f'{task} fold {fold}')
                if len(arrays['test_index']) != counted[(task, fold)]:
                    raise QualificationError('Biological test rows were truncated')
                write_procedure(root, task, fold, arrays)
                jobs.append((task, fold))
                del arrays
    finally:
        del screens
    sealed = snapshot_files(root, 'sealed-test')
    args = snapshot_files(root, 'procedures')
    write(root / 'inputs-sealed.json', dict(sealed_test=sealed, fit_args=args, utc=_now()))
    for task, fold in jobs:
        assert_snapshot(root, sealed)
        fit_procedure(root / 'procedures' / task / f'fold-{fold}', check=guard.check)
    assert_snapshot(root, sealed)
    write_prediction_manifest(root, jobs, expected=REGISTERED_PROCEDURES)
    guard.finish('fit')
    print(dict(fitted=True, procedures=len(jobs), test_metrics=False))


def _pool_metric(target, prediction):
    if not len(target):
        return None
    scored = metrics_v3(target, prediction)
    return dict(mse=scored['mse'], mae=scored['mae'], bias=scored['bias'],
        correlation=scored['pearson'], pairs=scored['pairs'])


def _model_metrics(targets, prediction, support):
    """Mean and both infections, on common support and on the full test block."""
    def block(mask):
        return {key: _pool_metric(targets[key][mask], prediction[mask])
            for key in ('mean', 'infection0', 'infection1')}
    full = np.ones(len(support), dtype=bool)
    return dict(common_support=block(support), full_population=block(full))


def write_prediction_manifest(root, jobs, expected):
    root = Path(root)
    manifest = dict(procedures=[], files={}, biological_confirmation=False)
    for task, fold in jobs:
        manifest['procedures'].append(dict(task=task, fold=fold, path=f'procedures/{task}/fold-{fold}'))
    for directory in ('procedures', 'sealed-test'):
        for path in (root / directory).rglob('*'):
            if path.is_file():
                manifest['files'][path.relative_to(root).as_posix()] = digest(path)
    if len(manifest['procedures']) != expected:
        raise RuntimeError('Partial prediction roster')
    write(root / 'all-predictions-sealed.json', manifest)
    return manifest


def _concat_named(folds, name_of):
    return np.concatenate([np.asarray(fold['predictions'][name_of(fold)], dtype=float) for fold in folds])


def evaluate_task(folds, reference_names):
    """Validation-selected gate. Every baseline's full metric block is retained.

    A TEST-minimum model is reported as a diagnostic and is not a nominee.
    No new fit or selection is performed.
    """
    if len(reference_names) != 16:
        raise ValueError('Sixteen fixed simple references required')
    for fold in folds:
        if fold['selected_simple'] not in reference_names:
            raise ValueError('Selected simple is outside the fixed roster')
        for family in ('I0', 'I1', 'I2'):
            if family not in fold['aligned_names'] or fold['aligned_names'][family] not in fold['predictions']:
                raise ValueError('Validation-selected aligned family is missing')
        for key in ('shuffle0', 'shuffle1', 'shuffle2'):
            if fold['shuffled_names'][key] not in fold['predictions']:
                raise ValueError('Validation-selected shuffled policy is missing')
    support = np.concatenate([np.asarray(fold['support'], dtype=bool) for fold in folds])
    pairs = np.concatenate([np.asarray(fold['pairs'], dtype=np.int64) for fold in folds])
    targets = {key: np.concatenate([np.asarray(fold[key], dtype=float) for fold in folds])
        for key in ('mean', 'infection0', 'infection1')}
    series = {}

    def add(slot, role, names, prediction):
        series[slot] = dict(role=role, names=list(names), selection='validation', nominee=False,
            **_model_metrics(targets, prediction, support))

    add('selected_policy', 'validation_selected_full_policy',
        [fold['selected_policy'] for fold in folds],
        _concat_named(folds, lambda fold: fold['selected_policy']))
    add('selected_simple', 'validation_selected_simple',
        [fold['selected_simple'] for fold in folds],
        _concat_named(folds, lambda fold: fold['selected_simple']))
    for family in ('I0', 'I1', 'I2'):
        add(family, 'validation_selected_aligned_family',
            [fold['aligned_names'][family] for fold in folds],
            _concat_named(folds, lambda fold, family=family: fold['aligned_names'][family]))
    for key in ('shuffle0', 'shuffle1', 'shuffle2'):
        add(key, 'validation_selected_shuffled_full_policy',
            [fold['shuffled_names'][key] for fold in folds],
            _concat_named(folds, lambda fold, key=key: fold['shuffled_names'][key]))
    references = {}
    for name in reference_names:
        prediction = np.concatenate([np.asarray(fold['predictions'][name], dtype=float) for fold in folds])
        references[name] = dict(role='fixed_simple_reference', names=[name], selection='fixed',
            nominee=False, **_model_metrics(targets, prediction, support))
        series['reference:' + name] = references[name]
    n_full = int(len(support))
    n_supported = int(support.sum())
    coverage = float(n_supported / n_full) if n_full else 0.
    supported_pairs = pairs[support]
    reasons = []
    if coverage < .90:
        reasons.append('coverage')
    if n_supported < 50:
        reasons.append('supported_pairs')
    genes = np.unique(supported_pairs) if n_supported else np.asarray([])
    if len(genes) < 10:
        reasons.append('supported_genes')
    for fold in folds:
        fractions = fold['permutable']
        if any(float(fractions.get(f'shuffle{i}', 0.)) < .8 for i in range(3)):
            reasons.append('shuffle_informative')
            break
    diagnostic = _test_best_diagnostic(series)
    report = dict(coverage=coverage, supported_pairs=n_supported, full_test_pairs=n_full,
        supported_genes=int(len(genes)), population_definitions=dict(
            common_support='both endpoints have at least two distinct TRAIN partners',
            full_population='every original test pair; unsupported rows keep the stored fixed single-sum fallback'),
        models=series, fixed_simple_references=references, test_best_diagnostic=diagnostic,
        biological_confirmation=False, reasons=[])
    if n_supported == 0:
        reasons.extend(['mse_gain', 'upper_bound', 'infection', 'reference_veto', 'shuffle_veto'])
        report.update(reasons=reasons, passes=False, uncertainty=None)
        return report
    policy_mean = series['selected_policy']['common_support']['mean']
    simple_mean = series['selected_simple']['common_support']['mean']
    policy_mse, simple_mse = policy_mean['mse'], simple_mean['mse']
    gain = None if simple_mse <= 0 else (simple_mse - policy_mse) / simple_mse
    if gain is None or gain < .05:
        reasons.append('mse_gain')
    policy_prediction = _concat_named(folds, lambda fold: fold['selected_policy'])
    simple_prediction = _concat_named(folds, lambda fold: fold['selected_simple'])
    delta = (policy_prediction[support] - targets['mean'][support])**2 - (
        simple_prediction[support] - targets['mean'][support])**2
    uncertainty = paired_gene_uncertainty(delta, supported_pairs, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    upper = float(uncertainty['interval'][1])
    if upper >= 0:
        reasons.append('upper_bound')
    if any(series['selected_policy']['common_support'][key]['mse']
           > series['selected_simple']['common_support'][key]['mse']
           for key in ('infection0', 'infection1')):
        reasons.append('infection')
    reference_losses = [name for name, block in references.items()
        if policy_mse > block['common_support']['mean']['mse']]
    if reference_losses:
        reasons.append('reference_veto')
    shuffle_losses = [key for key in ('shuffle0', 'shuffle1', 'shuffle2')
        if policy_mse >= series[key]['common_support']['mean']['mse']]
    if shuffle_losses:
        reasons.append('shuffle_veto')
    versus = {slot: policy_mse - block['common_support']['mean']['mse']
        for slot, block in series.items() if slot != 'selected_policy'}
    report.update(reasons=reasons, passes=not reasons, mse_gain=gain, policy_mse=policy_mse,
        simple_mse=simple_mse, upper_conditional_bound=upper, uncertainty=uncertainty,
        reference_losses=reference_losses, shuffle_losses=shuffle_losses,
        policy_minus_baseline_mse=versus)
    return report


def _test_best_diagnostic(series):
    """Lowest common-support mean MSE. This lookup does not select or nominate."""
    ranked = []
    for slot, block in series.items():
        metrics = block['common_support']['mean']
        if metrics is None:
            continue
        ranked.append((metrics['mse'], len(ranked), slot))
    if not ranked:
        return dict(nominee=False, selection='test', name=None,
            label='TEST-best diagnostic; not a nominee and not used for selection')
    _, _, slot = min(ranked)
    return dict(nominee=False, selection='test', slot=slot, names=series[slot]['names'],
        mse=series[slot]['common_support']['mean']['mse'],
        label='TEST-best diagnostic; not a nominee and not used for selection')


def program_gate(task_reports):
    if set(task_reports) != set(TASKS):
        raise ValueError('Six-task roster required')
    failed = [task for task in TASKS if not task_reports[task]['passes']]
    return dict(passes=not failed, failed_tasks=failed, biological_confirmation=False, rescue=False)


def _verify_prediction_seal(root, expected_procedures):
    root = Path(root)
    manifest = read(root / 'all-predictions-sealed.json')
    if len(manifest['procedures']) != expected_procedures:
        raise ValueError('Partial prediction roster')
    for name, expected in manifest['files'].items():
        if digest(root / name) != expected:
            raise ValueError('Sealed artifact drift: ' + name)
    return manifest


def _score_roster(root, *, expected_procedures, apply_program_gate):
    root = Path(root)
    verify_binding(root)
    _refuse_markers(root, ('scoring-start.json', 'results.json', 'replay-start.json', 'replay.json'))
    manifest = _verify_prediction_seal(root, expected_procedures)
    write(root / 'scoring-start.json', dict(utc=_now(),
        seal_sha256=digest(root / 'all-predictions-sealed.json'),
        protocol_sha256=digest(root / 'protocol.json')))
    guard = Guard(root, PROGRAM_CAP)
    if (root / 'fit-resource.json').exists():
        guard.add_prior(('fit-resource.json',))
    task_folds = {}
    comparisons = []
    for procedure in manifest['procedures']:
        guard.check()
        proc = root / procedure['path']
        choice = read(proc / 'selection.json')
        with np.load(proc / 'queries.npz', allow_pickle=False) as queries:
            query_pairs = queries['pairs'].copy()
            gene_names = queries['gene_names'].copy()
            test_index = queries['test_index'].copy()
            support = queries['supported'].copy()
        sealed_path = root / 'sealed-test' / f"{procedure['task']}__fold-{procedure['fold']}.npz"
        with np.load(sealed_path, allow_pickle=False) as sealed:
            if not np.array_equal(sealed['test_index'], test_index):
                raise ValueError('Test index drifted')
            if not np.array_equal(gene_names[query_pairs], sealed['pair_names']):
                raise ValueError('Test pair identity drifted')
            targets = {key: sealed[key].copy() for key in ('mean', 'infection0', 'infection1')}
        with np.load(proc / 'predictions.npz', allow_pickle=False) as predictions:
            vectors = {record['name']: predictions[record['prediction_key']].copy() for record in choice['models']}
        simple_names = [record['name'] for record in choice['models'] if str(record['family']).startswith('B')]
        if len(simple_names) != 16:
            raise ValueError('Sixteen stored simple references required')
        aligned_names = {family: choice['family_selections'][family] for family in ('I0', 'I1', 'I2')}
        fold = dict(pairs=query_pairs, support=support, predictions=vectors,
            selected_simple=choice['selected_simple'], selected_policy=choice['selected_policy'],
            aligned_names=aligned_names, shuffled_names=choice['shuffled_policies'],
            permutable=choice['permutable'], reference_names=simple_names, **targets)
        task_folds.setdefault(procedure['task'], []).append(fold)
        comparisons.append(dict(task=procedure['task'], fold=procedure['fold'],
            selected_simple=choice['selected_simple'], selected_policy=choice['selected_policy']))
    results = dict(protocol_sha256=digest(root / 'protocol.json'), biological_confirmation=False,
        procedures=comparisons, tasks={})
    if apply_program_gate:
        reports = {}
        for task in TASKS:
            folds = task_folds.get(task, [])
            if len(folds) != 5:
                raise ValueError('Task fold roster is incomplete')
            reference_names = folds[0]['reference_names']
            if any(fold['reference_names'] != reference_names for fold in folds[1:]):
                raise ValueError('Simple roster differs across folds')
            reports[task] = evaluate_task(folds, reference_names)
        results['tasks'] = reports
        results['program'] = program_gate(reports)
    write(root / 'results.json', results)
    guard.finish('score')
    print(dict(scored=True, procedures=len(comparisons),
        passes=results.get('program', {}).get('passes')))
    return results


def score(root):
    return _score_roster(root, expected_procedures=REGISTERED_PROCEDURES, apply_program_gate=True)


_PRIOR_REQUIRED = (
    'protocol.json', 'code-binding.json', 'bank-expectations.json',
    'profile-started.json', 'profile-resource.json', 'profile-qualification.json',
)
_PRIOR_ALLOWED = set(_PRIOR_REQUIRED) | {'.gitignore'}
_BIOLOGY_MARKERS = (
    'fit-started.json', 'inputs-sealed.json', 'all-predictions-sealed.json',
    'scoring-start.json', 'results.json', 'replay-start.json', 'replay.json',
    'fit-resource.json', 'score-resource.json',
)
_PRIOR_SOURCE_PROTOCOL = 'opal-frontier-growth-v1'


def _refuse_prior_biology(prior):
    prior = Path(prior)
    if not prior.is_dir():
        raise ValueError('Prior profile directory is missing')
    for name in _BIOLOGY_MARKERS:
        if (prior / name).exists():
            raise QualificationError(
                'Prior profile has biological stage output; packing will not read it: ' + name)
    for name in ('procedures', 'sealed-test'):
        if (prior / name).exists():
            raise QualificationError(
                'Prior profile has biological stage output; packing will not read it: ' + name)
    if any(prior.glob('failure-*.json')):
        raise QualificationError('Prior profile has a preserved failure; packing will not continue it')


def _unexpected_prior_files(prior):
    found = []
    for path in sorted(item for item in Path(prior).rglob('*') if item.is_file()):
        relative = path.relative_to(prior).as_posix()
        if relative in _PRIOR_ALLOWED or relative.startswith(('profile/', 'profile-sealed/')):
            continue
        found.append(relative)
    if found:
        raise QualificationError('Unexpected prior files: ' + ', '.join(found[:20]))


def _verify_bound_hashes(files):
    if not isinstance(files, dict) or not files:
        raise ValueError('Code binding files are missing')
    for name, expected in files.items():
        path = ROOT / str(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Code binding drift: ' + str(name))


def _verify_prior_environment(binding):
    env = binding['environment']
    if env['numpy'] != np.__version__ or env['torch'] != torch.__version__ or env['python'] != sys.version:
        raise ValueError('Numerical environment drift')
    if env['threads'] != 1 or torch.get_num_threads() != 1:
        raise ValueError('Numerical thread drift')
    if binding.get('model_contract') != CONTRACT:
        raise ValueError('Model contract drift')


def verify_prior_profile(prior):
    """Check the measured v1 profile without opening a biological bank or refitting."""
    prior = Path(prior)
    _refuse_prior_biology(prior)
    _unexpected_prior_files(prior)
    missing = [name for name in _PRIOR_REQUIRED if not (prior / name).is_file()]
    if missing:
        raise ValueError('Prior profile file missing: ' + ', '.join(missing))
    if not (prior / 'profile').is_dir() or not (prior / 'profile-sealed').is_dir():
        raise ValueError('Prior profile directories are missing')
    protocol = read(prior / 'protocol.json')
    binding = read(prior / 'code-binding.json')
    if protocol.get('id') != _PRIOR_SOURCE_PROTOCOL:
        raise ValueError('Prior profile is not growth v1')
    if digest(prior / 'protocol.json') != binding.get('protocol_sha256'):
        raise ValueError('Prior protocol drift')
    if protocol.get('model_contract') != CONTRACT or binding.get('model_contract') != CONTRACT:
        raise ValueError('Prior model contract drift')
    _verify_bound_hashes(binding['files'])
    _verify_prior_environment(binding)
    expectations = read(prior / 'bank-expectations.json')
    if expectations.get('banks_opened') is not False:
        raise QualificationError('Prior bank expectations do not record banks as unopened')
    qualification = read(prior / 'profile-qualification.json')
    if qualification.get('qualified') is not False:
        raise ValueError('Prior profile qualification is not the unchanged failed forecast')
    if qualification.get('biological_fits') != 0:
        raise QualificationError('Prior profile recorded a biological fit')
    if qualification.get('protocol_sha256') != binding['protocol_sha256']:
        raise ValueError('Prior qualification protocol drift')
    checked = forecast_program(qualification['measurements'], qualification['profile_bytes'])
    for key in ('qualified', 'blocker', 'forecast_artifact_bytes', 'forecast_wall_seconds',
                'forecast_cpu_seconds', 'forecast_peak_rss_bytes', 'profile_bytes'):
        if checked.get(key) != qualification.get(key):
            raise ValueError('Prior qualification field does not match its measurements: ' + key)
    measured = _bytes(prior / 'profile') + _bytes(prior / 'profile-sealed')
    if measured != int(qualification['profile_bytes']):
        raise ValueError(
            f'Prior profile bytes drifted: measured={measured} recorded={qualification["profile_bytes"]}')
    if len(qualification['measurements']) != 4:
        raise ValueError('Prior profile roster is not the four registered procedures')
    names = set()
    for record in qualification['measurements']:
        if int(record['persisted_models']) != 28 or int(record['threads']) != 1:
            raise ValueError('Prior profile did not persist 28 single-thread models')
        if int(record['profile_queries']) != PROFILE_QUERY_COUNT:
            raise ValueError('Prior profile query count drifted')
        name = f"{record['world']}-g{int(record['genes'])}"
        names.add(name)
    needed = {f'{world}-g{genes}' for world in PROFILE_WORLDS for genes in PROFILE_GENES}
    directories = {path.name for path in (prior / 'profile').iterdir() if path.is_dir()}
    sealed = {path.name for path in (prior / 'profile-sealed').iterdir() if path.is_file()}
    if names != needed or directories != needed or sealed != {f'{name}.npz' for name in needed}:
        raise ValueError('Prior profile directories do not match the registered roster')
    return qualification, binding


def _matching_outer_receipt(prior):
    receipt_path = ROOT / 'results' / 'frontier-execution-20261026' / 'growth-profile-v1.json'
    if not receipt_path.is_file():
        return None
    receipt = read(receipt_path)
    command = receipt.get('command')
    if not isinstance(command, list) or 'profile' not in command or '--output' not in command:
        return None
    output = command[command.index('--output') + 1]
    cwd = Path(receipt.get('cwd') or '.')
    try:
        resolved = (cwd / output).resolve()
    except OSError:
        return None
    if resolved != Path(prior).resolve():
        return None
    return receipt


def _prior_profile_clock(prior):
    resource = read(prior / 'profile-resource.json')
    wall = float(resource['wall_seconds'])
    cpu = float(resource['cpu_seconds'])
    sources = [dict(kind='profile-resource', path='profile-resource.json',
                    wall_seconds=wall, cpu_seconds=cpu,
                    peak_rss_bytes=int(resource['peak_rss_bytes']))]
    receipt = _matching_outer_receipt(prior)
    if receipt is not None:
        outer_wall = float(receipt['wall_seconds'])
        outer_cpu = float(receipt['sampled_cpu_seconds'])
        sources.append(dict(
            kind='outer-watchdog',
            path='results/frontier-execution-20261026/growth-profile-v1.json',
            wall_seconds=outer_wall, cpu_seconds=outer_cpu,
            peak_rss_bytes=int(receipt['sampled_tree_peak_bytes'])))
        wall = max(wall, outer_wall)
        cpu = max(cpu, outer_cpu)
    failed_path = ROOT / 'results/frontier-execution-20261026/packed-qualification-v1.json'
    failed = read(failed_path)
    wall += float(failed['wall_seconds'])
    cpu += float(failed['sampled_cpu_seconds'])
    sources.append(dict(kind='preserved failed packing attempt', path=failed_path.as_posix(), wall_seconds=failed['wall_seconds'], cpu_seconds=failed['sampled_cpu_seconds']))
    return wall, cpu, sources


def forecast_packed_program(procedures, retained_profile_bytes, packed_profile_bytes,
                            prior_peak_rss_bytes, packing_peak_rss_bytes,
                            prior_wall_seconds, prior_cpu_seconds,
                            stage_wall_seconds, stage_cpu_seconds):
    """Forecast the unchanged 30-procedure roster after replacing completion files.

    Per procedure, stored bytes become the measured v1 artifact minus the full
    completion checkpoints plus the packed completion checkpoints. Retained v1
    profile bytes and the new packed profile are both kept. Caps are unchanged.
    """
    if len(procedures) != 4:
        raise ValueError('Packed forecast requires the four measured profile procedures')
    adjusted = []
    walls = []
    cpus = []
    for record in procedures:
        full = int(record['full_completion_bytes'])
        packed = int(record['packed_completion_bytes'])
        previous = int(record['v1_artifact_bytes'])
        if full > previous:
            raise ValueError('Completion bytes exceed the measured procedure')
        updated = previous - full + packed
        if int(record.get('new_artifact_bytes', updated)) != updated:
            raise ValueError('Adjusted procedure bytes are inconsistent')
        adjusted.append(updated)
        walls.append(float(record['v1_wall_seconds']) + float(record['packing_wall_seconds']))
        cpus.append(float(record['v1_cpu_seconds']) + float(record['packing_cpu_seconds']))
    retained = int(retained_profile_bytes)
    packed_profile = int(packed_profile_bytes)
    combined_profile = retained + packed_profile
    forecast_artifacts = max(adjusted) * REGISTERED_PROCEDURES * 1.25 + retained + packed_profile
    forecast_wall = max(walls) * REGISTERED_PROCEDURES * 1.25
    forecast_cpu = max(cpus) * REGISTERED_PROCEDURES * 1.25
    forecast_peak = max(int(prior_peak_rss_bytes), int(packing_peak_rss_bytes))
    cumulative_wall = float(prior_wall_seconds) + float(stage_wall_seconds)
    cumulative_cpu = float(prior_cpu_seconds) + float(stage_cpu_seconds)
    reasons = []
    if forecast_artifacts > PROGRAM_CAP['artifact_bytes']:
        reasons.append('forecast exceeds the registered program artifact cap')
    if forecast_wall > PROGRAM_CAP['wall_seconds'] or forecast_cpu > PROGRAM_CAP['cpu_seconds']:
        reasons.append('forecast exceeds the registered program time cap')
    if forecast_peak > PROGRAM_CAP['rss_bytes']:
        reasons.append('forecast exceeds the registered program memory cap')
    if combined_profile > PROFILE_CAP['artifact_bytes']:
        reasons.append('retained profile plus packed profile exceeds the 64MiB profile cap')
    if cumulative_wall > PROFILE_CAP['wall_seconds'] or cumulative_cpu > PROFILE_CAP['cpu_seconds']:
        reasons.append('cumulative profile wall/CPU exceeds 600 seconds')
    return dict(
        qualified=not reasons, blocker=None if not reasons else '; '.join(reasons),
        margin=1.25, procedures=REGISTERED_PROCEDURES,
        retained_profile_bytes=retained, packed_profile_bytes=packed_profile,
        combined_profile_bytes=combined_profile, adjusted_procedure_bytes=adjusted,
        forecast_wall_seconds=forecast_wall, forecast_cpu_seconds=forecast_cpu,
        forecast_artifact_bytes=forecast_artifacts, forecast_peak_rss_bytes=forecast_peak,
        prior_peak_rss_bytes=int(prior_peak_rss_bytes),
        packing_peak_rss_bytes=int(packing_peak_rss_bytes),
        cumulative_profile_wall_seconds=cumulative_wall,
        cumulative_profile_cpu_seconds=cumulative_cpu,
        artifact_cap_bytes=PROGRAM_CAP['artifact_bytes'],
        profile_cap_bytes=PROFILE_CAP['artifact_bytes'],
        source_cap_changed=False, scientific_gate_changed=False, menu_changed=False,
        exact_prediction_comparison='numpy.array_equal',
        profile_query_count=PROFILE_QUERY_COUNT)


def _pack_procedure(prior, root, measurement, guard):
    name = f"{measurement['world']}-g{int(measurement['genes'])}"
    source = prior / 'profile' / name
    sealed = prior / 'profile-sealed' / f'{name}.npz'
    live = _bytes(source) + sealed.stat().st_size
    if live != int(measurement['artifact_bytes']):
        raise ValueError(f'Prior procedure artifact bytes drifted: {name}')
    choice = read(source / 'selection.json')
    if len(choice['models']) != 28 or int(choice.get('seed', -1)) != FIT_SEED:
        raise ValueError(f'Prior selection roster drifted: {name}')
    referenced = {
        f'profile/{name}/selection.json': digest(source / 'selection.json'),
        f'profile/{name}/queries.npz': digest(source / 'queries.npz'),
        f'profile/{name}/predictions.npz': digest(source / 'predictions.npz'),
        f'profile/{name}/fit-args.npz': digest(source / 'fit-args.npz'),
        f'profile-sealed/{name}.npz': digest(sealed),
    }
    started, cpu_started = time.perf_counter(), time.process_time()
    guard.check()
    packed_dir = root / 'profile-packed' / name
    packed_dir.mkdir(parents=True, exist_ok=False)
    with np.load(source / 'queries.npz', allow_pickle=False) as queries:
        pairs = queries['pairs'].copy()
        singles = queries['singles'].copy()
        supported_saved = queries['supported'].copy()
    if len(pairs) != PROFILE_QUERY_COUNT or len(singles) != int(measurement['genes']):
        raise ValueError(f'Prior query workload drifted: {name}')
    choice_models = choice['models']
    with np.load(source / 'predictions.npz', allow_pickle=False) as saved:
        stored = {record['prediction_key']: saved[record['prediction_key']].copy() for record in choice_models}
    models = []
    completion_bytes = 0
    packed_bytes = 0
    completions = 0
    for record in choice_models:
        guard.check()
        source_path = source / 'checkpoints' / record['checkpoint']
        source_sha = digest(source_path)
        if source_sha != record['checkpoint_sha256']:
            raise ValueError('Prior checkpoint hash mismatch: ' + record['checkpoint'])
        loaded = load_checkpoint(source_path, source_sha, historical_roots=(prior, root))
        completion = str(loaded.state['kind']) == 'completion'
        if completion:
            if not str(record['family']).startswith('I0') or loaded.base is None:
                raise ValueError('Completion checkpoint is outside the packed I0 roster')
            destination = packed_dir / record['checkpoint']
            save_checkpoint(loaded, destination)
            packed_sha = digest(destination)
            model = load_checkpoint(destination, packed_sha, historical_roots=(prior, root))
            source_size = source_path.stat().st_size
            destination_size = destination.stat().st_size
            completion_bytes += source_size
            packed_bytes += destination_size
            completions += 1
            entry = dict(name=record['name'], family=record['family'], checkpoint=record['checkpoint'],
                source_sha256=source_sha, packed=True, packed_sha256=packed_sha,
                packed_bytes=destination_size, source_bytes=source_size,
                packed_path=(Path('profile-packed') / name / record['checkpoint']).as_posix())
        else:
            if str(record['family']).startswith('I0'):
                raise ValueError('I0 checkpoint is not a completion model')
            model = loaded
            entry = dict(name=record['name'], family=record['family'], checkpoint=record['checkpoint'],
                source_sha256=source_sha, packed=False, packed_sha256=None,
                packed_bytes=0, source_bytes=source_path.stat().st_size, packed_path=None)
        prediction, supported = tournament_tool.predictions(model, pairs, singles)
        if not np.array_equal(prediction, stored[record['prediction_key']]):
            raise ValueError('Prediction replay mismatch: ' + name + ' ' + record['checkpoint'])
        if not np.array_equal(supported, supported_saved):
            raise ValueError('Support replay mismatch: ' + name + ' ' + record['checkpoint'])
        if digest(source_path) != source_sha:
            raise ValueError('Prior checkpoint changed during packing')
        models.append(entry)
    if completions != COMPLETION_CHECKPOINTS_PER_PROCEDURE:
        raise ValueError(f'Expected 4 completion checkpoints in {name}')
    for relative, expected in referenced.items():
        if digest(prior / relative) != expected:
            raise ValueError('Prior profile file changed during packing: ' + relative)
    new_artifact = int(measurement['artifact_bytes']) - completion_bytes + packed_bytes
    return dict(
        name=name, world=measurement['world'], genes=int(measurement['genes']),
        v1_wall_seconds=float(measurement['wall_seconds']),
        v1_cpu_seconds=float(measurement['cpu_seconds']),
        v1_peak_rss_bytes=int(measurement['peak_rss_bytes']),
        v1_artifact_bytes=int(measurement['artifact_bytes']),
        packing_wall_seconds=time.perf_counter() - started,
        packing_cpu_seconds=time.process_time() - cpu_started,
        full_completion_bytes=completion_bytes, packed_completion_bytes=packed_bytes,
        new_artifact_bytes=new_artifact, predictions_replayed=len(models),
        source_files=referenced, models=models)


def qualify_packed(root, prior):
    """Freeze a separate v2 folder, then pack and replay the measured v1 profile."""
    started, cpu_started = time.perf_counter(), time.process_time()
    root = Path(root)
    prior = Path(prior).resolve()
    freeze(root)
    root_resolved = root.resolve()
    if prior == root_resolved or root_resolved.is_relative_to(prior) or prior.is_relative_to(root_resolved):
        raise ValueError('v2 output must be a separate directory from the prior profile')
    qualification, binding = verify_prior_profile(prior)
    source_sha = digest(prior / 'profile-qualification.json')
    write(root / 'packed-started.json', dict(
        utc=_now(), protocol_sha256=digest(root / 'protocol.json'),
        source_profile_qualification_sha256=source_sha,
        source_protocol_sha256=binding['protocol_sha256'],
        prior_profile=prior.as_posix(), biological_fits=0, banks_opened=False))
    guard = Guard(root, PROFILE_CAP)
    prior_wall, prior_cpu, clock_sources = _prior_profile_clock(prior)
    guard.retained_bytes = int(qualification['profile_bytes'])
    guard.prior_wall = prior_wall + (time.perf_counter() - started)
    guard.prior_cpu = prior_cpu + (time.process_time() - cpu_started)
    guard.check()
    procedures = []
    for measurement in qualification['measurements']:
        procedures.append(_pack_procedure(prior, root, measurement, guard))
    if sum(item['predictions_replayed'] for item in procedures) != 112:
        raise RuntimeError('Packed replay did not cover 112 models')
    retained = int(qualification['profile_bytes'])
    prior_peak = max(int(item['peak_rss_bytes']) for item in qualification['measurements'])
    if prior_peak != int(qualification['forecast_peak_rss_bytes']):
        raise ValueError('Prior peak evidence drifted')
    result = forecast_packed_program(
        procedures, retained, _bytes(root / 'profile-packed'), prior_peak, guard.peak,
        prior_wall, prior_cpu, time.perf_counter() - started, time.process_time() - cpu_started)
    result.update(
        prior_profile_qualified=False, prior_blocker=qualification.get('blocker'),
        prior_forecast_artifact_bytes=qualification['forecast_artifact_bytes'],
        prior_forecast_wall_seconds=qualification['forecast_wall_seconds'],
        prior_forecast_cpu_seconds=qualification['forecast_cpu_seconds'],
        prior_forecast_peak_rss_bytes=qualification['forecast_peak_rss_bytes'],
        source_profile_qualification_sha256=source_sha,
        source_protocol_sha256=binding['protocol_sha256'],
        source_code_binding_sha256=digest(prior / 'code-binding.json'),
        prior_profile=prior.as_posix(), clock_sources=clock_sources,
        reused_measurements=qualification['measurements'], packed_procedures=procedures,
        protocol_sha256=digest(root / 'protocol.json'), utc=_now(),
        biological_fits=0, banks_opened=False)
    if digest(prior / 'profile-qualification.json') != source_sha:
        raise ValueError('Prior profile qualification changed')
    write(root / 'packed-qualification.json', result)
    guard.finish('packed')
    print({key: result[key] for key in (
        'qualified', 'blocker', 'forecast_artifact_bytes', 'combined_profile_bytes',
        'forecast_wall_seconds', 'forecast_cpu_seconds')})
    return result


def active_predictor(installed_root=None, historical_roots=()):
    """Return the mixed loader after checking the imported model and packing modules."""
    model_module = importlib.import_module('opal.frontier_tournament')
    packing_module = importlib.import_module('opal.frontier_packing_v2')
    if installed_root is None:
        if model_module.Predictor is not Predictor:
            raise ImportError('Checkout predictor import drifted')
        if packing_module.PACKING_CONTRACT != PACKING_CONTRACT:
            raise ImportError('Checkout packing import drifted')
    else:
        installed = Path(installed_root).resolve()
        checkout = ROOT.resolve()
        src = (checkout / 'src').resolve()
        if installed == checkout or installed.is_relative_to(checkout):
            raise ValueError('Installed replay root must be outside the checkout')
        for module in (model_module, packing_module):
            origin = Path(module.__file__).resolve()
            if origin.is_relative_to(src) or not origin.is_relative_to(installed):
                raise ValueError('Replay imported checkout rather than installed package')
    return packing_module.MixedLoader(historical_roots=historical_roots)


def replay(root, installed_root=None):
    root = Path(root)
    verify_binding(root)
    _refuse_markers(root, ('replay-start.json', 'replay.json'))
    if not (root / 'results.json').exists() or not (root / 'scoring-start.json').exists():
        raise ValueError('Replay requires sealed scoring')
    manifest = _verify_prediction_seal(root, REGISTERED_PROCEDURES)
    predictor = active_predictor(installed_root, historical_roots=(root,))
    write(root / 'replay-start.json', dict(utc=_now(), results_sha256=digest(root / 'results.json')))
    guard = Guard(root, PROGRAM_CAP)
    guard.add_prior(('fit-resource.json', 'score-resource.json'))
    counts = dict(checkpoints=0, refits=0)
    for procedure in manifest['procedures']:
        guard.check()
        proc = root / procedure['path']
        choice = read(proc / 'selection.json')
        with np.load(proc / 'queries.npz', allow_pickle=False) as queries:
            pairs, singles = queries['pairs'], queries['singles']
        with np.load(proc / 'predictions.npz', allow_pickle=False) as saved:
            stored = {record['prediction_key']: saved[record['prediction_key']].copy() for record in choice['models']}
        for record in choice['models']:
            model = predictor.load(proc / 'checkpoints' / record['checkpoint'], record['checkpoint_sha256'])
            prediction, _ = tournament_tool.predictions(model, pairs, singles)
            if not np.array_equal(prediction, stored[record['prediction_key']]):
                raise ValueError('Prediction replay mismatch')
            counts['checkpoints'] += 1
    if counts['refits'] != 0:
        raise RuntimeError('Replay refit is refused')
    model_module = importlib.import_module('opal.frontier_tournament')
    packing_module = importlib.import_module('opal.frontier_packing_v2')
    write(root / 'replay.json', dict(passed=True, refits=0, counts=counts,
        installed=installed_root is not None,
        imported_module=model_module.__file__,
        imported_packing=packing_module.__file__,
        results_sha256=digest(root / 'results.json')))
    guard.finish('replay')
    print(dict(replayed=True, **counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'profile', 'qualify-packed', 'fit', 'score', 'replay'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--installed-root', type=Path)
    parser.add_argument('--prior-profile', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.stage != 'replay' and args.installed_root is not None:
        raise ValueError('Installed-root is only meaningful for replay')
    if args.stage != 'qualify-packed' and args.prior_profile is not None:
        raise ValueError('prior-profile is only meaningful for qualify-packed')
    if args.stage == 'qualify-packed' and args.prior_profile is None:
        raise ValueError('qualify-packed requires --prior-profile')
    preexisting = root.exists()
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        if args.stage == 'freeze':
            freeze(root)
        elif args.stage == 'qualify-packed':
            qualify_packed(root, args.prior_profile.resolve())
        elif args.stage == 'profile':
            profile(root)
        elif args.stage == 'fit':
            fit(root)
        elif args.stage == 'score':
            score(root)
        else:
            replay(root, args.installed_root)
    except Exception as error:
        failed_new = root.is_dir() and not preexisting
        failed_stage = root.is_dir() and preexisting and args.stage != 'freeze'
        if (failed_new or failed_stage) and not any(root.glob('failure-*.json')):
            failure = root / f'failure-{args.stage}-{time.time_ns()}.json'
            if not failure.exists():
                write(failure, dict(error=type(error).__name__, message=str(error),
                    wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                    partial_outputs_preserved=True))
        raise


if __name__ == '__main__':
    main()
