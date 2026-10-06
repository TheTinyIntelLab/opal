"""Opal research implementation. No biological checkpoint is bundled."""
__version__ = "0.1.0"

def load_model(path):
    from pathlib import Path
    from .artifacts import read_json
    path = Path(path)
    schema = read_json(path / 'manifest.json').get('schema')
    if schema in {'opal-state-response-v1', 'opal-state-response-ensemble-v1'}:
        from .state import StatePredictor
        return StatePredictor(path)
    if schema == 'opal-contextual-observed-singles-v2':
        if (path / 'estimated_manifest.json').exists():
            from .contextual.main_effects import load_estimated_checkpoint
            return load_estimated_checkpoint(path)
        from .contextual.inference import ContextPredictor
        return ContextPredictor(path)
    from .inference import Predictor
    return Predictor(path)
