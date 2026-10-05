"""Opal research implementation. No biological checkpoint is bundled."""
__version__ = "0.1.0"

def load_model(path):
    from .inference import Predictor
    return Predictor(path)
