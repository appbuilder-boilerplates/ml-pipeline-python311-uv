"""The ML pipeline. Its public functions have the eval adapter's shape, so an eval bundle's adapter.py only binds
them (README.md): `load_data`, `train`, `predict`, `save_model`, `describe`.

Imports inside the package are relative, so it imports both as `ml_pipeline` (from this directory) and as
`<component>.ml_pipeline` (from the repository root, which is what the eval's Modal image puts on PYTHONPATH).
"""
from .data import Dataset, load_data, read_table
from .model import Model, describe, predict, save_model, train

__all__ = ["Dataset", "Model", "describe", "load_data", "predict", "read_table", "save_model", "train"]
