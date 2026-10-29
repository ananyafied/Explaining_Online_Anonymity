"""Synthetic data with known ground truth for testing explainability methods."""
from .config import SynthConfig
from .generate import SyntheticDataset, generate

__all__ = ["SynthConfig", "SyntheticDataset", "generate"]
