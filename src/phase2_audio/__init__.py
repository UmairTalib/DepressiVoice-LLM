"""
Phase 2: Audio Feature Extraction

This package contains utilities for extracting WavLM-Large features
from patient audio recordings for depression detection.

Main components:
- extract_features.py: Main extraction script
- normalize.py: Batch normalization
- validate.py: Feature validation utilities
- audiomodel.py: WaveLM wrapper
- dataset.py: Audio dataset loader
"""

__version__ = "1.0.0"

from .audiomodel import WaveLMAudioModel
from .dataset import AudioDataset

__all__ = ["WaveLMAudioModel", "AudioDataset"]
