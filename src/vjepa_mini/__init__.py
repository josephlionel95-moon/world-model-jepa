"""vjepa-mini: a small-scale, from-scratch V-JEPA for learning and research.

Faithful to Bardes et al. (2024) "Revisiting Feature Prediction for Learning
Visual Representations from Video", scaled down to train on a single T4 GPU
using synthetic video (Moving MNIST).
"""

from vjepa_mini.config import VJEPAConfig
from vjepa_mini.models.vjepa import VJEPA
from vjepa_mini.models.vit import VideoViT
from vjepa_mini.models.predictor import Predictor
from vjepa_mini.data.moving_mnist import MovingMNIST
from vjepa_mini.data.masking import MultiBlockMaskGenerator
from vjepa_mini.train.trainer import Trainer
from vjepa_mini.eval.attentive_probe import AttentiveProbe, train_probe

__version__ = "0.1.0"

__all__ = [
    "VJEPAConfig",
    "VJEPA",
    "VideoViT",
    "Predictor",
    "MovingMNIST",
    "MultiBlockMaskGenerator",
    "Trainer",
    "AttentiveProbe",
    "train_probe",
]
