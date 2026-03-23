# 模型模块初始化
from .image_classifier import train_image_classifier, load_image_model, predict_image_from_bytes
from .spectrum_model import load_spectrum_model, predict_spectrum
from .system_manager import SystemManager

__all__ = [
    'train_image_classifier', 'load_image_model', 'predict_image_from_bytes',
    'load_spectrum_model', 'predict_spectrum',
    'SystemManager'
]