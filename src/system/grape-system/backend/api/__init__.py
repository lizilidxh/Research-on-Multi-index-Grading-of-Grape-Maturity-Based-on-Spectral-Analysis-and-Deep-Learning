# API模块初始化
from .health import register_health_routes
from .image_predict import register_image_predict_routes
from .spectrum_predict import register_spectrum_predict_routes
from .stats import register_stats_routes

__all__ = [
    'register_health_routes',
    'register_image_predict_routes',
    'register_spectrum_predict_routes',
    'register_stats_routes'
]