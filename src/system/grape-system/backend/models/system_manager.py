# =============================================================================
#  系统模型管理器
# =============================================================================
from backend.config import NAMES, PALETTE, ICONS, TIPS
from backend.models.image_classifier import load_image_model, train_image_classifier, predict_image_from_bytes
from backend.models.spectrum_model import load_spectrum_model, predict_spectrum


class SystemManager:
    def __init__(self):
        # 高光谱模型
        self.spec_model = None
        self.spec_pca = None
        self.spec_scaler = None
        self.wavelengths = None
        self.spec_loaded = False

        # 图像模型
        self.img_model = None
        self.img_scaler = None
        self.img_loaded = False

        # 模型指标
        self.img_acc = 0.0
        self.img_cv_mean = 0.0
        self.img_cv_std = 0.0

    def load_all(self):
        """加载所有模型"""
        self._load_spec_model()
        self._load_img_model()

    def _load_spec_model(self):
        """加载高光谱模型"""
        spec_model, spec_pca, spec_scaler, wavelengths = load_spectrum_model()
        self.spec_model = spec_model
        self.spec_pca = spec_pca
        self.spec_scaler = spec_scaler
        self.wavelengths = wavelengths
        self.spec_loaded = spec_model is not None

    def _load_img_model(self):
        """加载图像模型"""
        try:
            self.img_model, self.img_scaler = load_image_model()
            self.img_loaded = True
        except Exception as e:
            print(f'  [图像模型] 加载失败: {e}')
            self.img_loaded = False

    def predict_image(self, img_bytes: bytes):
        """图像预测完整流程（带业务封装）"""
        if not self.img_loaded:
            raise RuntimeError('图像模型未加载')

        # 基础预测
        pred_result = predict_image_from_bytes(self.img_model, self.img_scaler, img_bytes)
        label = pred_result['label']

        # 封装业务返回结果
        return {
            'label': label,
            'name': NAMES[label],
            'icon': ICONS[label],
            'color': PALETTE[label],
            'tip': TIPS[label],
            'confidence': pred_result['confidence'],
            'probabilities': {NAMES[i]: round(p, 4) for i, p in enumerate(pred_result['probabilities'])},
            'image_analysis': pred_result['image_analysis']
        }

    def predict_spectrum(self, X_raw: np.ndarray):
        """高光谱预测流程"""
        return predict_spectrum(self.spec_model, self.spec_pca, self.spec_scaler, X_raw)