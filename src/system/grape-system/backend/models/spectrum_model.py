# =============================================================================
#  高光谱模型加载与预测
# =============================================================================
import os
import pickle
import numpy as np
import pandas as pd
from backend.config import MODEL_DIR, DATA_DIR
from backend.utils import preprocess_spectrum, get_wavelength_columns


def load_spectrum_model():
    """
    加载高光谱相关模型：分类器 + PCA + 标准化器
    """
    try:
        # 加载模型文件
        with open(os.path.join(MODEL_DIR, 'stacking_model.pkl'), 'rb') as f:
            spec_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'pca_model.pkl'), 'rb') as f:
            spec_pca = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'scaler_spec.pkl'), 'rb') as f:
            spec_scaler = pickle.load(f)

        # 获取波长列表
        df = pd.read_csv(os.path.join(DATA_DIR, '聚类标签数据集.csv'), nrows=1)
        skip = {'序号', 'maturity_label', 'maturity_name'}
        wavelengths = [float(c) for c in df.columns if c not in skip]

        print(f'  [高光谱模型] 加载成功，{len(wavelengths)} 波段')
        return spec_model, spec_pca, spec_scaler, wavelengths
    except Exception as e:
        print(f'  [高光谱模型] 加载失败: {e}')
        return None, None, None, None


def predict_spectrum(spec_model, spec_pca, spec_scaler, X_raw: np.ndarray):
    """
    高光谱数据预测
    """
    if spec_model is None or spec_pca is None or spec_scaler is None:
        raise RuntimeError('高光谱模型未加载')

    # 预处理
    X_snv = preprocess_spectrum(X_raw)
    X_n = spec_scaler.transform(X_snv)
    X_pca = spec_pca.transform(X_n)

    # 预测
    labels = spec_model.predict(X_pca).tolist()
    probs = spec_model.predict_proba(X_pca)

    return labels, probs