# =============================================================================
#  高光谱数据处理工具
# =============================================================================
import numpy as np
from scipy.signal import savgol_filter


def preprocess_spectrum(X_raw: np.ndarray) -> np.ndarray:
    """
    高光谱数据预处理：SG平滑 + SNV标准化
    """
    # SG平滑
    X_sg = savgol_filter(X_raw, 11, 3, axis=1)

    # SNV标准化
    mean = X_sg.mean(axis=1, keepdims=True)
    std = X_sg.std(axis=1, keepdims=True)
    X_snv = (X_sg - mean) / (std + 1e-10)

    return X_snv


def get_wavelength_columns(df, skip_cols={'序号', 'maturity_label', 'maturity_name'}):
    """
    从DataFrame中提取波长列
    """
    return [c for c in df.columns if c not in skip_cols]