# =============================================================================
#  图像特征提取工具
# =============================================================================
import numpy as np
import cv2
from config import HSV_MASK_LOW, HSV_MASK_HIGH, MASK_MIN_RATIO


def extract_image_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    从 BGR 图像提取 38 维颜色+纹理特征

    特征组成：
      HSV 统计量 (6维)  ─ 色相/饱和度/明度 均值+标准差
      BGR 统计量 (6维)  ─ 三通道均值+标准差
      HSV 直方图(18维)  ─ H(8桶)+S(6桶)+V(4桶)
      纹理特征  (8维)   ─ Sobel梯度均值/标准差×4方向
    """
    # ── 预处理：去除背景（简单阈值，保留彩色区域）──
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 掩码：排除过白(背景)和过黑区域
    mask = cv2.inRange(hsv, HSV_MASK_LOW, HSV_MASK_HIGH)

    # 如果掩码太小（图像几乎全白），使用全图
    if mask.sum() < mask.size * MASK_MIN_RATIO:
        mask = np.ones(mask.shape, dtype=np.uint8) * 255

    # ── HSV 统计量 (6维) ──
    h = hsv[:, :, 0][mask > 0].astype(float)
    s = hsv[:, :, 1][mask > 0].astype(float)
    v = hsv[:, :, 2][mask > 0].astype(float)
    hsv_feats = np.array([
        h.mean(), h.std(),
        s.mean(), s.std(),
        v.mean(), v.std(),
    ])

    # ── BGR 统计量 (6维) ──
    bgr_feats = []
    for ch in range(3):
        px = img_bgr[:, :, ch][mask > 0].astype(float)
        bgr_feats += [px.mean(), px.std()]
    bgr_feats = np.array(bgr_feats)

    # ── HSV 直方图 (18维) ──
    h_hist = cv2.calcHist([hsv], [0], mask, [8], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [6], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask, [4], [0, 256]).flatten()
    total = mask.sum() / 255.0 + 1e-6
    hist_feats = np.concatenate([h_hist / total, s_hist / total, v_hist / total])

    # ── 纹理特征：Sobel 梯度 (8维) ──
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    sobelx = cv2.Sobel(gray_masked, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_masked, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    px_mag = mag[mask > 0]
    px_sx = np.abs(sobelx[mask > 0])
    px_sy = np.abs(sobely[mask > 0])
    tex_feats = np.array([
        px_mag.mean(), px_mag.std(),
        px_sx.mean(), px_sx.std(),
        px_sy.mean(), px_sy.std(),
        px_mag.max() if len(px_mag) else 0,
        np.percentile(px_mag, 75) if len(px_mag) else 0,
    ])

    return np.concatenate([hsv_feats, bgr_feats, hist_feats, tex_feats])