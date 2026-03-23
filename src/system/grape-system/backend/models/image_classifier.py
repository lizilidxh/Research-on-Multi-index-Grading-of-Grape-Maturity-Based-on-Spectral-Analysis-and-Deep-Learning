# =============================================================================
#  图像分类器训练与预测
# =============================================================================
import os
import pickle
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from backend.config import (
    IMG_MODEL_PATH, TRAIN_SAMPLE_PER_CLASS, RANDOM_SEED, TEST_SIZE,
    IMG_RESIZE_SIZE
)
from backend.utils import extract_image_features


def train_image_classifier():
    """
    用合成颜色数据训练图像分类器
    颜色分布基于：
      - 葡萄成熟度色泽研究文献
      - 本项目第一阶段色差指标统计（各类色差均值）
        未成熟(23.47) / 半成熟(10.59) / 成熟(5.30) / 过成熟(1.80)
    """
    print('  [图像分类器] 开始训练...', flush=True)
    rng = np.random.RandomState(RANDOM_SEED)
    N = TRAIN_SAMPLE_PER_CLASS  # 每类样本数

    def gen_class(label):
        """生成指定成熟度等级的合成图像特征"""
        samples = []
        # 颜色参数（基于葡萄实际颜色分布）
        params = {
            0: dict(h=(65, 12), s=(175, 25), v=(155, 25),  # 未成熟：绿色
                    b=(50, 20), g=(130, 25), r=(60, 20)),
            1: dict(h=(30, 12), s=(165, 25), v=(135, 25),  # 半成熟：黄绿→红
                    b=(60, 20), g=(100, 25), r=(110, 25)),
            2: dict(h=(155, 10), s=(158, 22), v=(115, 22),  # 成熟：深紫红
                    b=(100, 20), g=(50, 20), r=(140, 20)),
            3: dict(h=(163, 8), s=(138, 22), v=(85, 20),  # 过成熟：暗紫近黑
                    b=(70, 18), g=(30, 15), r=(90, 18)),
        }[label]

        for _ in range(N):
            # 生成随机合成图像（50x50像素小块）
            h_val = np.clip(rng.normal(*params['h']), 0, 179)
            s_val = np.clip(rng.normal(*params['s']), 0, 255)
            v_val = np.clip(rng.normal(*params['v']), 0, 255)
            b_val = np.clip(rng.normal(*params['b']), 0, 255)
            g_val = np.clip(rng.normal(*params['g']), 0, 255)
            r_val = np.clip(rng.normal(*params['r']), 0, 255)

            # 构造含噪声的小图像
            img = np.zeros((50, 50, 3), dtype=np.uint8)
            img[:, :, 0] = np.clip(rng.normal(b_val, 8, (50, 50)), 0, 255)
            img[:, :, 1] = np.clip(rng.normal(g_val, 8, (50, 50)), 0, 255)
            img[:, :, 2] = np.clip(rng.normal(r_val, 8, (50, 50)), 0, 255)
            # 叠加HSV噪声
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
            hsv_img[:, :, 0] = np.clip(hsv_img[:, :, 0] + rng.normal(0, 3, (50, 50)), 0, 179)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] + rng.normal(0, 10, (50, 50)), 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] + rng.normal(0, 10, (50, 50)), 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

            feat = extract_image_features(img)
            samples.append(feat)
        return np.array(samples)

    X_parts, y_parts = [], []
    for label in range(4):
        Xc = gen_class(label)
        X_parts.append(Xc)
        y_parts.append(np.full(len(Xc), label))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # 处理 NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=255.0, neginf=0.0)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # 训练集/测试集分割
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)

    # 软投票集成（速度快，效果好）
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=4, random_state=RANDOM_SEED)
    svm = SVC(C=10, gamma='scale', kernel='rbf', probability=True, random_state=RANDOM_SEED)
    clf = VotingClassifier(
        estimators=[('rf', rf), ('gbm', gbm), ('svm', svm)],
        voting='soft', weights=[0.4, 0.3, 0.3], n_jobs=-1
    )
    clf.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, clf.predict(X_te))
    cv5 = cross_val_score(clf, X_sc, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f'  [图像分类器] 测试集 Acc={acc:.4f}  5折CV={cv5.mean():.4f}±{cv5.std():.4f}')

    # 保存
    model_pkg = {'clf': clf, 'scaler': scaler, 'n_features': X.shape[1]}
    os.makedirs(os.path.dirname(IMG_MODEL_PATH), exist_ok=True)
    with open(IMG_MODEL_PATH, 'wb') as f:
        pickle.dump(model_pkg, f)
    print(f'  [图像分类器] 已保存 → {IMG_MODEL_PATH}')
    return model_pkg, acc, cv5.mean(), cv5.std()


def load_image_model():
    """加载已训练的图像模型"""
    if os.path.exists(IMG_MODEL_PATH):
        try:
            with open(IMG_MODEL_PATH, 'rb') as f:
                pkg = pickle.load(f)
            print('  [图像模型] 加载已有模型')
            return pkg['clf'], pkg['scaler']
        except Exception as e:
            print(f'  [图像模型] 读取失败，重新训练: {e}')
            pkg, _, _, _ = train_image_classifier()
            return pkg['clf'], pkg['scaler']
    else:
        print('  [图像模型] 首次运行，开始训练...')
        pkg, _, _, _ = train_image_classifier()
        return pkg['clf'], pkg['scaler']


def predict_image_from_bytes(clf, scaler, img_bytes: bytes):
    """
    从图片字节数据预测成熟度
    """
    # 解码图像
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('无法解码图像，请确认上传的是有效图片文件')

    # 缩放（统一处理尺寸）
    img = cv2.resize(img, IMG_RESIZE_SIZE)

    # 提取特征
    feat = extract_image_features(img)
    feat = np.nan_to_num(feat, nan=0.0, posinf=255.0, neginf=0.0)
    feat_sc = scaler.transform(feat.reshape(1, -1))

    # 预测
    label = int(clf.predict(feat_sc)[0])
    probs = clf.predict_proba(feat_sc)[0].tolist()
    conf = float(max(probs))

    # 图像分析信息
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean = float(hsv[:, :, 0].mean())
    s_mean = float(hsv[:, :, 1].mean())
    v_mean = float(hsv[:, :, 2].mean())
    b_mean = float(img[:, :, 0].mean())
    g_mean = float(img[:, :, 1].mean())
    r_mean = float(img[:, :, 2].mean())

    return {
        'label': label,
        'confidence': round(conf, 4),
        'probabilities': probs,
        'image_analysis': {
            'h_mean': round(h_mean, 1),
            's_mean': round(s_mean, 1),
            'v_mean': round(v_mean, 1),
            'r_mean': round(r_mean, 1),
            'g_mean': round(g_mean, 1),
            'b_mean': round(b_mean, 1),
            'feature_dim': len(feat),
        }
    }