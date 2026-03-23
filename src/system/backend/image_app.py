# =============================================================================
#  葡萄成熟度识别系统 v2 —— 图像识别后端
#  文件：src/system/backend/image_app.py
#
#  技术说明：
#    - 用户上传葡萄 RGB 图片
#    - 提取 HSV 颜色直方图 + 统计特征（14维）
#    - 送入 RandomForest 图像分类器预测成熟度
#    - 同时保留原高光谱 CSV 预测接口
#
#  图像分类器训练数据：
#    基于葡萄色泽研究的颜色分布合成数据（1600样本）
#    + 利用原高光谱标签中各类色差均值校准颜色范围
#    实际使用时可替换为真实图片数据
#
#  运行：python image_app.py   → http://localhost:5000
# =============================================================================

import os, io, time, pickle, traceback, warnings
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
warnings.filterwarnings('ignore')

# ── 路径 ──────────────────────────────────────────────────────────────────────
BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BACKEND_DIR)))
MODEL_DIR    = os.path.join(PROJECT_ROOT, 'results', '模型结果')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'results', '聚类结果')
IMG_MODEL_PATH = os.path.join(MODEL_DIR, 'image_classifier.pkl')

app = Flask(__name__)

# ── CORS ─────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>',             methods=['OPTIONS'])
def options_handler(path=''):
    from flask import make_response
    r = make_response()
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r

# ── 常量 ──────────────────────────────────────────────────────────────────────
NAMES   = ['未成熟', '半成熟', '成熟', '过成熟']
PALETTE = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
ICONS   = ['🌱', '🍈', '🍇', '🍷']
TIPS    = [
    '果实颜色偏绿，糖度较低，建议继续等待14-21天',
    '颜色开始转变，部分成熟，建议等待7-14天',
    '颜色饱满，糖度最佳，可以采摘！',
    '颜色过深，糖分开始降解，建议立即采摘或加工',
]


# =============================================================================
#  图像特征提取
# =============================================================================

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
    mask = cv2.inRange(hsv,
                       np.array([0,  30,  30]),
                       np.array([180, 255, 240]))
    # 如果掩码太小（图像几乎全白），使用全图
    if mask.sum() < mask.size * 0.05:
        mask = np.ones(mask.shape, dtype=np.uint8) * 255

    # ── HSV 统计量 (6维) ──
    h = hsv[:,:,0][mask>0].astype(float)
    s = hsv[:,:,1][mask>0].astype(float)
    v = hsv[:,:,2][mask>0].astype(float)
    hsv_feats = np.array([
        h.mean(), h.std(),
        s.mean(), s.std(),
        v.mean(), v.std(),
    ])

    # ── BGR 统计量 (6维) ──
    bgr_feats = []
    for ch in range(3):
        px = img_bgr[:,:,ch][mask>0].astype(float)
        bgr_feats += [px.mean(), px.std()]
    bgr_feats = np.array(bgr_feats)

    # ── HSV 直方图 (18维) ──
    h_hist = cv2.calcHist([hsv],[0], mask,[8],[0,180]).flatten()
    s_hist = cv2.calcHist([hsv],[1], mask,[6],[0,256]).flatten()
    v_hist = cv2.calcHist([hsv],[2], mask,[4],[0,256]).flatten()
    total  = mask.sum() / 255.0 + 1e-6
    hist_feats = np.concatenate([h_hist/total, s_hist/total, v_hist/total])

    # ── 纹理特征：Sobel 梯度 (8维) ──
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    sobelx = cv2.Sobel(gray_masked, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_masked, cv2.CV_64F, 0, 1, ksize=3)
    mag    = np.sqrt(sobelx**2 + sobely**2)
    px_mag = mag[mask>0]
    px_sx  = np.abs(sobelx[mask>0])
    px_sy  = np.abs(sobely[mask>0])
    tex_feats = np.array([
        px_mag.mean(), px_mag.std(),
        px_sx.mean(),  px_sx.std(),
        px_sy.mean(),  px_sy.std(),
        px_mag.max() if len(px_mag) else 0,
        np.percentile(px_mag, 75) if len(px_mag) else 0,
    ])

    return np.concatenate([hsv_feats, bgr_feats, hist_feats, tex_feats])


# =============================================================================
#  图像分类器：训练 & 保存
# =============================================================================

def train_image_classifier():
    """
    用合成颜色数据训练图像分类器
    颜色分布基于：
      - 葡萄成熟度色泽研究文献
      - 本项目第一阶段色差指标统计（各类色差均值）
        未成熟(23.47) / 半成熟(10.59) / 成熟(5.30) / 过成熟(1.80)
    """
    print('  [图像分类器] 开始训练...', flush=True)
    rng = np.random.RandomState(42)
    N   = 400   # 每类样本数

    def gen_class(label):
        """生成指定成熟度等级的合成图像特征"""
        samples = []
        # 颜色参数（基于葡萄实际颜色分布）
        params = {
            0: dict(h=(65,12), s=(175,25), v=(155,25),  # 未成熟：绿色
                    b=(50,20),  g=(130,25), r=(60,20)),
            1: dict(h=(30,12),  s=(165,25), v=(135,25), # 半成熟：黄绿→红
                    b=(60,20),  g=(100,25), r=(110,25)),
            2: dict(h=(155,10), s=(158,22), v=(115,22), # 成熟：深紫红
                    b=(100,20), g=(50,20),  r=(140,20)),
            3: dict(h=(163,8),  s=(138,22), v=(85,20),  # 过成熟：暗紫近黑
                    b=(70,18),  g=(30,15),  r=(90,18)),
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
            img[:,:,0] = np.clip(rng.normal(b_val, 8, (50,50)), 0, 255)
            img[:,:,1] = np.clip(rng.normal(g_val, 8, (50,50)), 0, 255)
            img[:,:,2] = np.clip(rng.normal(r_val, 8, (50,50)), 0, 255)
            # 叠加HSV噪声
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
            hsv_img[:,:,0] = np.clip(hsv_img[:,:,0] + rng.normal(0,3,(50,50)), 0, 179)
            hsv_img[:,:,1] = np.clip(hsv_img[:,:,1] + rng.normal(0,10,(50,50)), 0, 255)
            hsv_img[:,:,2] = np.clip(hsv_img[:,:,2] + rng.normal(0,10,(50,50)), 0, 255)
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
    X_sc   = scaler.fit_transform(X)

    # 训练集/测试集分割
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, stratify=y, random_state=42)

    # 软投票集成（速度快，效果好）
    from sklearn.ensemble import VotingClassifier
    rf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                      max_depth=4, random_state=42)
    svm = SVC(C=10, gamma='scale', kernel='rbf', probability=True, random_state=42)
    clf = VotingClassifier(
        estimators=[('rf',rf),('gbm',gbm),('svm',svm)],
        voting='soft', weights=[0.4,0.3,0.3], n_jobs=-1
    )
    clf.fit(X_tr, y_tr)

    acc  = accuracy_score(y_te, clf.predict(X_te))
    cv5  = cross_val_score(clf, X_sc, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f'  [图像分类器] 测试集 Acc={acc:.4f}  5折CV={cv5.mean():.4f}±{cv5.std():.4f}')

    # 保存
    model_pkg = {'clf': clf, 'scaler': scaler, 'n_features': X.shape[1]}
    with open(IMG_MODEL_PATH, 'wb') as f:
        pickle.dump(model_pkg, f)
    print(f'  [图像分类器] 已保存 → results/模型结果/image_classifier.pkl')
    return model_pkg, acc, cv5.mean(), cv5.std()


# =============================================================================
#  全局模型加载
# =============================================================================

class SystemManager:
    def __init__(self):
        # 高光谱模型
        self.spec_model  = None
        self.spec_pca    = None
        self.spec_scaler = None
        self.wavelengths = None
        self.spec_loaded = False
        # 图像模型
        self.img_model   = None
        self.img_scaler  = None
        self.img_loaded  = False
        # 图像模型训练指标
        self.img_acc     = 0.0
        self.img_cv_mean = 0.0
        self.img_cv_std  = 0.0

    def load_all(self):
        self._load_spec_model()
        self._load_img_model()

    def _load_spec_model(self):
        try:
            with open(os.path.join(MODEL_DIR,'stacking_model.pkl'),'rb') as f:
                self.spec_model  = pickle.load(f)
            with open(os.path.join(MODEL_DIR,'pca_model.pkl'),'rb') as f:
                self.spec_pca    = pickle.load(f)
            with open(os.path.join(MODEL_DIR,'scaler_spec.pkl'),'rb') as f:
                self.spec_scaler = pickle.load(f)
            df = pd.read_csv(os.path.join(DATA_DIR,'聚类标签数据集.csv'), nrows=1)
            skip = {'序号','maturity_label','maturity_name'}
            self.wavelengths = [float(c) for c in df.columns if c not in skip]
            self.spec_loaded = True
            print(f'  [高光谱模型] 加载成功，{len(self.wavelengths)} 波段')
        except Exception as e:
            print(f'  [高光谱模型] 加载失败: {e}')

    def _load_img_model(self):
        if os.path.exists(IMG_MODEL_PATH):
            try:
                with open(IMG_MODEL_PATH,'rb') as f:
                    pkg = pickle.load(f)
                self.img_model  = pkg['clf']
                self.img_scaler = pkg['scaler']
                self.img_loaded = True
                print('  [图像模型] 加载已有模型')
                return
            except Exception as e:
                print(f'  [图像模型] 读取失败，重新训练: {e}')

        # 首次运行：训练并保存
        print('  [图像模型] 首次运行，开始训练...')
        pkg, acc, cv_m, cv_s = train_image_classifier()
        self.img_model   = pkg['clf']
        self.img_scaler  = pkg['scaler']
        self.img_loaded  = True
        self.img_acc     = acc
        self.img_cv_mean = cv_m
        self.img_cv_std  = cv_s

    def predict_image(self, img_bytes: bytes):
        """图像预测完整流程"""
        if not self.img_loaded:
            raise RuntimeError('图像模型未加载')
        # 解码图像
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('无法解码图像，请确认上传的是有效图片文件')
        # 缩放（统一处理尺寸）
        img = cv2.resize(img, (224, 224))
        # 提取特征
        feat = extract_image_features(img)
        feat = np.nan_to_num(feat, nan=0.0, posinf=255.0, neginf=0.0)
        feat_sc = self.img_scaler.transform(feat.reshape(1,-1))
        # 预测
        label = int(self.img_model.predict(feat_sc)[0])
        probs = self.img_model.predict_proba(feat_sc)[0].tolist()
        conf  = float(max(probs))
        # 返回图像分析信息（用于前端展示）
        hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_mean = float(hsv[:,:,0].mean())
        s_mean = float(hsv[:,:,1].mean())
        v_mean = float(hsv[:,:,2].mean())
        b_mean = float(img[:,:,0].mean())
        g_mean = float(img[:,:,1].mean())
        r_mean = float(img[:,:,2].mean())
        return {
            'label':  label,
            'name':   NAMES[label],
            'icon':   ICONS[label],
            'color':  PALETTE[label],
            'tip':    TIPS[label],
            'confidence': round(conf, 4),
            'probabilities': {NAMES[i]: round(p,4) for i,p in enumerate(probs)},
            'image_analysis': {
                'h_mean': round(h_mean,1),
                's_mean': round(s_mean,1),
                'v_mean': round(v_mean,1),
                'r_mean': round(r_mean,1),
                'g_mean': round(g_mean,1),
                'b_mean': round(b_mean,1),
                'feature_dim': len(feat),
            }
        }

    def predict_spectrum(self, X_raw: np.ndarray):
        """高光谱预测流程"""
        if not self.spec_loaded:
            raise RuntimeError('高光谱模型未加载')
        X_sg  = savgol_filter(X_raw, 11, 3, axis=1)
        mean  = X_sg.mean(axis=1, keepdims=True)
        std   = X_sg.std(axis=1,  keepdims=True)
        X_snv = (X_sg - mean) / (std + 1e-10)
        X_n   = self.spec_scaler.transform(X_snv)
        X_pca = self.spec_pca.transform(X_n)
        labels = self.spec_model.predict(X_pca).tolist()
        probs  = self.spec_model.predict_proba(X_pca)
        return labels, probs


mgr = SystemManager()


# =============================================================================
#  API 端点
# =============================================================================

@app.route('/api/health')
def health():
    return jsonify({
        'status':       'ok',
        'spec_model':   mgr.spec_loaded,
        'img_model':    mgr.img_loaded,
        'wavelengths':  len(mgr.wavelengths) if mgr.wavelengths else 0,
        'img_accuracy': round(mgr.img_acc, 4),
        'img_cv':       f'{mgr.img_cv_mean:.4f}±{mgr.img_cv_std:.4f}',
        'model_info': {
            'image':    'Stacking(RF+GBM+SVM) → 38维颜色纹理特征',
            'spectrum': 'Stacking(SVM+RF+GBM+BP) → SG+SNV+PCA预处理',
        }
    })


@app.route('/api/predict/image', methods=['POST'])
def predict_image():
    """图像预测：上传一张葡萄图片"""
    if 'image' not in request.files:
        return jsonify({'error': '请上传图片文件（字段名: image）'}), 400

    file = request.files['image']
    ext  = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in {'jpg','jpeg','png','bmp','webp'}:
        return jsonify({'error': '仅支持 jpg/png/bmp/webp 格式'}), 400

    try:
        t0  = time.time()
        res = mgr.predict_image(file.read())
        res['elapsed_ms'] = round((time.time()-t0)*1000, 1)
        res['success']    = True
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/file', methods=['POST'])
def predict_file():
    """高光谱CSV文件预测"""
    if not mgr.spec_loaded:
        return jsonify({'error': '高光谱模型未就绪'}), 503
    if 'file' not in request.files:
        return jsonify({'error': '请上传 CSV 文件'}), 400
    file = request.files['file']
    try:
        df       = pd.read_csv(io.StringIO(file.read().decode('utf-8-sig')))
        skip     = {'序号','maturity_label','maturity_name'}
        wave_cols = [c for c in df.columns if c not in skip]
        X_raw    = df[wave_cols].values.astype(float)
        X_raw    = np.nan_to_num(X_raw, nan=np.nanmean(X_raw))
        t0       = time.time()
        labels, probs = mgr.predict_spectrum(X_raw)
        elapsed  = time.time() - t0
        dist     = dict(Counter(labels))
        results  = []
        ids      = df['序号'].tolist() if '序号' in df.columns else list(range(1,len(labels)+1))
        for i, (lb, pr) in enumerate(zip(labels, probs)):
            results.append({
                'sample_id': ids[i], 'label': lb,
                'name': NAMES[lb], 'color': PALETTE[lb],
                'confidence': round(float(pr.max()),4),
                'probabilities': {NAMES[j]: round(float(pr[j]),4) for j in range(4)},
            })
        return jsonify({
            'success': True, 'count': len(results),
            'elapsed_ms': round(elapsed*1000,1),
            'distribution': {NAMES[k]: dist.get(k,0) for k in range(4)},
            'results': results,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def stats():
    try:
        df   = pd.read_csv(os.path.join(DATA_DIR,'指标数据_含标签.csv'))
        dist = df['maturity_label'].value_counts().sort_index().to_dict()
        return jsonify({
            'total_samples': len(df),
            'distribution':  {NAMES[k]: dist.get(k,0) for k in range(4)},
            'model_metrics': {
                'spectrum_accuracy': 0.8773,
                'spectrum_f1':       0.8775,
                'spectrum_cv':       '0.8736±0.0112',
                'image_accuracy':    round(mgr.img_acc,4),
                'image_cv':          f'{mgr.img_cv_mean:.4f}±{mgr.img_cv_std:.4f}',
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import io
    print('\n' + '='*56)
    print('  葡萄成熟度识别系统 v2 · 图像+高光谱双模式')
    print('='*56)
    mgr.load_all()
    print(f'\n  http://localhost:5000')
    app.run(debug=False, host='0.0.0.0', port=5000)