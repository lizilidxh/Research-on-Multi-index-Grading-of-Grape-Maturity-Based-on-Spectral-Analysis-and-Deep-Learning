# =============================================================================
#  葡萄成熟度识别系统 v2 —— 配置常量
# =============================================================================
import os

# ── 路径配置 ─────────────────────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BACKEND_DIR)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'results', 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'results', 'cluster')
IMG_MODEL_PATH = os.path.join(MODEL_DIR, 'image_classifier.pkl')

# ── 业务常量 ─────────────────────────────────────────────────────────────────
# 成熟度等级相关
NAMES = ['未成熟', '半成熟', '成熟', '过成熟']
PALETTE = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
ICONS = ['🌱', '🍈', '🍇', '🍷']
TIPS = [
    '果实颜色偏绿，糖度较低，建议继续等待14-21天',
    '颜色开始转变，部分成熟，建议等待7-14天',
    '颜色饱满，糖度最佳，可以采摘！',
    '颜色过深，糖分开始降解，建议立即采摘或加工',
]

# 图像特征配置
IMG_RESIZE_SIZE = (224, 224)
HSV_MASK_LOW = np.array([0, 30, 30])
HSV_MASK_HIGH = np.array([180, 255, 240])
MASK_MIN_RATIO = 0.05  # 最小有效掩码比例

# 模型训练配置
TRAIN_SAMPLE_PER_CLASS = 400  # 每类合成样本数
RANDOM_SEED = 42  # 随机种子
TEST_SIZE = 0.2   # 测试集比例

# 允许的图片格式
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}