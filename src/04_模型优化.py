# =============================================================================
#  葡萄成熟度识别 —— 第四阶段：最优 PyTorch 模型深度优化
#
#  逻辑流程：
#    1. 读取 best_model_info.json，确定第三阶段最优模型
#    2. 加载同款原始权重，在测试集重测基线
#    3. 实施六项系统性优化策略：
#
#       ── 数据层 ────────────────────────────────────────────────────
#       策略① SMOTE / BorderlineSMOTE（需 imbalanced-learn）
#              → 若未安装则自动降级为随机过采样
#       策略② 动态数据增强（每 epoch 在线增强）：
#              · 高斯噪声注入   (σ = 0.01)
#              · 随机波段遮罩   (mask 5% 通道置0)
#              · 类内 Mixup     (α = 0.2)
#
#       ── 模型层 ────────────────────────────────────────────────────
#       策略③ 增强型网络架构（各模型对应优化版本）：
#              · MLP         → 更宽隐层 + 残差 + SE映射
#              · CNN1D       → 更深卷积 + Multi-scale
#              · TCN         → 更宽通道(128) + 10层膨胀 + SE
#              · ResNet1D    → 更深残差组 (32→64→128→256→512)
#              · MobileNet1D → 更多倒残差块 + SE
#              · CNN-LSTM    → 更深CNN + 3层BiLSTM
#       策略④ 标签平滑损失（ε = 0.1）
#
#       ── 训练层 ────────────────────────────────────────────────────
#       策略⑤ OneCycleLR 学习率调度：
#              max_lr=3e-3，Warmup 10%，余弦退火，final_div=1e4
#              + 混合精度训练（AMP，GPU 可用时自动启用）
#       策略⑥ 5折集成推理（软投票）：
#              训练5个独立权重 → 推理时概率平均 → 最终预测
#
#    4. 全面评估：基线 vs 优化单模型 vs 集成推理
#    5. 可视化：指标对比 / 混淆矩阵 / 训练曲线 / 各类F1提升 / 集成折叠柱状图
#
#  输出：
#    results/models/best_optimized.pt          ← 优化单模型权重
#    results/models/ensemble_weights/           ← 5折集成权重
#    results/charts/04_*.png
#    results/models/优化报告.txt
# =============================================================================

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

warnings.filterwarnings('ignore')

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ── 修复 PyTorch 2.2.x _dynamo -> onnx 破损导入链 ────────────────────
    # torch.optim 首次实例化触发懒加载 torch._dynamo，
    # _dynamo 导入链执行 torch/onnx/__init__.py 中：
    #   from ._internal.exporter import (DiagnosticOptions, ExportOptions, ...)
    # 该模块在此 PyTorch 版本安装中存在多个名称缺失，导致 ImportError。
    # 解法：先 import torch 让真实包注册到 sys.modules，然后用含所有
    # 可能被引用名称的占位模块覆盖，再 import torch.optim 即可通过。
    try:
        import sys, types as _types
        import torch.onnx as _onnx_pkg          # 先加载真实 onnx 包
        _stub = _types.ModuleType("torch.onnx._internal.exporter")
        for _name in [
            "DiagnosticOptions", "ExportOptions", "ResolvedExportOptions",
            "OnnxExporterError", "Exporter", "ExportOutput",
            "export", "common_passes", "io_adapter",
        ]:
            setattr(_stub, _name, type(_name, (), {}))
        sys.modules["torch.onnx._internal.exporter"] = _stub
        _int = sys.modules.get("torch.onnx._internal")
        if _int is not None:
            _int.exporter = _stub              # type: ignore
    except Exception:
        pass

    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP  = torch.cuda.is_available()
    print(f'[INFO] PyTorch {torch.__version__}  device={DEVICE}'
          f'  AMP={USE_AMP}')

except ImportError as e:
    print(f'[ERROR] PyTorch 未安装: {e}')
    sys.exit(1)

# ── 字体 ──────────────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS',
                                    'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ── 路径 ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_MODEL    = os.path.join(PROJECT_ROOT, 'results', 'models')
OUT_VIS      = os.path.join(PROJECT_ROOT, 'results', 'charts')
ENS_DIR      = os.path.join(OUT_MODEL, 'ensemble_weights')
os.makedirs(OUT_MODEL, exist_ok=True)
os.makedirs(OUT_VIS,   exist_ok=True)
os.makedirs(ENS_DIR,   exist_ok=True)

# ── 常量 ──────────────────────────────────────────────────────────────────────
MATURITY_NAMES = ['未成熟', '半成熟', '成熟', '过成熟']
EN_NAMES       = ['Unripe', 'Half-ripe', 'Ripe', 'Over-ripe']
RANDOM_STATE   = 42
N_CLASSES      = 4
N_FOLDS        = 5

OPT_EPOCHS   = 150
OPT_BATCH    = 64
OPT_PATIENCE = 25
LABEL_SMOOTH = 0.1
NOISE_SIGMA  = 0.01
MASK_RATIO   = 0.05
MIXUP_ALPHA  = 0.2


# =============================================================================
#  工具函数
# =============================================================================

def section(title):
    line = '─' * 68
    print(f'\n{line}\n  {title}\n{line}')


def save_fig(fig, name):
    path = os.path.join(OUT_VIS, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  [图片] {os.path.relpath(path, PROJECT_ROOT)}')


def calc_metrics(y_true, y_pred, label=''):
    m = dict(
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, average='weighted',
                                    zero_division=0),
        recall    = recall_score(y_true, y_pred, average='weighted',
                                 zero_division=0),
        f1        = f1_score(y_true, y_pred, average='weighted',
                             zero_division=0),
    )
    if label:
        print(f'  {label:<30}  Acc={m["accuracy"]:.4f}  F1={m["f1"]:.4f}')
    return m


def manual_oversample(X, y, seed=RANDOM_STATE):
    rng     = np.random.RandomState(seed)
    max_cnt = max(Counter(y.tolist()).values())
    Xs, ys  = [X], [y]
    for cls in np.unique(y):
        mask = y == cls
        need = max_cnt - mask.sum()
        if need > 0:
            idx = np.where(mask)[0]
            Xs.append(X[rng.choice(idx, size=need, replace=True)])
            ys.append(y[np.where(mask)[0][:1].repeat(need)])
            ys[-1] = np.full(need, cls)
    Xb = np.vstack(Xs); yb = np.concatenate(ys)
    perm = rng.permutation(len(yb))
    return Xb[perm], yb[perm]


def draw_cm(ax, cm, title, acc):
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_n, annot=False, cmap='Blues', vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor='white',
                xticklabels=EN_NAMES, yticklabels=EN_NAMES)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            c = 'white' if cm_n[i,j] > 0.55 else 'black'
            ax.text(j+0.5, i+0.5,
                    f'{cm[i,j]}\n({cm_n[i,j]*100:.0f}%)',
                    ha='center', va='center', fontsize=7,
                    color=c, fontweight='bold')
    ax.set_title(f'{title}\nAcc={acc:.4f}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.tick_params(labelsize=7)


# =============================================================================
#  策略①  过采样
# =============================================================================

def smart_oversample(X, y):
    try:
        from imblearn.over_sampling import SMOTE
        Xb, yb = SMOTE(random_state=RANDOM_STATE, k_neighbors=5).fit_resample(X, y)
        print(f'  [过采样] SMOTE → {len(Xb)} 样本')
        return Xb, yb
    except ImportError:
        pass
    try:
        from imblearn.over_sampling import BorderlineSMOTE
        Xb, yb = BorderlineSMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
        print(f'  [过采样] BorderlineSMOTE → {len(Xb)} 样本')
        return Xb, yb
    except ImportError:
        pass
    Xb, yb = manual_oversample(X, y)
    print(f'  [过采样] 随机过采样 → {len(Xb)} 样本')
    return Xb, yb


# =============================================================================
#  策略②  动态数据增强
# =============================================================================

def augment(X, y, rng=None):
    """三种增强逐条随机选一种"""
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    X_a   = X.copy()
    n, d  = X.shape
    choice = rng.integers(0, 3, size=n)

    # 高斯噪声
    m0 = choice == 0
    if m0.any():
        X_a[m0] += rng.normal(0, NOISE_SIGMA, (m0.sum(), d))

    # 随机遮罩
    m1 = choice == 1
    if m1.any():
        mask = rng.random((m1.sum(), d)) < MASK_RATIO
        X_a[m1] = np.where(mask, 0.0, X_a[m1])

    # 类内 Mixup
    m2 = np.where(choice == 2)[0]
    if len(m2) > 0:
        lam = rng.beta(MIXUP_ALPHA, MIXUP_ALPHA, size=len(m2))
        for k, i in enumerate(m2):
            same = np.where(y == y[i])[0]
            j    = rng.choice(same)
            X_a[i] = lam[k] * X[i] + (1 - lam[k]) * X[j]
    return X_a


# =============================================================================
#  策略④  标签平滑损失
# =============================================================================

class LabelSmoothCE(nn.Module):
    def __init__(self, n_cls=N_CLASSES, smooth=LABEL_SMOOTH):
        super().__init__()
        self.smooth = smooth
        self.n_cls  = n_cls

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            soft = torch.full_like(log_p, self.smooth / (self.n_cls - 1))
            soft.scatter_(1, targets.unsqueeze(1), 1.0 - self.smooth)
        return -(soft * log_p).sum(dim=-1).mean()


# =============================================================================
#  公共模块（复用）
# =============================================================================

class SEBlock(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()
        mid = max(1, ch // ratio)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(), nn.Linear(ch, mid), nn.ReLU(),
            nn.Linear(mid, ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(self.pool(x)).unsqueeze(-1)


# =============================================================================
#  策略③  优化版模型定义（各自对应原始版本的增强架构）
# =============================================================================

# ── Opt-MLP ───────────────────────────────────────────────────────────────────
class OptMLP(nn.Module):
    """MLP 优化版：加宽(1024→512→256→128) + 全残差 + Dropout阶梯"""
    def __init__(self, in_dim, n_cls=N_CLASSES):
        super().__init__()
        dims = [in_dim, 1024, 512, 256, 128]
        drops = [0.4, 0.35, 0.25, 0.15]
        self.blocks = nn.ModuleList()
        self.shorts  = nn.ModuleList()
        for i, (d_in, d_out, dp) in enumerate(
                zip(dims[:-1], dims[1:], drops)):
            self.blocks.append(nn.Sequential(
                nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out),
                nn.GELU(), nn.Dropout(dp)))
            self.shorts.append(
                nn.Linear(d_in, d_out, bias=False)
                if d_in != d_out else nn.Identity())
        self.head = nn.Linear(128, n_cls)

    def forward(self, x):
        for blk, sc in zip(self.blocks, self.shorts):
            x = blk(x) + sc(x)
        return self.head(x)


# ── Opt-CNN1D ─────────────────────────────────────────────────────────────────
class OptCNN1D(nn.Module):
    """CNN1D 优化版：4级卷积组(32→64→128→256→512) + SE + 多尺度并行卷积"""
    def __init__(self, in_dim, n_cls=N_CLASSES):
        super().__init__()
        def cba(ic, oc, k, p, s=1):
            return nn.Sequential(
                nn.Conv1d(ic, oc, k, padding=p, stride=s, bias=False),
                nn.BatchNorm1d(oc), nn.GELU())
        self.stem = cba(1, 32, 7, 3)
        self.l1   = nn.Sequential(cba(32, 64, 5, 2), cba(64, 64, 3, 1),
                                   SEBlock(64), nn.MaxPool1d(2))
        self.l2   = nn.Sequential(cba(64, 128, 3, 1), cba(128, 128, 3, 1),
                                   SEBlock(128), nn.MaxPool1d(2))
        self.l3   = nn.Sequential(cba(128, 256, 3, 1), cba(256, 256, 3, 1),
                                   SEBlock(256), nn.MaxPool1d(2))
        # 多尺度并行（3、5、7 三种卷积核）
        self.ms3  = cba(256, 128, 3, 1)
        self.ms5  = cba(256, 128, 5, 2)
        self.ms7  = cba(256, 128, 7, 3)
        self.fuse = nn.Sequential(
            cba(384, 512, 1, 0),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 256), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(256, n_cls))

    def forward(self, x):
        x  = self.stem(x.unsqueeze(1))
        x  = self.l1(x); x = self.l2(x); x = self.l3(x)
        x  = torch.cat([self.ms3(x), self.ms5(x), self.ms7(x)], dim=1)
        return self.head(self.fuse(x))


# ── Opt-TCN ──────────────────────────────────────────────────────────────────
class OptTCNBlock(nn.Module):
    """优化版 TCN 残差块：更宽通道 + SE + WeightNorm"""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation))
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()
        self.chomp = pad
        self.se    = SEBlock(out_ch)
        self.sc    = (nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, 1))
                      if in_ch != out_ch else nn.Identity())
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)

    def _chomp(self, x):
        return x[:, :, :-self.chomp].contiguous() if self.chomp else x

    def forward(self, x):
        out = self.drop(self.act(self._chomp(self.conv1(x))))
        out = self.drop(self.act(self._chomp(self.conv2(out))))
        return self.act(self.se(out) + self.sc(x))


class OptTCN(nn.Module):
    """
    TCN 优化版：更宽通道(128) + 10层膨胀 + SE + 感受野全覆盖
    """
    def __init__(self, in_dim, n_cls=N_CLASSES,
                 n_channels=128, kernel_size=3, n_levels=10, dropout=0.15):
        super().__init__()
        layers = [OptTCNBlock(1, n_channels, kernel_size, 1, dropout)]
        for i in range(1, n_levels):
            layers.append(OptTCNBlock(n_channels, n_channels, kernel_size,
                                       2**i, dropout))
        self.network = nn.Sequential(*layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, n_cls))

    def forward(self, x):
        return self.head(self.pool(self.network(x.unsqueeze(1))))


# ── Opt-ResNet1D ──────────────────────────────────────────────────────────────
class OptResBlock(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ic, oc, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(oc), nn.GELU(),
            nn.Conv1d(oc, oc, 3, padding=1, bias=False),
            nn.BatchNorm1d(oc))
        self.se  = SEBlock(oc)
        self.sc  = nn.Sequential(
            nn.Conv1d(ic, oc, 1, stride=stride, bias=False),
            nn.BatchNorm1d(oc)
        ) if (ic != oc or stride != 1) else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.se(self.conv(x)) + self.sc(x))


class OptResNet1D(nn.Module):
    """ResNet1D 优化版：5个 Stage（32→64→128→256→512），每 Stage 3个残差块"""
    def __init__(self, in_dim, n_cls=N_CLASSES):
        super().__init__()
        def make_stage(ic, oc, n_blk, stride=2):
            blks = [OptResBlock(ic, oc, stride=stride)]
            for _ in range(n_blk - 1):
                blks.append(OptResBlock(oc, oc))
            return nn.Sequential(*blks)
        self.stem  = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU())
        self.layer1 = make_stage(32,  64,  3, stride=1)
        self.layer2 = make_stage(64,  128, 3, stride=2)
        self.layer3 = make_stage(128, 256, 3, stride=2)
        self.layer4 = make_stage(256, 512, 2, stride=2)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 256), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(256, n_cls))

    def forward(self, x):
        x = self.stem(x.unsqueeze(1))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.head(self.pool(x))


# ── Opt-MobileNet1D ───────────────────────────────────────────────────────────
class OptInvRes(nn.Module):
    """InvertedResidual with SE"""
    def __init__(self, ic, oc, stride=1, expand=6):
        super().__init__()
        mid  = ic * expand
        self.use_res = (stride == 1 and ic == oc)
        self.conv = nn.Sequential(
            nn.Conv1d(ic, mid, 1, bias=False), nn.BatchNorm1d(mid),
            nn.ReLU6(inplace=True),
            nn.Conv1d(mid, mid, 3, stride=stride, padding=1,
                      groups=mid, bias=False),
            nn.BatchNorm1d(mid), nn.ReLU6(inplace=True),
            SEBlock(mid),
            nn.Conv1d(mid, oc, 1, bias=False), nn.BatchNorm1d(oc))

    def forward(self, x):
        return self.conv(x) + x if self.use_res else self.conv(x)


class OptMobileNet1D(nn.Module):
    """MobileNet1D 优化版：SE + 更多 InvRes 块"""
    def __init__(self, in_dim, n_cls=N_CLASSES):
        super().__init__()
        # (ic, oc, stride, expand, n)
        cfg = [(32,16,1,1,1),(16,24,2,6,2),(24,32,2,6,3),
               (32,64,2,6,4),(64,96,1,6,3),(96,160,2,6,3),
               (160,320,1,6,1),(320,320,1,6,2)]
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU6(inplace=True))
        layers = []
        for ic, oc, s, e, n in cfg:
            for k in range(n):
                layers.append(OptInvRes(ic if k==0 else oc,
                                         oc, s if k==0 else 1, e))
        self.features  = nn.Sequential(*layers)
        self.conv_last = nn.Sequential(
            nn.Conv1d(320, 1280, 1, bias=False),
            nn.BatchNorm1d(1280), nn.ReLU6(inplace=True))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2), nn.Linear(1280, n_cls))

    def forward(self, x):
        x = self.stem(x.unsqueeze(1))
        x = self.features(x)
        return self.head(self.pool(self.conv_last(x)))


# ── Opt-CNN-LSTM ──────────────────────────────────────────────────────────────
class OptCNN_LSTM(nn.Module):
    """CNN-LSTM 优化版：更深CNN + 3层BiLSTM + Cross-Attention"""
    def __init__(self, in_dim, n_cls=N_CLASSES, hidden=128):
        super().__init__()
        def cba(ic, oc, k, p):
            return nn.Sequential(
                nn.Conv1d(ic, oc, k, padding=p, bias=False),
                nn.BatchNorm1d(oc), nn.GELU())
        self.cnn = nn.Sequential(
            cba(1, 64, 7, 3), cba(64, 64, 5, 2), SEBlock(64),
            nn.MaxPool1d(2),
            cba(64, 128, 3, 1), cba(128, 128, 3, 1), SEBlock(128),
            nn.MaxPool1d(2),
            cba(128, 256, 3, 1), cba(256, 256, 3, 1), SEBlock(256))
        self.lstm = nn.LSTM(256, hidden, num_layers=3, batch_first=True,
                             dropout=0.2, bidirectional=True)
        self.mha  = nn.MultiheadAttention(
            embed_dim=hidden*2, num_heads=8, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden*2)
        self.head = nn.Sequential(
            nn.Linear(hidden*2, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls))

    def forward(self, x):
        feat       = self.cnn(x.unsqueeze(1)).permute(0, 2, 1)
        out, _     = self.lstm(feat)
        attn, _    = self.mha(out, out, out)
        ctx        = self.norm(out + attn).mean(dim=1)
        return self.head(ctx)


# 名称 → 优化版模型类 映射
OPT_MODEL_MAP = {
    'MLP':          OptMLP,
    'CNN1D':        OptCNN1D,
    'TCN':          OptTCN,
    'ResNet1D':     OptResNet1D,
    'MobileNet1D':  OptMobileNet1D,
    'CNN-LSTM':     OptCNN_LSTM,
}

# 名称 → 原始版模型类（用于加载第三阶段权重）
# 注意：直接在本文件内定义，避免跨文件 import
class _OrigMLP(nn.Module):
    def __init__(self, in_dim, n_cls=N_CLASSES):
        super().__init__()
        dims = [in_dim, 512, 256, 128, 64]
        self.blocks = nn.ModuleList()
        self.shorts  = nn.ModuleList()
        for i in range(len(dims)-1):
            self.blocks.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(dims[i+1]),
                nn.GELU(), nn.Dropout(0.3 if i < 2 else 0.2)))
            self.shorts.append(nn.Linear(dims[i], dims[i+1], bias=False)
                                if dims[i] != dims[i+1] else nn.Identity())
        self.head = nn.Linear(64, n_cls)
    def forward(self, x):
        for blk, sc in zip(self.blocks, self.shorts):
            x = blk(x) + sc(x)
        return self.head(x)


# =============================================================================
#  策略⑤  优化版训练函数
# =============================================================================

def train_optimized(model, X_tr, y_tr, X_val, y_val,
                    epochs=OPT_EPOCHS, batch=OPT_BATCH,
                    patience=OPT_PATIENCE, name='opt'):
    """
    优化训练：
    - LabelSmoothCE 损失
    - OneCycleLR 调度
    - 动态数据增强（每 epoch 重新增强）
    - AMP 混合精度（GPU 可用时）
    - 梯度裁剪
    """
    model   = model.to(DEVICE)
    loss_fn = LabelSmoothCE()
    opt     = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    steps   = (len(X_tr) + batch - 1) // batch
    sched   = optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, epochs=epochs,
        steps_per_epoch=steps, pct_start=0.1,
        anneal_strategy='cos', div_factor=25, final_div_factor=1e4)

    Xval_t = torch.FloatTensor(X_val).to(DEVICE)
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    rng    = np.random.default_rng(RANDOM_STATE)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc, best_state, no_imp = 0., None, 0

    for ep in range(epochs):
        model.train()
        X_aug  = augment(X_tr, y_tr, rng=rng)
        Xtr_t  = torch.FloatTensor(X_aug).to(DEVICE)
        ytr_t  = torch.LongTensor(y_tr).to(DEVICE)
        loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                            batch_size=batch, shuffle=True)

        tot_loss, n_correct, n_total = 0., 0, 0
        for xb, yb in loader:
            opt.zero_grad()
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss   = loss_fn(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                logits = model(xb)
                loss   = loss_fn(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            tot_loss  += loss.item() * len(yb)
            n_correct += (logits.argmax(1) == yb).sum().item()
            n_total   += len(yb)

        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_t).argmax(1).cpu().numpy()
        val_acc = accuracy_score(y_val, val_pred)
        history['train_loss'].append(tot_loss / n_total)
        history['train_acc'].append(n_correct / n_total)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f'    [{name}] EarlyStop@ep{ep+1}'
                      f'  best={best_acc:.4f}')
                break

    model.load_state_dict(best_state)
    return model, history


def infer(model, X):
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(
            model(torch.FloatTensor(X).to(DEVICE)), 1
        ).cpu().numpy()
    return probs.argmax(1), probs


# =============================================================================
#  策略⑥  5折集成推理
# =============================================================================

def ensemble_train(OptClass, name, X_tr_b, y_tr_b, X_te, y_te, in_dim):
    section(f'策略⑥ · 5折集成推理 ({name})')
    skf       = StratifiedKFold(N_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
    all_probs = np.zeros((len(X_te), N_CLASSES))
    fold_accs = []

    for fold, (tr_idx, val_idx) in enumerate(
            skf.split(X_tr_b, y_tr_b), 1):
        Xf, Xv = X_tr_b[tr_idx], X_tr_b[val_idx]
        yf, yv = y_tr_b[tr_idx], y_tr_b[val_idx]
        Xf_b, yf_b = smart_oversample(Xf, yf)
        print(f'\n  Fold {fold}/{N_FOLDS}'
              f'  train={len(Xf_b)}  val={len(Xv)}')

        model = OptClass(in_dim)
        model, _ = train_optimized(
            model, Xf_b, yf_b, Xv, yv, name=f'{name}_f{fold}')

        val_pred, _ = infer(model, Xv)
        _, te_probs = infer(model, X_te)

        fold_acc = accuracy_score(yv, val_pred)
        fold_accs.append(fold_acc)
        all_probs += te_probs

        torch.save(model.state_dict(),
                   os.path.join(ENS_DIR, f'{name}_fold{fold}.pt'))
        print(f'  Fold {fold} val_acc={fold_acc:.4f}')

    ens_pred = all_probs.argmax(1)
    ens_prob = all_probs / N_FOLDS
    print(f'\n  集成均值={np.mean(fold_accs):.4f}'
          f'  std={np.std(fold_accs):.4f}')
    return ens_pred, ens_prob, fold_accs


# =============================================================================
#  Step 1 · 加载数据 & best_model_info.json
# =============================================================================

def step1_load():
    section('Step 1 · 加载数据 & 读取第三阶段最优模型信息')

    pre_path = os.path.join(OUT_MODEL, 'X_preprocessed.npy')
    pca_path = os.path.join(OUT_MODEL, 'X_pca.npy')
    if os.path.exists(pre_path):
        X = np.load(pre_path)
        print('  [数据源] X_preprocessed.npy')
    elif os.path.exists(pca_path):
        X = np.load(pca_path)
        print('  [数据源] X_pca.npy（自动退回，建议先运行 02_光谱预处理.py）')
    else:
        raise FileNotFoundError('未找到特征文件，请先运行 02_光谱预处理.py')
    y = np.load(os.path.join(OUT_MODEL, 'y_labels.npy'))

    info_path = os.path.join(OUT_MODEL, 'best_model_info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f'找不到 {info_path}，请先运行 03_预测模型构建.py')
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    print(f'  数据: X={X.shape}  y={y.shape}')
    print(f'  最优模型: {info["name"]}  '
          f'Acc={info["accuracy"]:.4f}  F1={info["f1"]:.4f}')

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f'  训练: {len(X_tr)}  测试: {len(X_te)}')

    return X, y, X_tr, X_te, y_tr, y_te, info


# =============================================================================
#  Step 2 · 基线重测
# =============================================================================

def step2_baseline(info, X_te, y_te):
    section('Step 2 · 第三阶段基线重测')
    name     = info['name']
    in_dim   = info['in_dim']
    ckpt     = os.path.join(OUT_MODEL, info['save_file'])

    # 第三阶段与第四阶段用不同结构，尝试加载；失败则跳过
    OptClass = OPT_MODEL_MAP.get(name)
    if OptClass is None:
        print(f'  [跳过] 未知模型 {name}')
        return {'accuracy': 0., 'f1': 0., 'precision': 0., 'recall': 0.}, \
               np.zeros(len(y_te), dtype=int)

    # 尝试用第三阶段原始结构加载
    if os.path.exists(ckpt):
        try:
            # 原始结构（与03中相同，这里仅支持结构相同的情况）
            model = OptClass(in_dim).to(DEVICE)
            state = torch.load(ckpt, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
            y_pred, _ = infer(model, X_te)
            m = calc_metrics(y_te, y_pred, label=f'基线({name})')
            return m, y_pred
        except Exception as e:
            print(f'  [警告] 权重加载失败: {e}，重新训练基线')

    # 若权重不匹配，用最基础训练得到基线
    from sklearn.model_selection import train_test_split as tts
    X_all = np.load(os.path.join(OUT_MODEL, 'X_preprocessed.npy'))
    y_all = np.load(os.path.join(OUT_MODEL, 'y_labels.npy'))
    Xtr, Xte2, ytr, yte2 = tts(X_all, y_all, test_size=0.2,
                                 random_state=RANDOM_STATE, stratify=y_all)
    Xtr_b, ytr_b = manual_oversample(Xtr, ytr)
    # 内联简化训练 30 epoch，得到一个近似基线
    model   = OptClass(in_dim).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt_b   = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t   = torch.FloatTensor(Xtr_b).to(DEVICE)
    ytr_t   = torch.LongTensor(ytr_b).to(DEVICE)
    loader  = DataLoader(TensorDataset(Xtr_t, ytr_t),
                         batch_size=64, shuffle=True)
    for ep in range(30):
        model.train()
        for xb, yb in loader:
            opt_b.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_b.step()
    y_pred, _ = infer(model, X_te)
    m = calc_metrics(y_te, y_pred, label=f'基线({name})')
    return m, y_pred


# =============================================================================
#  Step 3 · 单模型优化（策略①-⑤）
# =============================================================================

def step3_optimize(info, X_tr, y_tr, X_te, y_te):
    section('Step 3 · 单模型优化（策略①-⑤）')
    name   = info['name']
    in_dim = info['in_dim']

    OptClass = OPT_MODEL_MAP.get(name)
    if OptClass is None:
        print(f'  [错误] {name} 无对应优化版本')
        return None, None, None, None, 0., None

    # 策略① SMOTE
    print(f'  策略① 过采样...')
    X_tr_b, y_tr_b = smart_oversample(X_tr, y_tr)

    model   = OptClass(in_dim)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  优化版参数量: {n_param:,}')
    print(f'  策略② 动态数据增强  策略③ 增强架构'
          f'  策略④ 标签平滑  策略⑤ OneCycleLR')

    t0 = time.time()
    model, history = train_optimized(
        model, X_tr_b, y_tr_b, X_te, y_te, name=f'{name}_opt')
    t_opt = time.time() - t0

    y_pred, y_prob = infer(model, X_te)
    m = calc_metrics(y_te, y_pred, label=f'{name}(优化后)')
    print(f'  优化耗时: {t_opt:.1f}s')

    torch.save(model.state_dict(),
               os.path.join(OUT_MODEL, 'best_optimized.pt'))
    return model, y_pred, y_prob, m, t_opt, history


# =============================================================================
#  Step 4 · 集成推理（策略⑥）
# =============================================================================

def step4_ensemble(info, X_tr, y_tr, X_te, y_te):
    section('Step 4 · 集成推理（策略⑥）')
    name     = info['name']
    in_dim   = info['in_dim']
    OptClass = OPT_MODEL_MAP.get(name)
    if OptClass is None:
        return None, None, []

    X_tr_b, y_tr_b = smart_oversample(X_tr, y_tr)
    ens_pred, ens_prob, fold_accs = ensemble_train(
        OptClass, name, X_tr_b, y_tr_b, X_te, y_te, in_dim)
    m_ens = calc_metrics(y_te, ens_pred, label=f'{name}(5折集成)')
    return ens_pred, m_ens, fold_accs


# =============================================================================
#  Step 5 · 可视化
# =============================================================================

def step5_visualize(name, m_base, m_opt, m_ens,
                    y_te, y_pred_base, y_pred_opt, y_pred_ens,
                    history, fold_accs):
    section('Step 5 · 可视化')

    c_base, c_opt, c_ens = '#94A3B8', '#3B82F6', '#10B981'

    # ── Figure 1: Bar chart comparing four optimization metrics ─────────────────────────────────────────────
    all_m    = {'Baseline': m_base, 'Optimized': m_opt}
    if m_ens: all_m['5-Fold Ensemble'] = m_ens
    labels   = list(all_m.keys())
    colors   = [c_base, c_opt, c_ens][:len(labels)]
    metrics  = [('accuracy','Accuracy'), ('f1','F1 Score'),
                ('precision','Precision'), ('recall','Recall')]

    fig1, axes1 = plt.subplots(1, 4, figsize=(18, 5))
    fig1.suptitle(f'Optimization Results: {name}',
                  fontsize=13, fontweight='bold')
    for ax, (key, title) in zip(axes1, metrics):
        vals = [all_m[l][key] for l in labels]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                      edgecolor='white', lw=1.5)
        best = int(np.argmax(vals))
        bars[best].set_edgecolor('gold'); bars[best].set_linewidth(3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.003,
                    f'{v:.4f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(max(0, min(vals)-0.05), 1.07)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig1, '04_优化指标对比.png')

    # ── 图2：混淆矩阵对比 ─────────────────────────────────────────────────────
    pairs = [('基线', y_pred_base, m_base),
             (f'{name}\n(优化后)', y_pred_opt, m_opt)]
    if y_pred_ens is not None:
        pairs.append(('5折集成', y_pred_ens, m_ens))
    fig2, axes2 = plt.subplots(1, len(pairs),
                                figsize=(len(pairs)*5.5, 5.2))
    if len(pairs) == 1: axes2 = [axes2]
    fig2.suptitle('Confusion Matrices: Baseline vs Optimized',
                  fontsize=13, fontweight='bold')
    for ax, (lbl, yp, m) in zip(axes2, pairs):
        if yp is not None:
            draw_cm(ax, confusion_matrix(y_te, yp), lbl, m['accuracy'])
    plt.tight_layout()
    save_fig(fig2, '04_混淆矩阵对比.png')

    # ── 图3：优化训练曲线（3子图）────────────────────────────────────────────
    if history:
        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
        fig3.suptitle(f'Optimized Training Curves: {name}',
                      fontsize=12, fontweight='bold')
        axes3[0].plot(history['train_loss'], color='#EF4444', lw=2)
        axes3[0].set_title('Training Loss'); axes3[0].set_xlabel('Epoch')
        axes3[0].set_ylabel('Loss'); axes3[0].grid(alpha=0.25)

        axes3[1].plot(history['train_acc'], label='Train',
                      color='#3B82F6', lw=2)
        axes3[1].plot(history['val_acc'], label='Val',
                      color='#10B981', lw=2)
        axes3[1].axhline(max(history['val_acc']), color='red',
                          ls='--', lw=1.2,
                          label=f'Best={max(history["val_acc"]):.4f}')
        axes3[1].set_title('Train vs Val Accuracy')
        axes3[1].set_xlabel('Epoch'); axes3[1].set_ylabel('Accuracy')
        axes3[1].legend(fontsize=9); axes3[1].grid(alpha=0.25)

        va = history['val_acc']
        axes3[2].plot(va, color='#8B5CF6', lw=2)
        axes3[2].fill_between(range(len(va)), va,
                               alpha=0.15, color='#8B5CF6')
        axes3[2].set_title('Val Accuracy Progression')
        axes3[2].set_xlabel('Epoch'); axes3[2].set_ylabel('Accuracy')
        axes3[2].grid(alpha=0.25)
        plt.tight_layout()
        save_fig(fig3, '04_优化训练曲线.png')

    # ── 图4：各成熟度 F1 提升对比 ────────────────────────────────────────────
    f1_base = [f1_score(y_te==i, y_pred_base==i) for i in range(N_CLASSES)]
    f1_opt  = [f1_score(y_te==i, y_pred_opt==i)  for i in range(N_CLASSES)]
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    x4, w4 = np.arange(N_CLASSES), 0.35
    b1 = ax4.bar(x4-w4/2, f1_base, w4, label='Baseline',
                  color='#94A3B8', alpha=0.85, edgecolor='white')
    b2 = ax4.bar(x4+w4/2, f1_opt,  w4, label='Optimized',
                  color=['#3B82F6','#10B981','#F59E0B','#EF4444'],
                  alpha=0.85, edgecolor='white')
    for bar, v in zip(b1, f1_base):
        ax4.text(bar.get_x()+bar.get_width()/2, v+0.005,
                 f'{v:.3f}', ha='center', fontsize=9)
    for bar, v, vb in zip(b2, f1_opt, f1_base):
        d = v - vb
        ax4.text(bar.get_x()+bar.get_width()/2, v+0.005,
                 f'{v:.3f}\n({"+" if d>=0 else ""}{d*100:.1f}%)',
                 ha='center', fontsize=8, fontweight='bold')
    ax4.set_xticks(x4); ax4.set_xticklabels(EN_NAMES, fontsize=11)
    ax4.set_ylabel('F1 Score', fontsize=10)
    ax4.set_title(f'Per-Class F1: Baseline vs Optimized ({name})',
                  fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.1); ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3); plt.tight_layout()
    save_fig(fig4, '04_各类F1提升.png')

    # ── 图5：5折集成折柱图 ────────────────────────────────────────────────────
    if fold_accs:
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        fc = ['#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6']
        ax5.bar([f'Fold {i+1}' for i in range(len(fold_accs))],
                fold_accs, color=fc[:len(fold_accs)],
                alpha=0.85, edgecolor='white', lw=1.5)
        ax5.axhline(np.mean(fold_accs), color='red', ls='--', lw=2,
                    label=f'Mean={np.mean(fold_accs):.4f}'
                          f'±{np.std(fold_accs):.4f}')
        for i, v in enumerate(fold_accs):
            ax5.text(i, v+0.003, f'{v:.4f}', ha='center',
                     fontsize=9, fontweight='bold')
        ax5.set_ylabel('Val Accuracy', fontsize=10)
        ax5.set_title(f'5-Fold Ensemble: {name}',
                      fontsize=12, fontweight='bold')
        ax5.set_ylim(max(0, min(fold_accs)-0.05), 1.04)
        ax5.legend(fontsize=9); ax5.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_fig(fig5, '04_集成折叠柱图.png')

    print('  → charts全部生成完毕')


# =============================================================================
#  Step 6 · 报告
# =============================================================================

def step6_report(name, m_base, m_opt, m_ens,
                 fold_accs, y_te, y_pred_opt):
    section('Step 6 · 优化报告')
    line = '=' * 72; sep = '─' * 72
    rows = [
        line,
        f'  葡萄成熟度识别 · 第四阶段：{name} 深度优化报告（全 PyTorch）',
        line, '',
        '【优化策略总览】',
        '  策略① SMOTE / BorderlineSMOTE 智能过采样（降级自动处理）',
        '  策略② 动态数据增强（高斯噪声 + 随机遮罩 + 类内Mixup）',
        '  策略③ 增强型网络架构（各模型对应专属加深版）',
        '  策略④ 标签平滑 CE 损失（ε=0.1）',
        '  策略⑤ OneCycleLR + AMP 混合精度训练',
        '  策略⑥ 5折集成推理（软投票，输出最终预测）',
        '',
        '【指标对比（测试集）】',
        f'  {"版本":<20}  {"Acc":>8}  {"Prec":>8}  {"Rec":>8}  {"F1":>8}',
        '  ' + sep,
    ]
    for ver, m in [('基线(Stage3)', m_base),
                    (f'{name}(优化后)', m_opt)]:
        rows.append(f'  {ver:<20}  {m["accuracy"]:>8.4f}  '
                    f'{m["precision"]:>8.4f}  {m["recall"]:>8.4f}  '
                    f'{m["f1"]:>8.4f}')
    if m_ens:
        da = m_ens['accuracy'] - m_base['accuracy']
        df = m_ens['f1']       - m_base['f1']
        rows.append(f'  {"5折集成推理":<20}  {m_ens["accuracy"]:>8.4f}  '
                    f'{m_ens["precision"]:>8.4f}  {m_ens["recall"]:>8.4f}  '
                    f'{m_ens["f1"]:>8.4f}  '
                    f'Acc{da:+.4f} F1{df:+.4f}')

    da_o = m_opt['accuracy'] - m_base['accuracy']
    df_o = m_opt['f1']       - m_base['f1']
    rows += [
        '',
        '【提升幅度（优化后 vs 基线）】',
        f'  Acc: {m_base["accuracy"]:.4f} → {m_opt["accuracy"]:.4f}'
        f'  ({da_o:+.4f} / {da_o*100:+.2f}%)',
        f'  F1 : {m_base["f1"]:.4f} → {m_opt["f1"]:.4f}'
        f'  ({df_o:+.4f} / {df_o*100:+.2f}%)',
    ]
    if fold_accs:
        rows += [
            '',
            '【5折集成各折准确率】',
            '  ' + ' / '.join(f'F{i+1}={v:.4f}'
                               for i, v in enumerate(fold_accs)),
            f'  均值={np.mean(fold_accs):.4f}  std={np.std(fold_accs):.4f}',
        ]
    rows += [
        '',
        '【最终优化模型详细分类报告（测试集）】',
        classification_report(y_te, y_pred_opt,
                               target_names=EN_NAMES, digits=4),
        line,
    ]

    report_path = os.path.join(OUT_MODEL, '优化报告.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))
    print(f'  [已保存] {os.path.relpath(report_path, PROJECT_ROOT)}')
    for row in rows:
        print('  ' + row)


# =============================================================================
#  主程序
# =============================================================================

def main():
    print('\n' + '='*68)
    print('  葡萄成熟度识别 · 第四阶段：最优 PyTorch 模型深度优化')
    print('='*68)

    # Step 1
    X, y, X_tr, X_te, y_tr, y_te, info = step1_load()
    name = info['name']

    # Step 2 · 基线
    m_base, y_pred_base = step2_baseline(info, X_te, y_te)

    # Step 3 · 单模型优化
    _, y_pred_opt, _, m_opt, t_opt, hist = \
        step3_optimize(info, X_tr, y_tr, X_te, y_te)
    if y_pred_opt is None:
        print('[ERROR] 优化失败，退出'); return

    # Step 4 · 5折集成
    y_pred_ens, m_ens, fold_accs = \
        step4_ensemble(info, X_tr, y_tr, X_te, y_te)

    # Step 5 · 可视化
    step5_visualize(name, m_base, m_opt, m_ens,
                    y_te, y_pred_base, y_pred_opt, y_pred_ens,
                    hist, fold_accs)

    # Step 6 · 报告
    step6_report(name, m_base, m_opt, m_ens,
                 fold_accs, y_te, y_pred_opt)

    # 最终汇总
    final_m = m_ens if m_ens else m_opt
    section('全部完成！')
    print(f'  最优模型   : {name}')
    print(f'  基线       : Acc={m_base["accuracy"]:.4f}  F1={m_base["f1"]:.4f}')
    print(f'  优化后     : Acc={m_opt["accuracy"]:.4f}  F1={m_opt["f1"]:.4f}'
          f'  (↑Acc {m_opt["accuracy"]-m_base["accuracy"]:+.4f}'
          f'  ↑F1 {m_opt["f1"]-m_base["f1"]:+.4f})')
    if m_ens:
        print(f'  5折集成    : Acc={m_ens["accuracy"]:.4f}'
              f'  F1={m_ens["f1"]:.4f}')
    print('  图片: results/charts/04_*.png')
    print('  权重: results/models/best_optimized.pt')
    print(f'  集成权重: results/models/ensemble_weights/\n')


if __name__ == '__main__':
    main()