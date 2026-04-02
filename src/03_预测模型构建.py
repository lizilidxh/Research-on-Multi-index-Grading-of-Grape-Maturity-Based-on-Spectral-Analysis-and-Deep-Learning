# =============================================================================
#  葡萄成熟度识别 —— 第三阶段：六种 PyTorch 深度学习模型构建与对比
#
#  模型清单（全部 PyTorch 实现，共6种）：
#    Model 1 — MLP          多层感知机（GELU激活 + 残差连接 + BN）
#    Model 2 — CNN1D        一维卷积网络（多尺度卷积 + SE通道注意力）
#    Model 3 — TCN          时序卷积网络（膨胀因果卷积，感受野覆盖全谱）
#    Model 4 — ResNet1D     一维残差网络（跳跃连接 + SE Block）
#    Model 5 — MobileNet1D  深度可分离卷积轻量网络（InvertedResidual）
#    Model 6 — CNN_LSTM     卷积 + 双向LSTM 混合（局部特征+序列建模）
#
#  训练策略（全部模型共用）：
#    - 手动随机过采样平衡类别
#    - 80/20 分层划分
#    - AdamW 优化器 + L2正则(1e-4)
#    - CosineWarmup 学习率调度（Warmup 5epoch + 余弦退火）
#    - 梯度裁剪（max_norm=1.0）
#    - Early Stopping（patience=20）
#    - Batch=64, 最大Epoch=100
#
#  评估体系：
#    - Accuracy / Precision / Recall / F1（加权）
#    - 混淆矩阵（7个模型全部展示）
#    - 5折分层交叉验证（全部模型）
#    - 训练曲线（Loss & Val Accuracy）
#    - 综合排行榜气泡图
#
#  输出：
#    results/models/torch_mlp.pt
#    results/models/torch_cnn1d.pt
#    results/models/torch_tcn.pt
#    results/models/torch_resnet1d.pt
#    results/models/torch_mobilenet1d.pt
#    results/models/torch_cnn_lstm.pt
#    results/models/best_model_info.json   ← 供第四阶段读取
#    results/charts/03_*.png
#    results/models/模型对比报告.txt
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
    # _dynamo 的导入链最终执行 torch/onnx/__init__.py 中：
    #   from ._internal.exporter import (DiagnosticOptions, ExportOptions, ...)
    # 该模块在此版本安装中存在多个名称缺失，导致 ImportError。
    # 解法：先 import torch 让 torch.onnx._internal.exporter 注册到
    # sys.modules，然后把整个模块替换成含所有可能被引用名称的空占位符，
    # 再 import torch.optim，_dynamo 触发时就能正常 import。
    try:
        import sys, types as _types, importlib as _imp

        # 先确保 torch.onnx 真实包已加载（不触发 _dynamo）
        import torch.onnx as _onnx_pkg

        # 构造一个含全部已知缺失名称的占位模块
        _stub = _types.ModuleType("torch.onnx._internal.exporter")
        for _name in [
            "DiagnosticOptions", "ExportOptions", "ResolvedExportOptions",
            "OnnxExporterError", "Exporter", "ExportOutput",
            "export", "common_passes", "io_adapter",
        ]:
            setattr(_stub, _name, type(_name, (), {}))

        # 覆盖注入（无论之前是否存在）
        sys.modules["torch.onnx._internal.exporter"] = _stub
        # 同时挂到父包属性上，避免 getattr 路径失效
        if hasattr(sys.modules.get("torch.onnx._internal", None), "__dict__"):
            sys.modules["torch.onnx._internal"].exporter = _stub  # type: ignore

    except Exception:
        pass  # 修复失败时静默继续，让原始 ImportError 在后面暴露

    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] PyTorch {torch.__version__}  device={DEVICE}')

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
os.makedirs(OUT_MODEL, exist_ok=True)
os.makedirs(OUT_VIS,   exist_ok=True)

# ── 常量 ──────────────────────────────────────────────────────────────────────
MATURITY_NAMES = ['未成熟', '半成熟', '成熟', '过成熟']
EN_NAMES       = ['Unripe', 'Half-ripe', 'Ripe', 'Over-ripe']
PALETTE        = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
RANDOM_STATE   = 42
N_CLASSES      = 4
N_FOLDS        = 5

# 统一训练超参
EPOCHS   = 100
BATCH    = 64
LR       = 1e-3
PATIENCE = 20
WD       = 1e-4   # weight decay

# 7模型颜色
MODEL_COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444',
                '#8B5CF6', '#06B6D4', '#F97316']


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
        print(f'  {label:<22}  Acc={m["accuracy"]:.4f}  '
              f'Prec={m["precision"]:.4f}  '
              f'Rec={m["recall"]:.4f}  F1={m["f1"]:.4f}')
    return m


def manual_oversample(X, y, seed=RANDOM_STATE):
    rng     = np.random.RandomState(seed)
    max_cnt = max(Counter(y.tolist()).values())
    Xs, ys  = [X], [y]
    for cls in np.unique(y):
        mask = y == cls
        need = max_cnt - mask.sum()
        if need > 0:
            idx   = np.where(mask)[0]
            extra = rng.choice(idx, size=need, replace=True)
            Xs.append(X[extra]); ys.append(y[extra])
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
            c = 'white' if cm_n[i, j] > 0.55 else 'black'
            ax.text(j+0.5, i+0.5,
                    f'{cm[i,j]}\n({cm_n[i,j]*100:.0f}%)',
                    ha='center', va='center', fontsize=7,
                    color=c, fontweight='bold')
    ax.set_title(f'{title}\nAcc={acc:.4f}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.tick_params(labelsize=7)


# =============================================================================
#  公共模块（各模型共享）
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels, ratio=8):
        super().__init__()
        mid = max(1, channels // ratio)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(),
            nn.Linear(mid, channels), nn.Sigmoid()
        )
    def forward(self, x):          # x: (B, C, L)
        w = self.fc(self.pool(x))  # (B, C)
        return x * w.unsqueeze(-1)


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """线性 Warmup + 余弦退火"""
    def __init__(self, optimizer, warmup_epochs, max_epochs):
        self.warmup  = warmup_epochs
        self.max_ep  = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        ep = self.last_epoch
        if ep < self.warmup:
            factor = (ep + 1) / max(self.warmup, 1)
        else:
            prog   = (ep - self.warmup) / max(self.max_ep - self.warmup, 1)
            factor = 0.5 * (1 + np.cos(np.pi * prog))
        return [base * factor for base in self.base_lrs]


# =============================================================================
#  Model 1 · MLP
# =============================================================================

class MLP(nn.Module):
    """
    多层感知机
    - 4个全连接块，每块：Linear → BN → GELU → Dropout
    - 块间残差连接（维度相同时直接相加）
    """
    def __init__(self, in_dim, n_cls=N_CLASSES):
        super().__init__()
        dims = [in_dim, 512, 256, 128, 64]
        self.blocks = nn.ModuleList()
        self.shorts  = nn.ModuleList()
        for i in range(len(dims) - 1):
            blk = nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.GELU(),
                nn.Dropout(0.3 if i < 2 else 0.2)
            )
            self.blocks.append(blk)
            # 如果维度不同，用 1x1 线性投影对齐残差
            if dims[i] != dims[i+1]:
                self.shorts.append(nn.Linear(dims[i], dims[i+1], bias=False))
            else:
                self.shorts.append(nn.Identity())
        self.head = nn.Linear(64, n_cls)

    def forward(self, x):
        for blk, sc in zip(self.blocks, self.shorts):
            x = blk(x) + sc(x)
        return self.head(x)


# =============================================================================
#  Model 2 · CNN1D
# =============================================================================

class CNN1D(nn.Module):
    """
    一维卷积网络
    - 三级卷积组（32→64→128→256），每级后接 SE 通道注意力
    - 多尺度卷积核（7、5、3），逐级下采样
    - Global Average Pooling + FC 分类头
    """
    def __init__(self, in_dim, n_cls=N_CLASSES):
        super().__init__()

        def conv_bn_act(in_ch, out_ch, k, p, stride=1):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=p, stride=stride),
                nn.BatchNorm1d(out_ch), nn.GELU())

        self.stem = conv_bn_act(1, 32, 7, 3)          # (B,32,L)
        self.layer1 = nn.Sequential(
            conv_bn_act(32, 64, 5, 2),
            conv_bn_act(64, 64, 3, 1),
            SEBlock(64), nn.MaxPool1d(2))              # (B,64,L/2)
        self.layer2 = nn.Sequential(
            conv_bn_act(64, 128, 3, 1),
            conv_bn_act(128, 128, 3, 1),
            SEBlock(128), nn.MaxPool1d(2))             # (B,128,L/4)
        self.layer3 = nn.Sequential(
            conv_bn_act(128, 256, 3, 1),
            conv_bn_act(256, 256, 3, 1),
            SEBlock(256),
            nn.AdaptiveAvgPool1d(1))                   # (B,256,1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls))

    def forward(self, x):
        x = self.stem(x.unsqueeze(1))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


# =============================================================================
#  Model 3 · TCN（时序卷积网络）
# =============================================================================

class TCNBlock(nn.Module):
    """
    单个 TCN 残差块
    - 膨胀因果卷积（dilation 指数增长），覆盖超长感受野
    - WeightNorm 替代 BatchNorm（序列模型更稳定）
    - 残差投影对齐通道数
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation          # causal padding
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=pad, dilation=dilation))
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size,
                      padding=pad, dilation=dilation))
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()
        # chomp: 去掉右侧多余 padding，保证因果性
        self.chomp = pad
        self.sc    = (nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, 1))
                      if in_ch != out_ch else nn.Identity())
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)

    def _chomp(self, x):
        return x[:, :, :-self.chomp].contiguous() if self.chomp else x

    def forward(self, x):
        out = self.act(self._chomp(self.conv1(x)))
        out = self.drop(out)
        out = self.act(self._chomp(self.conv2(out)))
        out = self.drop(out)
        return self.act(out + self.sc(x))


class TCN(nn.Module):
    """
    时序卷积网络 (Temporal Convolutional Network)
    - 8 层 TCN Block，dilation = 1,2,4,8,16,32,64,128
    - 感受野 = 2*(1+2+4+...+128)*(kernel_size-1)+1 = 512*2+1，完全覆盖728波段
    - Global Average Pooling + FC 分类头
    - 参数量适中，训练速度快，长距离依赖捕获能力强
    """
    def __init__(self, in_dim, n_cls=N_CLASSES,
                 n_channels=64, kernel_size=3,
                 n_levels=8, dropout=0.2):
        super().__init__()
        layers = []
        # 第一层：输入通道 1 → n_channels
        layers.append(TCNBlock(1, n_channels, kernel_size,
                                dilation=1, dropout=dropout))
        # 后续层：dilation 指数增长
        for i in range(1, n_levels):
            layers.append(TCNBlock(n_channels, n_channels, kernel_size,
                                    dilation=2**i, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, n_cls))

    def forward(self, x):
        # x: (B, L) → (B, 1, L)
        out = self.network(x.unsqueeze(1))   # (B, C, L)
        return self.head(self.pool(out))

#
# #  Model 4 · Transformer
# # =============================================================================
#
# class SpecTransformer(nn.Module):
#     """
#     光谱 Transformer
#     - 每个波段投影为 d_model 维 token
#     - 可学习 CLS Token + 可学习位置编码
#     - Pre-LN Transformer Encoder × 4层
#     - 用 CLS Token 输出做分类
#     """
#     def __init__(self, in_dim, n_cls=N_CLASSES,
#                  d_model=128, nhead=8, num_layers=4):
#         super().__init__()
#         self.proj     = nn.Linear(1, d_model)
#         self.cls_tok  = nn.Parameter(torch.zeros(1, 1, d_model))
#         self.pos_emb  = nn.Parameter(torch.zeros(1, in_dim + 1, d_model))
#         enc = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead,
#             dim_feedforward=d_model * 4,
#             dropout=0.1, batch_first=True,
#             activation='gelu', norm_first=True)   # Pre-LN
#         self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
#         self.norm    = nn.LayerNorm(d_model)
#         self.head    = nn.Sequential(
#             nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.2),
#             nn.Linear(64, n_cls))
#         nn.init.trunc_normal_(self.cls_tok, std=0.02)
#         nn.init.trunc_normal_(self.pos_emb, std=0.02)
#
#     def forward(self, x):
#         B, L = x.shape
#         tok = self.proj(x.unsqueeze(2))                # (B, L, d)
#         cls = self.cls_tok.expand(B, -1, -1)
#         tok = torch.cat([cls, tok], dim=1)             # (B, L+1, d)
#         tok = tok + self.pos_emb[:, :L + 1, :]
#         tok = self.norm(self.encoder(tok))
#         return self.head(tok[:, 0, :])                 # CLS token
#

# =============================================================================
#  Model 5 · ResNet1D
# =============================================================================

class ResBlock1D(nn.Module):
    """一维残差块（含可选下采样 + SE）"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch))
        self.se  = SEBlock(out_ch)
        self.sc  = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch)
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.se(self.conv(x)) + self.sc(x))


class ResNet1D(nn.Module):
    """
    一维 ResNet
    - Stem: Conv7 + BN + GELU
    - 4个 Stage（通道 32→64→128→256），每 Stage 含 2个残差块
    - Stage 2/3/4 第一块 stride=2 下采样
    - Global Average Pooling + FC
    """
    def __init__(self, in_dim, n_cls=N_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU())
        self.layer1 = nn.Sequential(ResBlock1D(32, 32), ResBlock1D(32, 32))
        self.layer2 = nn.Sequential(ResBlock1D(32, 64, stride=2),
                                     ResBlock1D(64, 64))
        self.layer3 = nn.Sequential(ResBlock1D(64, 128, stride=2),
                                     ResBlock1D(128, 128))
        self.layer4 = nn.Sequential(ResBlock1D(128, 256, stride=2),
                                     ResBlock1D(256, 256))
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls))

    def forward(self, x):
        x = self.stem(x.unsqueeze(1))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.pool(x))


# =============================================================================
#  Model 6 · MobileNet1D
# =============================================================================

class InvertedResidual1D(nn.Module):
    """MobileNetV2 风格的倒残差块（1D版）"""
    def __init__(self, in_ch, out_ch, stride=1, expand=6):
        super().__init__()
        mid_ch = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.conv = nn.Sequential(
            # Pointwise expand
            nn.Conv1d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm1d(mid_ch), nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv1d(mid_ch, mid_ch, 3, stride=stride,
                      padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm1d(mid_ch), nn.ReLU6(inplace=True),
            # Pointwise project
            nn.Conv1d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch))

    def forward(self, x):
        out = self.conv(x)
        return out + x if self.use_res else out


class MobileNet1D(nn.Module):
    """
    一维 MobileNetV2
    - 倒残差块 × 7组，通道逐步扩展（32→16→24→32→64→96→160→320）
    - ReLU6 激活，计算高效，参数量最小
    - 适合资源受限场景的轻量模型
    """
    def __init__(self, in_dim, n_cls=N_CLASSES):
        super().__init__()
        # (in_ch, out_ch, stride, expand_ratio, n_blocks)
        cfg = [
            (32,  16,  1, 1, 1),
            (16,  24,  2, 6, 2),
            (24,  32,  2, 6, 3),
            (32,  64,  2, 6, 4),
            (64,  96,  1, 6, 3),
            (96,  160, 2, 6, 3),
            (160, 320, 1, 6, 1),
        ]
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU6(inplace=True))
        layers = []
        for in_ch, out_ch, stride, expand, n in cfg:
            for k in range(n):
                layers.append(
                    InvertedResidual1D(in_ch if k == 0 else out_ch,
                                       out_ch,
                                       stride if k == 0 else 1,
                                       expand))
        self.features = nn.Sequential(*layers)
        self.conv_last = nn.Sequential(
            nn.Conv1d(320, 1280, 1, bias=False),
            nn.BatchNorm1d(1280), nn.ReLU6(inplace=True))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, n_cls))

    def forward(self, x):
        x = self.stem(x.unsqueeze(1))
        x = self.features(x)
        x = self.conv_last(x)
        return self.head(self.pool(x))


# =============================================================================
#  Model 7 · CNN_LSTM
# =============================================================================

class CNN_LSTM(nn.Module):
    """
    CNN-BiLSTM 混合模型
    - 前段 CNN 提取局部光谱特征（多尺度 + SE）
    - 后段 BiLSTM 捕获特征图中的序列依赖
    - Multi-head Attention 聚合 LSTM 输出
    """
    def __init__(self, in_dim, n_cls=N_CLASSES, hidden=128):
        super().__init__()
        # ── CNN 特征提取 ──
        self.cnn = nn.Sequential(
            # 第一卷积组
            nn.Conv1d(1, 64, 7, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            SEBlock(64), nn.MaxPool1d(2),
            # 第二卷积组
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            SEBlock(128), nn.MaxPool1d(2))
        # ── 双向 LSTM ──
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=hidden,
            num_layers=2, batch_first=True,
            dropout=0.2, bidirectional=True)
        # ── Multi-head Attention ──
        self.mha  = nn.MultiheadAttention(
            embed_dim=hidden * 2, num_heads=8,
            dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls))

    def forward(self, x):
        # CNN
        feat = self.cnn(x.unsqueeze(1))        # (B, 128, L')
        feat = feat.permute(0, 2, 1)           # (B, L', 128)
        # BiLSTM
        out, _ = self.lstm(feat)               # (B, L', 2H)
        # MHA
        attn, _ = self.mha(out, out, out)
        ctx     = self.norm(out + attn).mean(dim=1)  # 全局平均
        return self.head(ctx)


# =============================================================================
#  统一训练 / 推理函数
# =============================================================================

def train_model(model, X_tr, y_tr, X_val, y_val,
                epochs=EPOCHS, batch=BATCH, lr=LR,
                patience=PATIENCE, wd=WD, name='model'):
    """
    标准训练流程：
    AdamW + CosineWarmup + GradClip + EarlyStopping
    返回 (trained_model, history_dict)
    """
    model   = model.to(DEVICE)
    opt     = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched   = CosineWarmupScheduler(opt, warmup_epochs=5, max_epochs=epochs)
    loss_fn = nn.CrossEntropyLoss()

    Xval_t  = torch.FloatTensor(X_val).to(DEVICE)
    loader  = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr).to(DEVICE),
                      torch.LongTensor(y_tr).to(DEVICE)),
        batch_size=batch, shuffle=True, drop_last=False)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc, best_state, no_imp = 0., None, 0

    for ep in range(epochs):
        model.train()
        tot_loss, n_correct, n_total = 0., 0, 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss  += loss.item() * len(yb)
            n_correct += (logits.argmax(1) == yb).sum().item()
            n_total   += len(yb)
        sched.step()

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
                print(f'    [{name}] EarlyStop @ ep{ep+1:3d}'
                      f'  best_val={best_acc:.4f}')
                break

    model.load_state_dict(best_state)
    return model, history


def predict(model, X):
    """推理，返回 (y_pred, y_prob)"""
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(
            model(torch.FloatTensor(X).to(DEVICE)), 1
        ).cpu().numpy()
    return probs.argmax(1), probs


def cross_validate(ModelClass, X, y, n_folds=N_FOLDS,
                   epochs=60, patience=PATIENCE):
    """n折交叉验证，返回各折 accuracy 数组"""
    skf    = StratifiedKFold(n_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    in_dim = X.shape[1]
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr, yval = y[tr_idx], y[val_idx]
        Xtr_b, ytr_b = manual_oversample(Xtr, ytr)
        m, _ = train_model(
            ModelClass(in_dim), Xtr_b, ytr_b, Xval, yval,
            epochs=epochs, patience=patience, name=f'CV{fold}')
        preds, _ = predict(m, Xval)
        scores.append(accuracy_score(yval, preds))
        print(f'    fold {fold}: {scores[-1]:.4f}')
    return np.array(scores)


# =============================================================================
#  Step 1 · 数据加载
# =============================================================================

def step1_load():
    section('Step 1 · 加载数据 & 划分')

    # 优先使用 SG+SNV 预处理后的原始光谱（728维），深度学习效果更佳；
    # 若第二阶段尚未运行，自动退回 PCA 降维版本（7维）
    pre_path = os.path.join(OUT_MODEL, 'X_preprocessed.npy')
    pca_path = os.path.join(OUT_MODEL, 'X_pca.npy')

    if os.path.exists(pre_path):
        X = np.load(pre_path)
        print('  [数据源] X_preprocessed.npy（SG+SNV 预处理光谱，推荐）')
    elif os.path.exists(pca_path):
        X = np.load(pca_path)
        print('  [数据源] X_pca.npy（PCA降维，已自动退回）')
        print('  [提示]   建议先运行 02_光谱预处理.py 以获得更好效果')
    else:
        raise FileNotFoundError(
            '未找到特征文件。\n'
            '请先运行 02_光谱预处理.py 生成 X_preprocessed.npy / X_pca.npy')

    y = np.load(os.path.join(OUT_MODEL, 'y_labels.npy'))

    print(f'  特征矩阵: {X.shape}  标签: {y.shape}')
    dist = {MATURITY_NAMES[i]: int((y==i).sum()) for i in range(N_CLASSES)}
    print(f'  标签分布: {dist}')

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_tr_b, y_tr_b = manual_oversample(X_tr, y_tr)

    print(f'  训练: {len(X_tr)}  测试: {len(X_te)}'
          f'  过采样后训练: {len(X_tr_b)}')
    return X, y, X_tr, X_te, y_tr, y_te, X_tr_b, y_tr_b


# =============================================================================
#  Step 2-8 · 七模型逐一训练
# =============================================================================

MODEL_REGISTRY = [
    ('MLP',          MLP,             'torch_mlp.pt'),
    ('CNN1D',        CNN1D,           'torch_cnn1d.pt'),
    ('TCN',          TCN,             'torch_tcn.pt'),
    ('ResNet1D',     ResNet1D,        'torch_resnet1d.pt'),
    ('MobileNet1D',  MobileNet1D,     'torch_mobilenet1d.pt'),
    ('CNN-LSTM',     CNN_LSTM,        'torch_cnn_lstm.pt'),
]


def train_all_models(X_tr_b, y_tr_b, X_te, y_te):
    results = []
    in_dim  = X_tr_b.shape[1]

    for step_idx, (name, ModelClass, save_file) in enumerate(MODEL_REGISTRY, 2):
        section(f'Step {step_idx} · Model {step_idx-1}: {name}')

        model   = ModelClass(in_dim)
        n_param = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)
        print(f'  参数量: {n_param:,}')

        t0 = time.time()
        model, history = train_model(
            model, X_tr_b, y_tr_b, X_te, y_te,
            epochs=EPOCHS, batch=BATCH, lr=LR,
            patience=PATIENCE, name=name)
        t_train = time.time() - t0

        y_pred, y_prob = predict(model, X_te)
        m = calc_metrics(y_te, y_pred, label=name)
        print(f'  训练耗时: {t_train:.1f}s')

        torch.save(model.state_dict(),
                   os.path.join(OUT_MODEL, save_file))

        results.append(dict(
            name=name, ModelClass=ModelClass,
            save_file=save_file,
            model=model,
            y_pred=y_pred, y_prob=y_prob,
            y_te=y_te,
            metrics=m, t_train=t_train,
            history=history,
            n_params=n_param,
            in_dim=in_dim,
        ))

    return results


# =============================================================================
#  Step 9 · 5折交叉验证
# =============================================================================

def step9_cross_validation(X, y, results):
    section('Step 9 · 5折交叉验证（全部7个模型）')
    cv_results = {}
    for r in results:
        name = r['name']
        print(f'\n  ── {name} ──')
        scores = cross_validate(
            r['ModelClass'], X, y,
            n_folds=N_FOLDS,
            epochs=min(EPOCHS, 60),
            patience=PATIENCE)
        cv_results[name] = dict(scores=scores,
                                 mean=scores.mean(),
                                 std=scores.std())
        print(f'  {name}  均值={scores.mean():.4f}  '
              f'std={scores.std():.4f}')
    return cv_results


# =============================================================================
#  Step 10 · 可视化
# =============================================================================

def step10_visualize(results, cv_results):
    section('Step 10 · 可视化')

    names  = [r['name']                 for r in results]
    accs   = [r['metrics']['accuracy']  for r in results]
    f1s    = [r['metrics']['f1']        for r in results]
    precs  = [r['metrics']['precision'] for r in results]
    recs   = [r['metrics']['recall']    for r in results]
    times  = [r['t_train']              for r in results]
    params = [r['n_params']             for r in results]
    bc     = MODEL_COLORS[:len(results)]
    x      = np.arange(len(results))

    # ── 图1：四指标 2×2 柱状图 ────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))
    fig1.suptitle('PyTorch Deep Learning Models — Performance Comparison',
                  fontsize=14, fontweight='bold')
    for ax, vals, title in [
        (axes1[0,0], accs,  'Accuracy'),
        (axes1[0,1], f1s,   'F1 Score (Weighted)'),
        (axes1[1,0], precs, 'Precision (Weighted)'),
        (axes1[1,1], recs,  'Recall (Weighted)'),
    ]:
        bars = ax.bar(x, vals, color=bc, alpha=0.85,
                      edgecolor='white', lw=1.5)
        best = int(np.argmax(vals))
        bars[best].set_edgecolor('gold')
        bars[best].set_linewidth(3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f'{v:.4f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', rotation=45)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(max(0, min(vals)-0.06), 1.07)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig1, '03_七模型指标对比.png')

    # ── 图2：训练时间 & 参数量双轴图 ──────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2r = ax2.twinx()
    bars = ax2.barh(names, times, color=bc, alpha=0.75,
                    edgecolor='white', lw=1.5, label='Time (s)')
    ax2r.plot(range(len(names)), [p/1e6 for p in params],
              'D--', color='#1e293b', lw=2, ms=7, label='Params (M)')
    for bar, t in zip(bars, times):
        ax2.text(bar.get_width() + max(times)*0.01,
                 bar.get_y() + bar.get_height()/2,
                 f'{t:.1f}s', va='center', fontsize=9, fontweight='bold')
    ax2.set_xlabel('Training Time (s)', fontsize=10)
    ax2r.set_ylabel('Parameters (M)', fontsize=10)
    ax2.set_title('Training Time & Model Size', fontsize=12, fontweight='bold')
    lines1, lb1 = ax2.get_legend_handles_labels()
    lines2, lb2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1+lines2, lb1+lb2, fontsize=9, loc='lower right')
    ax2.set_xlim(0, max(times)*1.25)
    ax2.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_fig(fig2, '03_时间与参数量.png')

    # ── 图3：七模型混淆矩阵（2行4列，最后一格留空）────────────────────────────
    n = len(results)
    nrows, ncols = 2, 4
    fig3, axes3 = plt.subplots(nrows, ncols,
                                figsize=(ncols*4.5, nrows*4.8))
    fig3.suptitle('Confusion Matrices — All 7 PyTorch Models',
                  fontsize=13, fontweight='bold')
    for idx, r in enumerate(results):
        ax = axes3[idx // ncols, idx % ncols]
        cm = confusion_matrix(r['y_te'], r['y_pred'])
        draw_cm(ax, cm, r['name'], r['metrics']['accuracy'])
    # 隐藏多余的子图
    for idx in range(n, nrows * ncols):
        axes3[idx // ncols, idx % ncols].set_visible(False)
    plt.tight_layout()
    save_fig(fig3, '03_全部混淆矩阵.png')

    # ── 图4：训练曲线（Loss + Val Acc，2行7列）────────────────────────────────
    fig4, axes4 = plt.subplots(2, n, figsize=(n*3.5, 7))
    fig4.suptitle('Training Curves — Loss & Validation Accuracy',
                  fontsize=13, fontweight='bold')
    for col, r in enumerate(results):
        hist  = r['history']
        color = bc[col]
        # Loss
        axes4[0, col].plot(hist['train_loss'], color=color, lw=1.8)
        axes4[0, col].set_title(r['name'], fontsize=9, fontweight='bold')
        axes4[0, col].set_xlabel('Epoch', fontsize=7)
        axes4[0, col].set_ylabel('Loss', fontsize=7)
        axes4[0, col].grid(alpha=0.25)
        # Val Acc
        va   = hist['val_acc']
        axes4[1, col].plot(va, color=color, lw=1.8)
        axes4[1, col].axhline(max(va), color='red', ls='--', lw=1.2,
                               label=f'Best={max(va):.4f}')
        axes4[1, col].set_xlabel('Epoch', fontsize=7)
        axes4[1, col].set_ylabel('Val Acc', fontsize=7)
        axes4[1, col].legend(fontsize=6)
        axes4[1, col].grid(alpha=0.25)
    plt.tight_layout()
    save_fig(fig4, '03_训练曲线.png')

    # ── 图5：5折CV箱线图 ──────────────────────────────────────────────────────
    if cv_results:
        cv_names  = list(cv_results.keys())
        cv_data   = [cv_results[n]['scores'] for n in cv_names]
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        bp = ax5.boxplot(cv_data, patch_artist=True,
                          medianprops=dict(color='white', lw=2.5),
                          whiskerprops=dict(lw=1.8),
                          capprops=dict(lw=1.8),
                          flierprops=dict(marker='o', ms=5,
                                          markerfacecolor='red', alpha=0.5))
        cv_colors = [bc[names.index(n)] for n in cv_names]
        for patch, c in zip(bp['boxes'], cv_colors):
            patch.set_facecolor(c); patch.set_alpha(0.8)
        for i, n in enumerate(cv_names, 1):
            m, s = cv_results[n]['mean'], cv_results[n]['std']
            ax5.text(i, m + 0.005, f'{m:.4f}±{s:.4f}',
                     ha='center', va='bottom', fontsize=9,
                     fontweight='bold', color='#1e293b')
        ax5.set_xticklabels(cv_names, fontsize=10)
        ax5.set_ylabel('Accuracy', fontsize=10)
        ax5.set_title(f'{N_FOLDS}-Fold CV Accuracy — All 7 Models',
                      fontsize=12, fontweight='bold')
        ax5.set_ylim(max(0,
                         min(s.min() for s in cv_data) - 0.05), 1.04)
        ax5.grid(axis='y', alpha=0.3)
        ax5.set_facecolor('#f8fafc')
        plt.tight_layout()
        save_fig(fig5, '03_交叉验证箱线图.png')

    # ── 图6：综合气泡图（Acc vs F1，气泡大小 ∝ 速度）────────────────────────
    fig6, ax6 = plt.subplots(figsize=(11, 7))
    for i, r in enumerate(results):
        sz = max(150, 8000 / (r['t_train'] + 2))
        ax6.scatter(r['metrics']['accuracy'], r['metrics']['f1'],
                    s=sz, c=bc[i], alpha=0.85,
                    edgecolor='white', lw=2, zorder=5)
        ax6.annotate(r['name'],
                     (r['metrics']['accuracy'], r['metrics']['f1']),
                     textcoords='offset points', xytext=(8, 4),
                     fontsize=9.5, fontweight='bold')
    best_i = int(np.argmax(f1s))
    ax6.scatter(accs[best_i], f1s[best_i], s=500, c='none',
                edgecolor='gold', lw=3, zorder=10, label='★ Best Model')
    ax6.set_xlabel('Test Accuracy', fontsize=11)
    ax6.set_ylabel('Test F1 Score (Weighted)', fontsize=11)
    ax6.set_title('Model Landscape: Accuracy vs F1\n'
                  '(Bubble size ∝ 1/Training Time)',
                  fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.25)
    plt.tight_layout()
    save_fig(fig6, '03_模型气泡图.png')

    print('  → 6张图全部生成完毕')


# =============================================================================
#  Step 11 · 报告 & 保存最优模型信息
# =============================================================================

def step11_report(results, cv_results):
    section('Step 11 · 报告 & 最优模型信息')

    ranked = sorted(results, key=lambda r: r['metrics']['f1'], reverse=True)

    line = '=' * 72
    sep  = '─' * 72
    rows = [
        line,
        '  葡萄成熟度识别 · 第三阶段：七种 PyTorch 深度学习模型对比报告',
        line, '',
        '【模型清单】',
        '  1-MLP  2-CNN1D  3-TCN ,'
        '  4-ResNet1D  5-MobileNet1D  6-CNN-LSTM',
        '',
        '【测试集性能排行（按 F1 降序）】',
        f'  {"排名":<4}  {"模型":<14}  {"Acc":>8}  {"Prec":>8}'
        f'  {"Rec":>8}  {"F1":>8}  {"Params":>10}  {"Time(s)":>8}',
        '  ' + sep,
    ]
    for rank, r in enumerate(ranked, 1):
        m    = r['metrics']
        star = '  ★' if rank == 1 else ''
        rows.append(
            f'  {rank:<4}  {r["name"]:<14}  {m["accuracy"]:>8.4f}  '
            f'{m["precision"]:>8.4f}  {m["recall"]:>8.4f}  '
            f'{m["f1"]:>8.4f}  {r["n_params"]:>10,}  '
            f'{r["t_train"]:>8.1f}{star}')

    if cv_results:
        rows += ['', '【5折交叉验证结果（mean ± std）】',
                 f'  {"模型":<14}  {"mean":>8}  {"std":>8}  各折',
                 '  ' + sep]
        for name, res in cv_results.items():
            sc = '  '.join(f'{v:.4f}' for v in res['scores'])
            rows.append(f'  {name:<14}  {res["mean"]:>8.4f}'
                        f'  {res["std"]:>8.4f}  {sc}')

    best = ranked[0]
    rows += [
        '',
        f'【★ 最优模型：{best["name"]}】',
        f'  Acc={best["metrics"]["accuracy"]:.4f}  '
        f'F1={best["metrics"]["f1"]:.4f}  '
        f'参数量={best["n_params"]:,}',
        '',
        '【最优模型详细分类报告（测试集）】',
        classification_report(best['y_te'], best['y_pred'],
                               target_names=EN_NAMES, digits=4),
        line,
    ]

    report_path = os.path.join(OUT_MODEL, '模型对比报告.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))
    print(f'  [已保存] {os.path.relpath(report_path, PROJECT_ROOT)}')
    for row in rows:
        print('  ' + row)

    # ── 保存 best_model_info.json ──────────────────────────────────────────
    best_info = {
        'name':       best['name'],
        'save_file':  best['save_file'],
        'accuracy':   float(best['metrics']['accuracy']),
        'f1':         float(best['metrics']['f1']),
        'in_dim':     int(best['in_dim']),
        'n_params':   int(best['n_params']),
    }
    info_path = os.path.join(OUT_MODEL, 'best_model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)
    print(f'\n  最优模型信息已写入: {info_path}')
    print(f'  {best_info}')
    return best_info


# =============================================================================
#  主程序
# =============================================================================

def main():
    print('\n' + '='*68)
    print('  葡萄成熟度识别 · 第三阶段：七种 PyTorch 深度学习模型')
    print('='*68)

    # Step 1
    X, y, X_tr, X_te, y_tr, y_te, X_tr_b, y_tr_b = step1_load()

    # Step 2-8：七模型训练
    results = train_all_models(X_tr_b, y_tr_b, X_te, y_te)

    # Step 9：5折CV
    cv_results = step9_cross_validation(X, y, results)

    # Step 10：可视化
    step10_visualize(results, cv_results)

    # Step 11：报告 & 最优信息
    best_info = step11_report(results, cv_results)

    section('全部完成！下一步：运行 04_模型优化.py')
    print(f'  ★ 最优模型: {best_info["name"]}'
          f'  Acc={best_info["accuracy"]:.4f}'
          f'  F1={best_info["f1"]:.4f}')
    print('  图片: results/charts/03_*.png\n')


if __name__ == '__main__':
    main()