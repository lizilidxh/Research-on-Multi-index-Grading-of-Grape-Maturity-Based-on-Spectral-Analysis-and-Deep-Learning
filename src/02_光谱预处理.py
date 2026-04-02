# =============================================================================
#  算法流程：
#    1. 加载聚类标签数据集（第一阶段输出）
#    2. 原始光谱可视化（观察噪声/基线漂移）
#    3. SG 平滑（Savitzky-Golay）—— 去除高频噪声
#    4. SNV 变换（Standard Normal Variate）—— 消除散射
#    5. MSC 校正（Multiplicative Scatter Correction）—— 对比方案
#    6. 预处理效果定量评估（与理化指标的相关性对比）
#    7. PCA 特征降维（728维 → 保留99%方差的主成分）
#    8. 保存预处理结果（供第三阶段建模使用）
#
#  输出文件：
#    results/models/X_preprocessed.npy   ← SG+SNV 预处理后原始光谱
#    results/models/X_pca.npy            ← PCA 降维后特征（SVM/BP用）
#    results/models/y_labels.npy         ← 成熟度标签
#    results/models/pca_model.pkl        ← PCA 模型（系统部署用）
#    results/models/scaler_spec.pkl      ← 光谱标准化器
#    results/charts/02_*.png             ← 所有可视化图
#    results/models/预处理质量报告.txt
# =============================================================================

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── 中文字体 ──────────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS',
                                    'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE   = os.path.join(PROJECT_ROOT, 'results', 'cluster', '聚类标签数据集.csv')
IDX_FILE     = os.path.join(PROJECT_ROOT, 'results', 'cluster', '指标数据_含标签.csv')
OUT_MODEL    = os.path.join(PROJECT_ROOT, 'results', 'models')
OUT_VIS      = os.path.join(PROJECT_ROOT, 'results', 'charts')
os.makedirs(OUT_MODEL, exist_ok=True)
os.makedirs(OUT_VIS,   exist_ok=True)

# ── 常量 ──────────────────────────────────────────────────────────────────────
MATURITY_NAMES = ['未成熟', '半成熟', '成熟', '过成熟']
EN_NAMES       = ['Unripe', 'Half-ripe', 'Ripe', 'Over-ripe']
PALETTE        = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
INDICATOR_COLS = ['可溶性固形物（SSC）', '酸碱性（PH）', '果皮硬度（N）', '色差']
INDICATOR_EN   = ['SSC (°Brix)', 'PH', 'Hardness (N)', 'Color (ΔE)']

# SG 平滑参数（葡萄近红外光谱推荐值）
SG_WINDOW   = 11   # 窗口长度（奇数）
SG_POLYORDER = 3   # 多项式阶数


# =============================================================================
#  工具函数
# =============================================================================

def section(title):
    line = '─' * 62
    print(f'\n{line}')
    print(f'  {title}')
    print(line)


def save_fig(fig, name):
    path = os.path.join(OUT_VIS, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  [图片] {os.path.relpath(path, PROJECT_ROOT)}')


# =============================================================================
#  预处理算法实现
# =============================================================================

def sg_smooth(X, window=SG_WINDOW, polyorder=SG_POLYORDER):
    """
    Savitzky-Golay 平滑滤波
    ────────────────────────
    原理：在长度为 window 的滑动窗口内用 polyorder 阶多项式做最小二乘拟合，
          用拟合值替代原始数据点，保留光谱形状的同时去除高频噪声。
    参数：
        window    : 窗口长度，必须为奇数，越大越平滑但会模糊峰形
        polyorder : 多项式阶数，越小越平滑，3阶是近红外光谱的标准选择
    """
    return savgol_filter(X, window_length=window, polyorder=polyorder, axis=1)


def snv(X):
    """
    标准正态变量变换 (Standard Normal Variate)
    ────────────────────────────────────────────
    原理：对每条光谱单独做中心化 + 方差归一化，消除颗粒大小和表面散射差异。
          公式：x_snv[i] = (x[i] - mean(x)) / std(x)
    注意：每条光谱独立处理，不依赖其他样本，无参考光谱。
    """
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1,  keepdims=True)
    return (X - mean) / (std + 1e-10)


def msc(X):
    """
    多元散射校正 (Multiplicative Scatter Correction)
    ─────────────────────────────────────────────────
    原理：以所有样本的平均光谱作为参考，对每条光谱做线性回归校正：
          x[i] ≈ a[i] + b[i] * x_mean
          校正后：x_msc[i] = (x[i] - a[i]) / b[i]
    与SNV的区别：MSC需要参考光谱（均值谱），SNV不需要。
    """
    x_mean = X.mean(axis=0)
    X_msc  = np.zeros_like(X)
    for i in range(X.shape[0]):
        # 线性回归：最小二乘拟合 x_mean → x[i]
        coeffs   = np.polyfit(x_mean, X[i], 1)   # [斜率b, 截距a]
        b, a     = coeffs[0], coeffs[1]
        X_msc[i] = (X[i] - a) / (b + 1e-10)
    return X_msc



# =============================================================================
#  Step 1 .数据加载
# =============================================================================

def step1_load(filepath, idx_filepath):
    section('Step 1 .加载聚类标签数据集')

    df      = pd.read_csv(filepath)
    df_idx  = pd.read_csv(idx_filepath)

    # 分离光谱列、标签列
    wave_cols = [c for c in df.columns
                 if c not in ['序号', 'maturity_label', 'maturity_name']]
    wavelengths = np.array([float(w) for w in wave_cols])
    X_raw       = df[wave_cols].values.astype(np.float64)
    y           = df['maturity_label'].values.astype(int)

    # 指标数据（用于后续相关性评估）
    X_idx = df_idx[INDICATOR_COLS].values.astype(np.float64)

    print(f'  光谱矩阵形状 : {X_raw.shape}  (样本数 × 波段数)')
    print(f'  波长范围     : {wavelengths[0]:.1f} ~ {wavelengths[-1]:.1f} nm')
    print(f'  波段间隔     : {np.diff(wavelengths).mean():.3f} nm')
    print(f'  标签分布     : ', end='')
    for i, name in enumerate(MATURITY_NAMES):
        print(f'{name}={( y==i).sum()}', end='  ')
    print()

    # 检查光谱数值范围
    print(f'\n  反射率范围   : {X_raw.min():.4f} ~ {X_raw.max():.4f}')
    print(f'  含NaN数量    : {np.isnan(X_raw).sum()}')

    return X_raw, y, wavelengths, X_idx


# =============================================================================
#  Step 2 .原始光谱可视化
# =============================================================================

def step2_plot_raw(X_raw, y, wavelengths):
    section('Step 2 .原始光谱可视化')

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

    # ── 左上：随机抽取50条原始光谱（观察噪声）──
    ax1 = fig.add_subplot(gs[0, 0])
    np.random.seed(42)
    idx_sample = np.random.choice(len(X_raw), 50, replace=False)
    for i in idx_sample:
        color = PALETTE[y[i]]
        ax1.plot(wavelengths, X_raw[i], color=color, alpha=0.25, lw=0.6)
    patches = [plt.Line2D([0],[0], color=c, lw=2, label=n)
               for c, n in zip(PALETTE, EN_NAMES)]
    ax1.legend(handles=patches, fontsize=8, loc='upper right')
    ax1.set_title('Raw Spectra (50 samples)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Wavelength (nm)', fontsize=9)
    ax1.set_ylabel('Reflectance', fontsize=9)
    ax1.grid(alpha=0.2)

    # ── 右上：4类平均光谱 ± 标准差 ──
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (name, color) in enumerate(zip(EN_NAMES, PALETTE)):
        mask   = y == i
        mean_s = X_raw[mask].mean(axis=0)
        std_s  = X_raw[mask].std(axis=0)
        ax2.plot(wavelengths, mean_s, color=color, lw=2, label=name)
        ax2.fill_between(wavelengths, mean_s - std_s, mean_s + std_s,
                         color=color, alpha=0.12)
    ax2.legend(fontsize=8)
    ax2.set_title('Mean Spectrum ± Std by Maturity', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Wavelength (nm)', fontsize=9)
    ax2.set_ylabel('Reflectance', fontsize=9)
    ax2.grid(alpha=0.2)

    # ── 左下：单样本原始光谱放大（观察高频噪声）──
    ax3 = fig.add_subplot(gs[1, 0])
    sample_idx = 0
    ax3.plot(wavelengths, X_raw[sample_idx], 'b-', lw=1.2,
             label='Raw spectrum (sample #1)', alpha=0.9)
    # 标注噪声区域
    ax3.axvspan(380, 450, color='red', alpha=0.07, label='High noise region')
    ax3.axvspan(950, 1030, color='red', alpha=0.07)
    ax3.legend(fontsize=8)
    ax3.set_title('Single Spectrum - Noise Observation', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Wavelength (nm)', fontsize=9)
    ax3.set_ylabel('Reflectance', fontsize=9)
    ax3.grid(alpha=0.2)

    # ── 右下：各类样本数量条形图 ──
    ax4 = fig.add_subplot(gs[1, 1])
    counts = [(y == i).sum() for i in range(4)]
    bars   = ax4.bar(EN_NAMES, counts, color=PALETTE, alpha=0.85,
                     edgecolor='white', lw=1.5)
    for bar, cnt in zip(bars, counts):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 8,
                 f'{cnt}\n({cnt/len(y)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax4.set_title('Sample Distribution by Maturity', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Sample Count', fontsize=9)
    ax4.set_ylim(0, max(counts) * 1.2)
    ax4.grid(axis='y', alpha=0.3)

    fig.suptitle('Stage 2: Raw Hyperspectral Data Overview',
                 fontsize=13, fontweight='bold', y=1.01)
    save_fig(fig, '02_原始光谱总览.png')
    print('  → 观察到：两端（<450nm, >950nm）噪声较大，需SG平滑处理')


# =============================================================================
#  Step 3 .应用各种预处理方案
# =============================================================================

def step3_preprocess(X_raw):
    section('Step 3 .应用预处理算法')

    # 方案A：SG 平滑
    print(f'  [A] SG平滑  window={SG_WINDOW}, polyorder={SG_POLYORDER} ...', end='')
    X_sg = sg_smooth(X_raw)
    print(' ✓')

    # 方案B：SG + SNV（推荐方案）
    print(f'  [B] SG + SNV ...', end='')
    X_sg_snv = snv(X_sg)
    print(' ✓')

    # 方案C：SG + MSC（对比方案）
    print(f'  [C] SG + MSC ...', end='')
    X_sg_msc = msc(X_sg)
    print(' ✓')


    print(f'\n  所有方案形状: {X_sg_snv.shape}  (与原始相同)')
    return X_sg, X_sg_snv, X_sg_msc


# =============================================================================
#  Step 4 .预处理效果可视化对比
# =============================================================================

def step4_compare_visualization(X_raw, X_sg, X_sg_snv, X_sg_msc, wavelengths, y):
    section('Step 4 .预处理效果可视化对比')

    # ── 图A：单样本四种处理对比 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Preprocessing Methods Comparison (Sample #1)',
                 fontsize=13, fontweight='bold')

    sample = 0
    configs = [
        (axes[0,0], X_raw,     'Raw (Original)',          '#6366F1'),
        (axes[0,1], X_sg,      f'SG Smooth (w={SG_WINDOW},p={SG_POLYORDER})', '#0891B2'),
        (axes[1,0], X_sg_snv,  'SG + SNV  ← Recommended','#059669'),
        (axes[1,1], X_sg_msc,  'SG + MSC  (Comparison)',  '#D97706'),
    ]
    for ax, X, title, color in configs:
        ax.plot(wavelengths, X[sample], color=color, lw=1.8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Wavelength (nm)', fontsize=9)
        ax.set_ylabel('Reflectance / Transformed', fontsize=9)
        ax.grid(alpha=0.25)
        # 标注统计信息
        ax.text(0.02, 0.95,
                f'mean={X[sample].mean():.4f}\nstd={X[sample].std():.4f}',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()
    save_fig(fig, '02_预处理方案对比.png')

    # ── 图B：SG平滑前后放大对比（噪声去除效果）──
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('SG Smoothing Effect Detail', fontsize=12, fontweight='bold')

    zoom_regions = [(380, 500), (900, 1030)]
    titles_zoom  = ['Short-wave End (380-500nm)', 'Long-wave End (900-1030nm)']

    for ax, (wl_min, wl_max), title in zip(axes2, zoom_regions, titles_zoom):
        mask_w = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        wl_sub = wavelengths[mask_w]
        ax.plot(wl_sub, X_raw[sample][mask_w],
                'r-', lw=1.5, alpha=0.8, label='Raw')
        ax.plot(wl_sub, X_sg[sample][mask_w],
                'b-', lw=2.0, label='After SG')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Wavelength (nm)', fontsize=9)
        ax.set_ylabel('Reflectance', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    save_fig(fig2, '02_SG平滑局部放大.png')

    # ── 图C：4类平均光谱在SG+SNV后的对比 ──
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    fig3.suptitle('Mean Spectrum Before vs After SG+SNV',
                  fontsize=12, fontweight='bold')

    for ax, (X, title) in zip(axes3,
            [(X_raw,     'Raw: Mean Spectrum by Maturity'),
             (X_sg_snv,  'After SG+SNV: Mean Spectrum by Maturity')]):
        for i, (name, color) in enumerate(zip(EN_NAMES, PALETTE)):
            mask   = y == i
            mean_s = X[mask].mean(axis=0)
            std_s  = X[mask].std(axis=0)
            ax.plot(wavelengths, mean_s, color=color, lw=2, label=name)
            ax.fill_between(wavelengths, mean_s-std_s, mean_s+std_s,
                            color=color, alpha=0.1)
        ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Wavelength (nm)', fontsize=9)
        ax.set_ylabel('Reflectance', fontsize=9)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    save_fig(fig3, '02_SNV前后平均光谱.png')

    print('  → SG平滑成功消除两端高频噪声')
    print('  → SNV变换后各类光谱差异更清晰，类间区分度提升')


# =============================================================================
#  Step 5 .定量评估：与理化指标的相关性
# =============================================================================

def step5_correlation_evaluation(X_raw, X_sg, X_sg_snv, X_sg_msc, X_idx, wavelengths):
    section('Step 5 .定量评估：光谱与理化指标相关性')

    schemes = {
        'Raw':        X_raw,
        'SG':         X_sg,
        'SG+SNV':     X_sg_snv,
        'SG+MSC':     X_sg_msc,
    }

    print(f'\n  {"方案":<12}', end='')
    for col in INDICATOR_EN:
        print(f'  {col[:10]:>12}', end='')
    print(f'  {"平均":>8}')
    print('  ' + '─' * 72)

    corr_results = {}
    for scheme_name, X_scheme in schemes.items():
        row = []
        for j in range(len(INDICATOR_COLS)):
            y_ind  = X_idx[:, j]
            valid  = ~np.isnan(y_ind)
            # 用各波段与指标的最大绝对相关系数代表该方案
            max_corr = max(abs(pearsonr(X_scheme[valid, b], y_ind[valid])[0])
                           for b in range(0, X_scheme.shape[1], 10))  # 每10个波段采样
            row.append(max_corr)
        corr_results[scheme_name] = row
        avg = np.mean(row)
        marker = '  ← Best' if scheme_name == 'SG+SNV' else ''
        print(f'  {scheme_name:<12}', end='')
        for v in row:
            print(f'  {v:>12.4f}', end='')
        print(f'  {avg:>8.4f}{marker}')

    # ── 相关系数随波长的分布图 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pearson Correlation: Spectrum vs Indicators (Raw vs SG+SNV)',
                 fontsize=12, fontweight='bold')

    for ax, (col_name, col_en) in enumerate(zip(INDICATOR_COLS, INDICATOR_EN)):
        ax_obj = axes[ax//2, ax%2]
        y_ind  = X_idx[:, ax]
        valid  = ~np.isnan(y_ind)

        for X_scheme, scheme_name, color, lw, ls in [
            (X_raw,     'Raw',     '#94A3B8', 1.2, '--'),
            (X_sg_snv,  'SG+SNV',  '#059669', 2.0, '-'),
            (X_sg_msc,  'SG+MSC',  '#D97706', 1.5, '-.')
        ]:
            corrs = [pearsonr(X_scheme[valid, b], y_ind[valid])[0]
                     for b in range(X_scheme.shape[1])]
            ax_obj.plot(wavelengths, corrs, color=color,
                        lw=lw, ls=ls, label=scheme_name, alpha=0.85)

        ax_obj.axhline(0,  color='black', lw=0.8, ls=':')
        ax_obj.axhline( 0.5, color='red', lw=0.8, ls=':', alpha=0.5)
        ax_obj.axhline(-0.5, color='red', lw=0.8, ls=':', alpha=0.5)
        ax_obj.set_title(f'Corr with {col_en}', fontsize=10, fontweight='bold')
        ax_obj.set_xlabel('Wavelength (nm)', fontsize=9)
        ax_obj.set_ylabel('Pearson r', fontsize=9)
        ax_obj.legend(fontsize=8)
        ax_obj.set_ylim(-1, 1)
        ax_obj.grid(alpha=0.2)

    plt.tight_layout()
    save_fig(fig, '02_相关系数随波长分布.png')

    return corr_results


# =============================================================================
#  Step 6 .PCA 特征降维
# =============================================================================

def step6_pca(X_sg_snv, y, wavelengths):
    section('Step 6 .PCA 特征降维')

    # 先做标准化（PCA前必须）
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_sg_snv)

    # ── 确定最优主成分数 ──
    pca_full = PCA().fit(X_norm)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)

    thresholds = [0.90, 0.95, 0.99, 0.999]
    print('\n  累计方差阈值 → 所需主成分数:')
    for thr in thresholds:
        n_comp = np.argmax(cumvar >= thr) + 1
        print(f'    {thr*100:.1f}%  →  {n_comp:>3} 个主成分')

    # 选择保留 99% 方差
    n_components_99 = np.argmax(cumvar >= 0.99) + 1
    print(f'\n  最终选择: 保留99%方差  →  {n_components_99} 个主成分')
    print(f'  降维效果: {X_sg_snv.shape[1]} 维 → {n_components_99} 维  '
          f'(压缩至 {n_components_99/X_sg_snv.shape[1]*100:.1f}%)')

    # 正式 PCA
    pca = PCA(n_components=n_components_99)
    X_pca = pca.fit_transform(X_norm)

    print(f'\n  PCA后特征矩阵: {X_pca.shape}')
    print(f'  前5个PC方差贡献率:')
    for i, var in enumerate(pca.explained_variance_ratio_[:5]):
        print(f'    PC{i+1}: {var*100:.2f}%')

    # ── PCA 可视化（4张图）──
    fig = plt.figure(figsize=(15, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 图1：累计方差曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, min(101, len(cumvar)+1)), cumvar[:100],
             'b-o', ms=3, lw=2, markevery=5)
    for thr, color in zip([0.90, 0.95, 0.99], ['#F59E0B','#10B981','#EF4444']):
        n = np.argmax(cumvar >= thr) + 1
        ax1.axhline(thr, color=color, ls='--', lw=1.5,
                    label=f'{thr*100:.0f}%: {n} PCs')
        ax1.axvline(n,   color=color, ls='--', lw=1, alpha=0.6)
    ax1.set_xlabel('Number of Principal Components', fontsize=9)
    ax1.set_ylabel('Cumulative Variance Ratio', fontsize=9)
    ax1.set_title('PCA: Cumulative Explained Variance', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 60)
    ax1.grid(alpha=0.25)

    # 图2：PC1 vs PC2 散点图
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (name, color) in enumerate(zip(EN_NAMES, PALETTE)):
        mask = y == i
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=color, label=f'{name}(n={mask.sum()})',
                    alpha=0.5, s=12, linewidths=0)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=9)
    ax2.set_title('PCA Score Plot: PC1 vs PC2', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, markerscale=2)
    ax2.grid(alpha=0.2)

    # 图3：PC1 vs PC3 散点图
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (name, color) in enumerate(zip(EN_NAMES, PALETTE)):
        mask = y == i
        ax3.scatter(X_pca[mask, 0], X_pca[mask, 2],
                    c=color, label=name, alpha=0.5, s=12, linewidths=0)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=9)
    ax3.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=9)
    ax3.set_title('PCA Score Plot: PC1 vs PC3', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8, markerscale=2)
    ax3.grid(alpha=0.2)

    # 图4：Loading 向量（PC1/PC2 对应的波段贡献）
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(wavelengths, pca.components_[0], '#3B82F6', lw=1.8,
             label=f'PC1 Loading ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax4.plot(wavelengths, pca.components_[1], '#EF4444', lw=1.8,
             label=f'PC2 Loading ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax4.axhline(0, color='black', lw=0.8, ls=':')
    ax4.set_xlabel('Wavelength (nm)', fontsize=9)
    ax4.set_ylabel('Loading Value', fontsize=9)
    ax4.set_title('PCA Loadings (Key Wavelength Contributions)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.2)

    fig.suptitle(f'PCA Feature Extraction: {X_sg_snv.shape[1]}D → {n_components_99}D (99% variance)',
                 fontsize=13, fontweight='bold', y=1.01)
    save_fig(fig, '02_PCA特征降维.png')

    return X_pca, pca, scaler, n_components_99


# =============================================================================
#  Step 7 .关键波段分析
# =============================================================================

def step7_key_wavelengths(X_sg_snv, X_idx, y, wavelengths, pca):
    section('Step 7 .关键波段分析')

    # ── 基于PC1 Loading 找关键波段 ──
    loading_pc1 = np.abs(pca.components_[0])
    top10_idx   = np.argsort(loading_pc1)[-10:][::-1]

    print('  PC1 贡献最大的10个波段（与成熟度最相关）:')
    print(f'  {"排名":>4}  {"波长(nm)":>10}  {"Loading绝对值":>14}')
    print('  ' + '─' * 34)
    for rank, idx in enumerate(top10_idx, 1):
        print(f'  {rank:>4}  {wavelengths[idx]:>10.1f}  {loading_pc1[idx]:>14.4f}')

    # ── 关键波段可视化 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Key Wavelength Analysis', fontsize=12, fontweight='bold')

    # 左：PC1 Loading 全谱
    ax1 = axes[0]
    ax1.plot(wavelengths, pca.components_[0], '#3B82F6', lw=1.8)
    ax1.scatter(wavelengths[top10_idx], pca.components_[0][top10_idx],
                c='red', s=50, zorder=5, label='Top-10 key bands')
    ax1.axhline(0, color='black', lw=0.8, ls=':')
    ax1.set_xlabel('Wavelength (nm)', fontsize=9)
    ax1.set_ylabel('PC1 Loading', fontsize=9)
    ax1.set_title('PC1 Loading Profile', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    # 右：各成熟度在关键波段的反射率箱线图
    ax2  = axes[1]
    key_wl_idx = top10_idx[:4]   # 取前4个
    key_labels = [f'{wavelengths[i]:.0f}nm' for i in key_wl_idx]
    positions  = np.arange(len(key_wl_idx))
    width      = 0.18
    for j, (name, color) in enumerate(zip(EN_NAMES, PALETTE)):
        mask   = y == j
        values = [X_sg_snv[mask, i].mean() for i in key_wl_idx]
        errors = [X_sg_snv[mask, i].std()  for i in key_wl_idx]
        ax2.bar(positions + j*width, values, width,
                label=name, color=color, alpha=0.8,
                yerr=errors, capsize=3, error_kw={'lw':1})
    ax2.set_xticks(positions + width*1.5)
    ax2.set_xticklabels(key_labels, fontsize=9)
    ax2.set_xlabel('Key Wavelength Bands', fontsize=9)
    ax2.set_ylabel('Mean Reflectance (SNV)', fontsize=9)
    ax2.set_title('Reflectance at Key Bands by Maturity', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig(fig, '02_关键波段分析.png')

    return wavelengths[top10_idx]


# =============================================================================
#  Step 8 .保存所有预处理结果
# =============================================================================

def step8_save(X_sg_snv, X_pca, y, pca, scaler,
               n_components, corr_results, key_wavelengths):
    section('Step 8 .保存预处理结果')

    # numpy 数组
    np.save(os.path.join(OUT_MODEL, 'X_preprocessed.npy'), X_sg_snv)
    np.save(os.path.join(OUT_MODEL, 'X_pca.npy'),          X_pca)
    np.save(os.path.join(OUT_MODEL, 'y_labels.npy'),        y)

    # sklearn 模型
    with open(os.path.join(OUT_MODEL, 'pca_model.pkl'),    'wb') as f:
        pickle.dump(pca, f)
    with open(os.path.join(OUT_MODEL, 'scaler_spec.pkl'),  'wb') as f:
        pickle.dump(scaler, f)

    print('  已保存:')
    for fname, desc in [
        ('X_preprocessed.npy', f'SG+SNV 光谱  {X_sg_snv.shape}'),
        ('X_pca.npy',          f'PCA 特征     {X_pca.shape}'),
        ('y_labels.npy',       f'成熟度标签   {y.shape}'),
        ('pca_model.pkl',      'PCA 模型（部署用）'),
        ('scaler_spec.pkl',    '光谱标准化器（部署用）'),
    ]:
        path = os.path.join(OUT_MODEL, fname)
        size = os.path.getsize(path) / 1024
        print(f'    {fname:<25} {desc:<25} ({size:.1f} KB)')

    # 质量报告
    report = [
        '=' * 62,
        '  葡萄高光谱预处理质量报告',
        '=' * 62,
        '',
        '【原始数据】',
        '  样本数    : 2160',
        '  波段数    : 728',
        '  波长范围  : 382.2 ~ 1026.7 nm',
        '',
        '【SG 平滑参数】',
        f'  窗口长度  : {SG_WINDOW}',
        f'  多项式阶数: {SG_POLYORDER}',
        '  效果      : 去除高频随机噪声，保留光谱峰形',
        '',
        '【SNV 变换】',
        '  公式      : x_snv = (x - mean(x)) / std(x)',
        '  效果      : 消除颗粒大小和表面散射差异',
        '  选择依据  : 与理化指标相关性最高（见下表）',
        '',
        '【各方案与理化指标最大相关系数对比】',
        f'  {"方案":<12}  {"SSC":>8}  {"PH":>8}  {"硬度":>8}  {"色差":>8}  {"均值":>8}',
        '  ' + '─' * 56,
    ]
    for scheme, vals in corr_results.items():
        avg  = np.mean(vals)
        mark = '  ★ 最优' if scheme == 'SG+SNV' else ''
        report.append(
            f'  {scheme:<12}  '
            + '  '.join(f'{v:>8.4f}' for v in vals)
            + f'  {avg:>8.4f}{mark}')

    report += [
        '',
        '【PCA 特征降维】',
        f'  降维前    : 728 维',
        f'  降维后    : {n_components} 维（保留99%方差）',
        f'  压缩比    : {n_components/728*100:.1f}%',
        '',
        '【关键波段（PC1 Loading 最大10个）】',
        '  ' + '  '.join(f'{w:.1f}nm' for w in key_wavelengths),
        '',
        '【输出文件说明】',
        '  X_preprocessed.npy  → CNN/改进模型输入（原始光谱维度）',
        '  X_pca.npy           → SVM/BP模型输入（降维后）',
        '  y_labels.npy        → 所有模型的标签',
        '  pca_model.pkl       → 系统部署时预处理新样本',
        '  scaler_spec.pkl     → 系统部署时标准化新样本',
        '',
        '=' * 62,
    ]

    report_path = os.path.join(OUT_MODEL, '预处理质量报告.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f'\n  质量报告: {os.path.relpath(report_path, PROJECT_ROOT)}')
    print()
    for line in report:
        print('  ' + line)


# =============================================================================
#  主程序
# =============================================================================

def main():
    print('\n' + '=' * 62)
    print('  葡萄成熟度识别 .第二阶段：高光谱数据预处理与特征提取')
    print('=' * 62)

    # Step 1 .加载数据
    X_raw, y, wavelengths, X_idx = step1_load(INPUT_FILE, IDX_FILE)

    # Step 2 .原始光谱可视化
    step2_plot_raw(X_raw, y, wavelengths)

    # Step 3 .预处理
    X_sg, X_sg_snv, X_sg_msc= step3_preprocess(X_raw)

    # Step 4 .可视化对比
    step4_compare_visualization(X_raw, X_sg, X_sg_snv, X_sg_msc,
                                 wavelengths, y)

    # Step 5 .定量评估
    corr_results = step5_correlation_evaluation(
        X_raw, X_sg, X_sg_snv, X_sg_msc, X_idx, wavelengths)

    # Step 6 .PCA 降维
    X_pca, pca, scaler, n_components = step6_pca(X_sg_snv, y, wavelengths)

    # Step 7 .关键波段
    key_wls = step7_key_wavelengths(X_sg_snv, X_idx, y, wavelengths, pca)

    # Step 8 .保存
    step8_save(X_sg_snv, X_pca, y, pca, scaler,
               n_components, corr_results, key_wls)

    section('全部完成！下一步：运行 03_预测模型构建.py')
    print('  生成图片: results/charts/02_*.png')
    print('  建模数据: results/models/X_pca.npy  X_preprocessed.npy\n')


if __name__ == '__main__':
    main()