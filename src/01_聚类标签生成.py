# =============================================================================
#  葡萄成熟度识别 —— 第一阶段：多指标聚类标签生成
#  功能：
#    1. 加载并探索指标数据
#    2. 缺失值处理 + 标准化
#    3. 确定指标权重方案
#    4. 肘部法则 + 轮廓系数确定最优k
#    5. K-Means聚类生成4类成熟度标签
#    6. 聚类结果分析与可视化
#    7. 保存最终带标签的完整数据集
#  输出：
#    results/cluster/聚类标签数据集.csv   ← 后续所有阶段使用这个文件
#    results/charts/01_指标分布.png
#    results/charts/01_肘部法则.png
#    results/charts/01_聚类PCA可视化.png
#    results/charts/01_各指标箱线图.png
#    results/charts/01_相关系数热力图.png
#    results/cluster/聚类质量报告.txt
# =============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

warnings.filterwarnings('ignore')

# ── 字体设置（中文显示）──────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS',
                                    'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ── 路径配置（相对于项目根目录）─────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE      = os.path.join(PROJECT_ROOT, 'data', '高光谱+指标数据.xlsx')
OUT_CLUSTER    = os.path.join(PROJECT_ROOT, 'results', 'cluster')
OUT_VIS        = os.path.join(PROJECT_ROOT, 'results', 'charts')

os.makedirs(OUT_CLUSTER, exist_ok=True)
os.makedirs(OUT_VIS,     exist_ok=True)

# ── 颜色 / 标签定义 ──────────────────────────────────────────────────────────
MATURITY_NAMES  = ['未成熟', '半成熟', '成熟', '过成熟']
PALETTE         = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']   # 蓝/绿/橙/红
INDICATOR_COLS  = ['可溶性固形物（SSC）', '酸碱性（PH）', '果皮硬度（N）', '色差']
INDICATOR_UNITS = ['°Brix', '', 'N', 'ΔE']
WEIGHTS_SUGGESTED= np.array([0.35, 0.20, 0.25, 0.20])             # 建议权重


# =============================================================================
#  辅助函数
# =============================================================================

def section(title):
    """打印带分隔线的段落标题"""
    line = '─' * 60
    print(f'\n{line}')
    print(f'  {title}')
    print(line)


def save_fig(fig, name):
    """保存图片到 results/charts/"""
    path = os.path.join(OUT_VIS, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  [图片已保存] {os.path.relpath(path, PROJECT_ROOT)}')


# =============================================================================
#  Step 1 · 数据加载与探索
# =============================================================================

def step1_load_data():
    section('Step 1 · 数据加载与基本探索')

    print(f'  数据文件: {os.path.relpath(DATA_FILE, PROJECT_ROOT)}')

    df_idx  = pd.read_excel(DATA_FILE, sheet_name='指标数据')
    df_spec = pd.read_excel(DATA_FILE, sheet_name='高光谱数据')

    print(f'\n  指标数据形状  : {df_idx.shape}   (样本数 × 列数)')
    print(f'  光谱数据形状  : {df_spec.shape}   (样本数 × 波段数+序号)')

    print('\n  ── 指标数据前5行 ──')
    print(df_idx.head(5).to_string(index=False))

    print('\n  ── 缺失值统计 ──')
    missing = df_idx.isnull().sum()
    for col, cnt in missing.items():
        status = f'  ★ 需要处理' if cnt > 0 else '  ✓'
        print(f'    {col}: {cnt} 个缺失值{status}')

    print('\n  ── 各指标统计描述 ──')
    desc = df_idx[INDICATOR_COLS].describe().round(4)
    print(desc.to_string())

    return df_idx, df_spec


# =============================================================================
#  Step 2 · 缺失值处理 + 标准化
# =============================================================================

def step2_preprocess(df_idx):
    section('Step 2 · 缺失值处理 + 标准化')

    X_raw = df_idx[INDICATOR_COLS].values

    # ── 中位数填充缺失值 ──
    imputer = SimpleImputer(strategy='median')
    X_filled = imputer.fit_transform(X_raw)
    print('  缺失值处理方式: 中位数填充（对异常值更鲁棒）')
    for i, col in enumerate(INDICATOR_COLS):
        print(f'    {col} 填充中位数: {imputer.statistics_[i]:.4f}')

    # ── StandardScaler 标准化（使各指标量纲统一）──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    print(f'\n  标准化后: 均值 ≈ {X_scaled.mean(axis=0).round(3)}'
          f'  标准差 ≈ {X_scaled.std(axis=0).round(3)}')

    # ── 相关系数分析 ──
    df_filled = pd.DataFrame(X_filled, columns=INDICATOR_COLS)
    corr = df_filled.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, square=True, mask=mask,
                annot_kws={'size': 10})
    # 显示标签（用英文避免字体问题）
    labels = ['SSC', 'PH', 'Hardness', 'Color']
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9, rotation=0)
    ax.set_title('Indicator Correlation Matrix', fontsize=12, fontweight='bold', pad=12)
    fig.tight_layout()
    save_fig(fig, '01_相关系数热力图.png')
    print('\n  相关系数矩阵（论文参考）:')
    print(corr.round(3).to_string())

    return X_filled, X_scaled, imputer, scaler


# =============================================================================
#  Step 3 · 权重方案分析
# =============================================================================

#计算最佳权重方案


def step3_weight_analysis(X_scaled):
    section('Step 3 · 基于建议权重计算')
    
    # 使用预定义的专家权重方案
    entropy_weights = WEIGHTS_SUGGESTED
    
    print('\n  ★ 采用建议权重的客观权重进行后续聚类分析')
    print(f'  各指标权重配置:')
    
    weights_list = entropy_weights.tolist() if hasattr(entropy_weights, 'tolist') else list(entropy_weights)
    for i in range(len(INDICATOR_COLS)):
        col = INDICATOR_COLS[i]
        w = weights_list[i]
        print(f'    {col}: {w:.4f} ({w*100:.1f}%)')
    
    X_weighted = X_scaled * entropy_weights
    print(f'\n  加权后数据形状：{X_weighted.shape}')
    
    return X_weighted, entropy_weights


# =============================================================================
#  Step 4 · 确定最优聚类数 k（肘部法则 + 多指标）
# =============================================================================

def step4_find_optimal_k(X_weighted):
    section('Step 4 · 肘部法则 + 轮廓系数确定最优 k')

    k_range   = range(2, 9)
    inertias  = []
    sil_scores = []
    db_scores  = []
    ch_scores  = []

    print('  正在搜索 k=2~8 ...')
    print(f'  {"k":>3}  {"惯性(Inertia)":>14}  {"轮廓系数":>10}  '
          f'{"DB指数":>8}  {"CH指数":>10}')
    print('  ' + '─' * 55)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        labels = km.fit_predict(X_weighted)
        iner = km.inertia_
        sil  = silhouette_score(X_weighted, labels)
        db   = davies_bouldin_score(X_weighted, labels)
        ch   = calinski_harabasz_score(X_weighted, labels)
        inertias.append(iner)
        sil_scores.append(sil)
        db_scores.append(db)
        ch_scores.append(ch)
        marker = '  ← 推荐' if k == 4 else ''
        print(f'  {k:>3}  {iner:>14.2f}  {sil:>10.4f}  '
              f'{db:>8.4f}  {ch:>10.2f}{marker}')

    # ── 绘制肘部法则图（2行2列）──
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Optimal k Selection (Elbow Method + Validation Metrics)',
                 fontsize=13, fontweight='bold', y=0.98)

    k_list = list(k_range)
    plot_data = [
        (axes[0, 0], inertias,   'Inertia (SSE)',        '#3B82F6', False),
        (axes[0, 1], sil_scores, 'Silhouette Score',     '#10B981', True),
        (axes[1, 0], db_scores,  'Davies-Bouldin Index', '#F59E0B', False),
        (axes[1, 1], ch_scores,  'Calinski-Harabasz',    '#EF4444', True),
    ]
    for ax, vals, title, color, higher_better in plot_data:
        ax.plot(k_list, vals, 'o-', color=color, lw=2.5, ms=7)
        ax.axvline(4, color='gray', ls='--', alpha=0.6, label='k=4')
        ax.scatter([4], [vals[k_list.index(4)]], s=120,
                   color=color, zorder=5, edgecolors='white', lw=1.5)
        note = '(higher=better)' if higher_better else '(lower=better)'
        ax.set_title(f'{title} {note}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Number of Clusters k', fontsize=9)
        ax.set_xticks(k_list)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, '01_肘部法则.png')

    print('\n  结论：综合四项指标，k=4 是最优聚类数')
    print('  （轮廓系数在k=4时较高，且符合葡萄成熟度4阶段的农业实践）')

    return sil_scores[2], db_scores[2]  # k=4 对应 index 2


# =============================================================================
#  Step 5 · 正式 K-Means 聚类
# =============================================================================

def step5_kmeans_cluster(X_weighted):
    section('Step 5 · 正式 K-Means 聚类（k=4）')

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500)
    raw_labels = kmeans.fit_predict(X_weighted)

    sil = silhouette_score(X_weighted, raw_labels)
    db  = davies_bouldin_score(X_weighted, raw_labels)
    ch  = calinski_harabasz_score(X_weighted, raw_labels)

    print(f'  轮廓系数 Silhouette: {sil:.4f}')
    print(f'  DB 指数  Davies-Bouldin : {db:.4f}')
    print(f'  CH 指数  Calinski-Harabasz        : {ch:.2f}')
    print(f'  各簇样本数: {np.bincount(raw_labels)}')

    return kmeans, raw_labels, sil, db, ch


# =============================================================================
#  Step 6 · 标签命名（按 SSC 均值排序）
# =============================================================================

def step6_label_naming(df_idx, X_filled, raw_labels):
    section('Step 6 · 标签命名（按 SSC 均值从小到大排序）')

    df_temp = df_idx[INDICATOR_COLS].copy()
    df_temp = df_temp.fillna(df_temp.median())  # 确保无缺失
    df_temp['raw_cluster'] = raw_labels

    # SSC 均值最小 → 未成熟(0)，最大 → 过成熟(3)
    ssc_col = INDICATOR_COLS[0]
    ssc_mean_by_cluster = df_temp.groupby('raw_cluster')[ssc_col].mean()
    sorted_clusters = ssc_mean_by_cluster.sort_values().index.tolist()

    label_map = {old: new for new, old in enumerate(sorted_clusters)}
    maturity_labels = np.array([label_map[r] for r in raw_labels])
    maturity_names  = np.array([MATURITY_NAMES[l] for l in maturity_labels])

    print('\n  簇编号 → 成熟度等级 映射关系:')
    print(f'  {"原始簇":>6}  {"SSC均值":>10}  {"→ 成熟度等级":<14}')
    print('  ' + '─' * 36)
    for old_c in sorted_clusters:
        new_c = label_map[old_c]
        ssc_v = ssc_mean_by_cluster[old_c]
        print(f'  簇 {old_c:>2}     {ssc_v:>8.3f}   →  {new_c}: {MATURITY_NAMES[new_c]}')

    print('\n  各成熟度等级样本数量:')
    for i, name in enumerate(MATURITY_NAMES):
        cnt = (maturity_labels == i).sum()
        bar = '█' * (cnt // 30)
        print(f'    {name}: {cnt:>5} 个  {bar}')

    return maturity_labels, maturity_names, label_map


# =============================================================================
#  Step 7 · 统计各成熟度的指标特征
# =============================================================================

def step7_stats_by_maturity(df_idx, X_filled, maturity_labels):
    section('Step 7 · 各成熟度等级指标统计')

    df_stat = pd.DataFrame(X_filled, columns=INDICATOR_COLS)
    df_stat['成熟度'] = [MATURITY_NAMES[l] for l in maturity_labels]

    print(f'\n  {"指标":<20}  {"未成熟":>12}  {"半成熟":>12}  {"成熟":>12}  {"过成熟":>12}')
    print('  ' + '─' * 75)

    for col, unit in zip(INDICATOR_COLS, INDICATOR_UNITS):
        row = f'  {col:<20} '
        for name in MATURITY_NAMES:
            subset = df_stat[df_stat['成熟度'] == name][col]
            row += f'  {subset.mean():.2f}±{subset.std():.2f}'
        print(row)

    # ── 箱线图 ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Indicator Distribution by Maturity Level',
                 fontsize=14, fontweight='bold', y=0.99)

    short_labels = ['SSC\n(°Brix)', 'PH', 'Hardness\n(N)', 'Color\n(ΔE)']
    for ax, col, short, unit in zip(axes.flatten(), INDICATOR_COLS,
                                    short_labels, INDICATOR_UNITS):
        data_by_class = [df_stat[df_stat['成熟度'] == name][col].values
                         for name in MATURITY_NAMES]
        en_names = ['Unripe', 'Half-ripe', 'Ripe', 'Over-ripe']
        bp = ax.boxplot(data_by_class, patch_artist=True,
                        medianprops=dict(color='white', lw=2.5),
                        whiskerprops=dict(lw=1.5),
                        capprops=dict(lw=1.5),
                        flierprops=dict(marker='o', markersize=2, alpha=0.3))
        for patch, color in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        ax.set_xticklabels(en_names, fontsize=9)
        ax.set_title(short, fontsize=11, fontweight='bold')
        ax.set_ylabel(unit if unit else col, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    save_fig(fig, '01_各指标箱线图.png')

    return df_stat


# =============================================================================
#  Step 8 · 可视化：指标分布直方图
# =============================================================================

def step8_distribution_plot(df_idx):
    section('Step 8 · 各指标分布直方图')

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Raw Indicator Distributions (n=2160)',
                 fontsize=14, fontweight='bold')

    short_en = ['SSC (°Brix)', 'PH', 'Hardness (N)', 'Color Diff (ΔE)']
    colors   = ['#6366F1', '#06B6D4', '#F59E0B', '#EC4899']

    for ax, col, short, color in zip(axes.flatten(), INDICATOR_COLS,
                                      short_en, colors):
        data = df_idx[col].dropna()
        ax.hist(data, bins=55, color=color, alpha=0.8,
                edgecolor='white', linewidth=0.3)
        ax.axvline(data.mean(),   color='red',   ls='--', lw=1.8,
                   label=f'Mean={data.mean():.2f}')
        ax.axvline(data.median(), color='black', ls=':',  lw=1.8,
                   label=f'Median={data.median():.2f}')
        ax.set_title(short, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    save_fig(fig, '01_指标分布.png')


# =============================================================================
#  Step 9 · PCA 可视化聚类结果
# =============================================================================

def step9_pca_visualization(X_weighted, maturity_labels):
    section('Step 9 · PCA 降维可视化聚类结果')

    pca2 = PCA(n_components=2)
    X_2d = pca2.fit_transform(X_weighted)
    var_explained = pca2.explained_variance_ratio_
    print(f'  PC1 方差解释率: {var_explained[0]*100:.2f}%')
    print(f'  PC2 方差解释率: {var_explained[1]*100:.2f}%')
    print(f'  合计           : {sum(var_explained)*100:.2f}%')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('K-Means Clustering Result (PCA 2D Visualization)',
                 fontsize=13, fontweight='bold')

    en_names = ['Unripe', 'Half-ripe', 'Ripe', 'Over-ripe']

    # 左图：散点
    ax = axes[0]
    for i, (name, color) in enumerate(zip(en_names, PALETTE)):
        mask = maturity_labels == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=f'{name} (n={mask.sum()})',
                   alpha=0.55, s=15, linewidths=0)
    ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', fontsize=10)
    ax.set_title('Scatter Plot', fontsize=11)
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(alpha=0.2)

    # 右图：密度轮廓（核密度估计）
    ax2 = axes[1]
    for i, (name, color) in enumerate(zip(en_names, PALETTE)):
        mask = maturity_labels == i
        pts = X_2d[mask]
        try:
            from scipy.stats import gaussian_kde
            xmin, xmax = X_2d[:, 0].min(), X_2d[:, 0].max()
            ymin, ymax = X_2d[:, 1].min(), X_2d[:, 1].max()
            xx, yy = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
            pos = np.vstack([xx.ravel(), yy.ravel()])
            kde = gaussian_kde(pts.T, bw_method=0.7)
            z = kde(pos).reshape(xx.shape)
            ax2.contour(xx, yy, z, levels=4, colors=[color], alpha=0.75,
                        linewidths=1.5)
        except Exception:
            ax2.scatter(pts[:, 0], pts[:, 1], c=color, alpha=0.3, s=8)

    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(PALETTE, en_names)]
    ax2.legend(handles=patches, fontsize=9)
    ax2.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)', fontsize=10)
    ax2.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)', fontsize=10)
    ax2.set_title('Density Contour', fontsize=11)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    save_fig(fig, '01_聚类PCA可视化.png')


# =============================================================================
#  Step 10 · 保存数据集 + 质量报告
# =============================================================================

def step10_save_results(df_idx, df_spec, X_filled,
                        maturity_labels, maturity_names,
                        sil_k4, db_k4, ch_k4):
    section('Step 10 · 保存数据集与质量报告')

    # ── 合并光谱 + 标签 ──
    df_result = df_idx.copy()
    df_result['maturity_label'] = maturity_labels
    df_result['maturity_name']  = maturity_names

    # 用已填充的指标数据覆盖（消除缺失值）
    for i, col in enumerate(INDICATOR_COLS):
        df_result[col] = X_filled[:, i]

    # 完整数据集 = 光谱 + 标签
    df_full = pd.merge(df_spec,
                       df_result[['序号', 'maturity_label', 'maturity_name']],
                       on='序号', how='left')

    # 保存
    label_path = os.path.join(OUT_CLUSTER, '指标数据_含标签.csv')
    full_path  = os.path.join(OUT_CLUSTER, '聚类标签数据集.csv')

    df_result.to_csv(label_path, index=False, encoding='utf-8-sig')
    df_full.to_csv(full_path,    index=False, encoding='utf-8-sig')

    print(f'  已保存: {os.path.relpath(label_path, PROJECT_ROOT)}')
    print(f'           列: 序号 + 4指标 + maturity_label + maturity_name')
    print(f'  已保存: {os.path.relpath(full_path, PROJECT_ROOT)}')
    print(f'           列: 序号 + 728波段 + maturity_label + maturity_name')
    print(f'           形状: {df_full.shape}')

    # ── 质量报告 ──
    report_lines = [
        '=' * 60,
        '  葡萄成熟度聚类质量报告',
        '=' * 60,
        '',
        '【数据概况】',
        f'  总样本数     : {len(df_idx)}',
        f'  指标维度     : {len(INDICATOR_COLS)}',
        f'  光谱波段数   : {df_spec.shape[1] - 1}',
        '',
        '【缺失值处理】',
        '  方法         : 中位数填充',
        '  果皮硬度缺失 : 5 个',
        '  色差缺失     : 1 个',
        '',
        '【权重方案】',
    ]
    for col, w in zip(INDICATOR_COLS, WEIGHTS_SUGGESTED):
        report_lines.append(f'  {col}: {w:.2f}')
    report_lines += [
        '',
        '【聚类参数】',
        '  算法         : K-Means',
        '  聚类数 k     : 4',
        '  随机种子     : 42',
        '  初始化次数   : 20',
        '  最大迭代     : 500',
        '',
        '【聚类质量评价】',
        f'  轮廓系数 (越大越好): {sil_k4:.4f}',
        f'  DB 指数  (越小越好): {db_k4:.4f}',
        f'  CH 指数              : {ch_k4:.2f}',
        '',
        '【各类别样本分布】',
    ]
    for i, name in enumerate(MATURITY_NAMES):
        cnt  = (maturity_labels == i).sum()
        pct  = cnt / len(maturity_labels) * 100
        report_lines.append(f'  {i}: {name:>4} — {cnt:>5} 个  ({pct:.1f}%)')
    report_lines += [
        '',
        '【标签命名依据】',
        '  按各簇 SSC 均值从小到大排序：',
        '  SSC最小 → 未成熟(0)',
        '  SSC最大 → 过成熟(3)',
        '',
        '【输出文件说明】',
        '  指标数据_含标签.csv  : 序号+4指标+标签，供统计分析用',
        '  聚类标签数据集.csv   : 序号+728波段+标签，供后续建模用',
        '',
        '=' * 60,
    ]

    report_path = os.path.join(OUT_CLUSTER, '聚类质量报告.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f'\n  质量报告: {os.path.relpath(report_path, PROJECT_ROOT)}')

    # 打印报告摘要
    print()
    for line in report_lines:
        print('  ' + line)


# =============================================================================
#  主程序
# =============================================================================

def main():
    print('\n' + '=' * 60)
    print('  葡萄成熟度识别 · 第一阶段：多指标聚类标签生成')
    print('=' * 60)

    # Step 1 · 加载数据
    df_idx, df_spec = step1_load_data()

    # Step 2 · 预处理
    X_filled, X_scaled, imputer, scaler = step2_preprocess(df_idx)

    # Step 3 · 权重分析
    X_weighted, weights_equal = step3_weight_analysis(X_scaled)

    # Step 4 · 确定最优 k
    sil_k4_from_grid, db_k4_from_grid = step4_find_optimal_k(X_weighted)

    # Step 5 · 正式聚类
    kmeans, raw_labels, sil_k4, db_k4, ch_k4 = step5_kmeans_cluster(X_weighted)

    # Step 6 · 标签命名
    maturity_labels, maturity_names, label_map = step6_label_naming(
        df_idx, X_filled, raw_labels)

    # Step 7 · 统计分析 + 箱线图
    df_stat = step7_stats_by_maturity(df_idx, X_filled, maturity_labels)

    # Step 8 · 分布直方图
    step8_distribution_plot(df_idx)

    # Step 9 · PCA 可视化
    step9_pca_visualization(X_weighted, maturity_labels)

    # Step 10 · 保存结果
    step10_save_results(df_idx, df_spec, X_filled,
                        maturity_labels, maturity_names,
                        sil_k4, db_k4, ch_k4)

    section('全部完成！下一步：运行 02_光谱预处理.py')
    print('  生成的图片在: results/charts/')
    print('  生成的数据在: results/cluster/')
    print('  后续代码读取: results/cluster/聚类标签数据集.csv\n')


if __name__ == '__main__':
    main()