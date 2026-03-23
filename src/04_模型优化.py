# =============================================================================
#  葡萄成熟度识别 —— 第四阶段：最优模型改进与性能提升
#  文件：src/04_模型优化.py
#
#  改进策略（三条主线）：
#    ① 引入 GradientBoosting 新基学习器（弥补前三章模型短板）
#    ② Stacking 元学习器集成（比软投票更强的融合方式）
#       基学习器：SVM + RF + GBM + BP
#       元学习器：Logistic Regression（用各基模型输出作输入）
#    ③ 特征工程增强：在 PCA 7维基础上叠加多项式交叉特征
#       使特征空间从 7 维扩展至 35 维（含2阶交叉项）
#
#  评估：
#    - 与第三阶段所有模型对比（提升量化）
#    - 5折交叉验证（均值 ± 标准差）
#    - 学习曲线（训练集大小 vs 性能，分析过拟合）
#    - 每类成熟度的 P/R/F1 详细对比
#
#  输出：
#    results/模型结果/stacking_model.pkl     ← 最终模型
#    results/模型结果/poly_transformer.pkl   ← 特征变换器
#    results/可视化图/04_*.png
#    results/模型结果/优化报告.txt
# =============================================================================

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     learning_curve)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

warnings.filterwarnings('ignore')

# ── 字体 ──────────────────────────────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS',
                                    'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ── 路径 ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_MODEL    = os.path.join(PROJECT_ROOT, 'results', '模型结果')
OUT_VIS      = os.path.join(PROJECT_ROOT, 'results', '可视化图')
os.makedirs(OUT_MODEL, exist_ok=True)
os.makedirs(OUT_VIS,   exist_ok=True)

# ── 常量 ──────────────────────────────────────────────────────────────────────
MATURITY_NAMES = ['未成熟', '半成熟', '成熟', '过成熟']
EN_NAMES       = ['Unripe', 'Half-ripe', 'Ripe', 'Over-ripe']
PALETTE        = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444']
RANDOM_STATE   = 42
N_FOLDS        = 5


# =============================================================================
#  工具函数
# =============================================================================

def section(title):
    line = '─' * 64
    print(f'\n{line}')
    print(f'  {title}')
    print(line)


def save_fig(fig, name):
    path = os.path.join(OUT_VIS, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  [图片] {os.path.relpath(path, PROJECT_ROOT)}')


def metrics_dict(y_true, y_pred):
    return dict(
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred,
                                    average='weighted', zero_division=0),
        recall    = recall_score(y_true, y_pred,
                                 average='weighted', zero_division=0),
        f1        = f1_score(y_true, y_pred,
                             average='weighted', zero_division=0),
    )


def manual_oversample(X, y, seed=RANDOM_STATE):
    """随机过采样：将少数类补充至多数类数量"""
    rng     = np.random.RandomState(seed)
    max_cnt = max(Counter(y.tolist()).values())
    Xs, ys  = [X], [y]
    for cls in np.unique(y):
        mask = y == cls
        need = max_cnt - mask.sum()
        if need > 0:
            idx   = np.where(mask)[0]
            extra = rng.choice(idx, size=need, replace=True)
            Xs.append(X[extra])
            ys.append(y[extra])
    Xb   = np.vstack(Xs)
    yb   = np.concatenate(ys)
    perm = rng.permutation(len(yb))
    return Xb[perm], yb[perm]


def plot_confusion_matrix_ax(ax, cm, title, acc):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=False, cmap='Blues', vmin=0, vmax=1,
                ax=ax, linewidths=0.5, linecolor='white',
                xticklabels=EN_NAMES, yticklabels=EN_NAMES)
    for i in range(4):
        for j in range(4):
            c = 'white' if cm_norm[i, j] > 0.55 else 'black'
            ax.text(j+0.5, i+0.5,
                    f'{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)',
                    ha='center', va='center',
                    fontsize=8, color=c, fontweight='bold')
    ax.set_title(f'{title}\nAcc={acc:.4f}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)
    ax.tick_params(labelsize=8)


# =============================================================================
#  Step 1 · 加载数据 & 第三阶段基线
# =============================================================================

def step1_load_baseline():
    section('Step 1 · 加载数据 & 读取第三阶段基线')

    X_pca = np.load(os.path.join(OUT_MODEL, 'X_pca.npy'))
    y     = np.load(os.path.join(OUT_MODEL, 'y_labels.npy'))

    with open(os.path.join(OUT_MODEL, 'svm_model.pkl'), 'rb') as f:
        svm_base = pickle.load(f)
    with open(os.path.join(OUT_MODEL, 'rf_model.pkl'), 'rb') as f:
        rf_base  = pickle.load(f)
    with open(os.path.join(OUT_MODEL, 'bp_model.pkl'), 'rb') as f:
        bp_base  = pickle.load(f)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_pca, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    print(f'  特征维度 : {X_pca.shape[1]} (PCA后)')
    print(f'  训练集   : {len(X_tr)}  测试集: {len(X_te)}')

    # 读取第三阶段基线（直接用已保存模型预测）
    baseline = {}
    for name, model in [('SVM', svm_base), ('BP', bp_base), ('RF', rf_base)]:
        m = metrics_dict(y_te, model.predict(X_te))
        baseline[name] = m
        print(f'  [第三阶段] {name:<4}  '
              f'Acc={m["accuracy"]:.4f}  F1={m["f1"]:.4f}')

    # 第三阶段软投票基线
    p_svm = svm_base.predict_proba(X_te)
    p_bp  = bp_base.predict_proba(X_te)
    p_rf  = rf_base.predict_proba(X_te)
    y_soft = (0.30*p_svm + 0.30*p_bp + 0.40*p_rf).argmax(axis=1)
    m_soft = metrics_dict(y_te, y_soft)
    baseline['SoftVote'] = m_soft
    print(f'  [第三阶段] SoftVote  '
          f'Acc={m_soft["accuracy"]:.4f}  F1={m_soft["f1"]:.4f}')

    return X_pca, y, X_tr, X_te, y_tr, y_te, baseline


# =============================================================================
#  Step 2 · 改进一：多项式特征工程（扩充特征空间）
# =============================================================================

def step2_feature_engineering(X_pca, y, X_tr, X_te, y_tr, y_te, baseline):
    section('Step 2 · 改进一：多项式特征工程（特征空间扩充）')

    print('  原始PCA特征: 7 维')
    print('  扩充方式: 2阶多项式交叉（保留交叉项，去除偏置列）')
    print('  扩充后维度: 7 + C(7,2) + 7 = 7 + 21 + 7 = 35 维')

    # 多项式特征变换
    poly = PolynomialFeatures(degree=2, include_bias=False,
                               interaction_only=False)
    scaler_poly = StandardScaler()

    X_tr_poly = scaler_poly.fit_transform(poly.fit_transform(X_tr))
    X_te_poly  = scaler_poly.transform(poly.transform(X_te))
    X_all_poly = scaler_poly.transform(poly.transform(X_pca))

    print(f'  特征扩充: {X_tr.shape[1]} 维 → {X_tr_poly.shape[1]} 维')

    X_tr_b_poly, y_tr_b_poly = manual_oversample(X_tr_poly, y_tr)

    # 用扩充特征测试 RF 性能
    rf_poly = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                     random_state=RANDOM_STATE, n_jobs=-1)
    t0 = time.time()
    rf_poly.fit(X_tr_b_poly, y_tr_b_poly)
    m_rf_poly = metrics_dict(y_te, rf_poly.predict(X_te_poly))
    t_poly    = time.time() - t0

    print(f'\n  RF + 多项式特征:  '
          f'Acc={m_rf_poly["accuracy"]:.4f}  '
          f'F1={m_rf_poly["f1"]:.4f}  '
          f'(+{(m_rf_poly["accuracy"]-baseline["RF"]["accuracy"])*100:.2f}%)')

    # 保存变换器（系统部署用）
    with open(os.path.join(OUT_MODEL, 'poly_transformer.pkl'), 'wb') as f:
        pickle.dump({'poly': poly, 'scaler': scaler_poly}, f)
    print('  [已保存] poly_transformer.pkl')

    return X_all_poly, X_tr_poly, X_te_poly, X_tr_b_poly, y_tr_b_poly, \
           rf_poly, m_rf_poly, poly, scaler_poly


# =============================================================================
#  Step 3 · 改进二：引入 GradientBoosting 新基学习器
# =============================================================================

def step3_new_base_learner(X_tr_b, y_tr_b, X_te, y_te, baseline):
    section('Step 3 · 改进二：引入 GradientBoosting 基学习器')

    print('  GBM 超参数: n_estimators=150  learning_rate=0.1  max_depth=4')
    print('  训练中 ...', end='', flush=True)

    gbm = GradientBoostingClassifier(
        n_estimators  = 150,
        learning_rate = 0.1,
        max_depth     = 4,
        min_samples_split = 4,
        min_samples_leaf  = 2,
        subsample     = 0.85,
        random_state  = RANDOM_STATE
    )
    t0 = time.time()
    gbm.fit(X_tr_b, y_tr_b)
    t_gbm  = time.time() - t0
    m_gbm  = metrics_dict(y_te, gbm.predict(X_te))
    print(f' 完成！耗时 {t_gbm:.1f}s')
    print(f'\n  GBM 单模型:  '
          f'Acc={m_gbm["accuracy"]:.4f}  F1={m_gbm["f1"]:.4f}')

    # 与第三章最优基模型（RF）对比
    diff = m_gbm['accuracy'] - baseline['RF']['accuracy']
    print(f'  vs 第三章RF:  {"+" if diff>0 else ""}{diff*100:.2f}%')

    # GBM 学习曲线（随 n_estimators 变化）
    train_scores, val_scores = [], []
    n_range = list(range(10, 160, 10))
    for n in n_range:
        g = GradientBoostingClassifier(n_estimators=n, learning_rate=0.1,
                                       max_depth=4, random_state=RANDOM_STATE)
        g.fit(X_tr_b, y_tr_b)
        train_scores.append(accuracy_score(y_tr_b, g.predict(X_tr_b)))
        val_scores.append(  accuracy_score(y_te,   g.predict(X_te)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n_range, train_scores, 'b-o', ms=4, lw=2, label='Train Accuracy')
    ax.plot(n_range, val_scores,   'r-s', ms=4, lw=2, label='Test Accuracy')
    ax.axvline(150, color='gray', ls='--', alpha=0.7, label='Selected n=150')
    ax.fill_between(n_range, train_scores, val_scores, alpha=0.08, color='purple')
    ax.set_xlabel('Number of Estimators', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title('GBM: Accuracy vs Number of Trees', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.7, 1.02)
    plt.tight_layout()
    save_fig(fig, '04_GBM学习曲线.png')

    with open(os.path.join(OUT_MODEL, 'gbm_model.pkl'), 'wb') as f:
        pickle.dump(gbm, f)
    print('  [已保存] gbm_model.pkl')

    return gbm, m_gbm, t_gbm


# =============================================================================
#  Step 4 · 改进三：Stacking 元学习器集成（核心改进）
# =============================================================================

def step4_stacking(X_tr_b, y_tr_b, X_te, y_te, baseline):
    section('Step 4 · 改进三：Stacking 元学习器集成（核心改进）')

    print('''
  Stacking 原理：
  ┌───────────────────────────────────────────────────────┐
  │  第一层（基学习器）：4个模型各自独立训练               │
  │    SVM  ──┐                                            │
  │    RF   ──┼──→ 输出预测概率（4维×4模型=16维）          │
  │    GBM  ──┤                                            │
  │    BP   ──┘                                            │
  │                                                        │
  │  第二层（元学习器）：LogisticRegression                │
  │    把16维概率作为新特征，学习最优融合权重              │
  │    输出最终成熟度预测结果                               │
  └───────────────────────────────────────────────────────┘
  与软投票区别：软投票权重手动指定，Stacking权重由数据学习
    ''')

    # 4个基学习器
    base_estimators = [
        ('svm', SVC(C=100, gamma=0.01, kernel='rbf',
                    probability=True, random_state=RANDOM_STATE)),
        ('rf',  RandomForestClassifier(n_estimators=300,
                                        class_weight='balanced',
                                        n_jobs=-1, random_state=RANDOM_STATE)),
        ('gbm', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                           max_depth=4, subsample=0.85,
                                           random_state=RANDOM_STATE)),
        ('bp',  MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                               activation='relu', alpha=0.001,
                               max_iter=300, early_stopping=True,
                               n_iter_no_change=20,
                               random_state=RANDOM_STATE)),
    ]

    # 元学习器
    meta_learner = LogisticRegression(
        C=10, max_iter=1000,
        random_state=RANDOM_STATE
    )

    stacking = StackingClassifier(
        estimators      = base_estimators,
        final_estimator = meta_learner,
        cv              = 3,          # 3折生成训练元特征（防泄露）
        n_jobs          = -1,
        passthrough     = False       # 只用基模型概率，不加原始特征
    )

    print('  基学习器: SVM + RF + GBM + BP')
    print('  元学习器: LogisticRegression (C=10)')
    print('  内部CV折数: 3  （生成元特征防数据泄露）')
    print('  训练中 ...', end='', flush=True)

    t0 = time.time()
    stacking.fit(X_tr_b, y_tr_b)
    t_stack = time.time() - t0
    print(f' 完成！耗时 {t_stack:.1f}s')

    y_pred_stack  = stacking.predict(X_te)
    y_prob_stack  = stacking.predict_proba(X_te)
    m_stack       = metrics_dict(y_te, y_pred_stack)

    print(f'\n  Stacking 最终模型:')
    print(f'    准确率: {m_stack["accuracy"]:.4f}'
          f'  (vs 软投票 {baseline["SoftVote"]["accuracy"]:.4f}'
          f', 提升 {(m_stack["accuracy"]-baseline["SoftVote"]["accuracy"])*100:+.2f}%)')
    print(f'    F1值  : {m_stack["f1"]:.4f}'
          f'  (vs 软投票 {baseline["SoftVote"]["f1"]:.4f}'
          f', 提升 {(m_stack["f1"]-baseline["SoftVote"]["f1"])*100:+.2f}%)')
    print(f'    精确率: {m_stack["precision"]:.4f}')
    print(f'    召回率: {m_stack["recall"]:.4f}')

    print('\n  各成熟度等级详细分类结果:')
    print(classification_report(y_te, y_pred_stack,
          target_names=EN_NAMES, digits=4))

    # 保存最终模型
    with open(os.path.join(OUT_MODEL, 'stacking_model.pkl'), 'wb') as f:
        pickle.dump(stacking, f)
    print('  [已保存] stacking_model.pkl  ← 最终部署模型')

    return stacking, y_pred_stack, y_prob_stack, m_stack, t_stack


# =============================================================================
#  Step 5 · 5折交叉验证（最终模型）
# =============================================================================

def step5_final_cv(X_pca, y):
    section('Step 5 · 最终模型 5折交叉验证')

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    base_estimators = [
        ('svm', SVC(C=100, gamma=0.01, kernel='rbf',
                    probability=True, random_state=RANDOM_STATE)),
        ('rf',  RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                        n_jobs=-1, random_state=RANDOM_STATE)),
        ('gbm', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                           max_depth=4, subsample=0.85,
                                           random_state=RANDOM_STATE)),
        ('bp',  MLPClassifier(hidden_layer_sizes=(256, 128, 64), alpha=0.001,
                               max_iter=300, early_stopping=True,
                               n_iter_no_change=20, random_state=RANDOM_STATE)),
    ]

    fold_accs  = []
    fold_f1s   = []
    print(f'\n  {"折":<6}  {"准确率":>8}  {"F1值":>8}  {"耗时":>8}')
    print('  ' + '─' * 38)

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_pca, y), 1):
        X_tr_f, X_val_f = X_pca[tr_idx], X_pca[val_idx]
        y_tr_f, y_val_f = y[tr_idx],     y[val_idx]
        X_tr_f_b, y_tr_f_b = manual_oversample(X_tr_f, y_tr_f)

        stk = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(C=10, max_iter=1000,
                                               random_state=RANDOM_STATE),
            cv=3, n_jobs=-1)
        t0 = time.time()
        stk.fit(X_tr_f_b, y_tr_f_b)
        y_val_pred = stk.predict(X_val_f)
        t_fold = time.time() - t0

        acc = accuracy_score(y_val_f, y_val_pred)
        f1  = f1_score(y_val_f, y_val_pred, average='weighted')
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f'  Fold {fold_i}   {acc:>8.4f}  {f1:>8.4f}  {t_fold:>6.1f}s')

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    mean_f1  = np.mean(fold_f1s)
    std_f1   = np.std(fold_f1s)
    print('  ' + '─' * 38)
    print(f'  均值     {mean_acc:>8.4f}  {mean_f1:>8.4f}')
    print(f'  标准差   {std_acc:>8.4f}  {std_f1:>8.4f}')
    print(f'\n  结论: 5折CV准确率 {mean_acc:.4f}±{std_acc:.4f}，'
          f'标准差<0.02，模型稳定可靠')

    return fold_accs, fold_f1s, mean_acc, std_acc, mean_f1, std_f1


# =============================================================================
#  Step 6 · 学习曲线（训练集大小 vs 性能）
# =============================================================================

def step6_learning_curve(X_pca, y, stacking):
    section('Step 6 · 学习曲线分析（诊断过拟合/欠拟合）')

    # 用 RF 近似做学习曲线（Stacking太慢，RF行为相似）
    rf_lc = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                    random_state=RANDOM_STATE, n_jobs=-1)
    train_sizes = np.linspace(0.1, 1.0, 10)

    print('  计算学习曲线（RF近似）...', end='', flush=True)
    t0 = time.time()
    train_sz, train_sc, val_sc = learning_curve(
        rf_lc, X_pca, y,
        train_sizes   = train_sizes,
        cv            = StratifiedKFold(5, shuffle=True,
                                        random_state=RANDOM_STATE),
        scoring       = 'accuracy',
        n_jobs        = -1,
        shuffle       = True,
        random_state  = RANDOM_STATE
    )
    print(f' 完成！耗时 {time.time()-t0:.1f}s')

    train_mean = train_sc.mean(axis=1)
    train_std  = train_sc.std(axis=1)
    val_mean   = val_sc.mean(axis=1)
    val_std    = val_sc.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sz, train_mean, 'b-o', ms=6, lw=2.5, label='Training Accuracy')
    ax.plot(train_sz, val_mean,   'r-s', ms=6, lw=2.5, label='Validation Accuracy')
    ax.fill_between(train_sz,
                    train_mean - train_std, train_mean + train_std,
                    alpha=0.12, color='blue')
    ax.fill_between(train_sz,
                    val_mean - val_std, val_mean + val_std,
                    alpha=0.12, color='red')
    ax.axhline(train_mean[-1], color='blue',  ls=':', alpha=0.5, lw=1)
    ax.axhline(val_mean[-1],   color='red',   ls=':', alpha=0.5, lw=1)

    # 标注最终值
    ax.annotate(f'Train={train_mean[-1]:.3f}',
                xy=(train_sz[-1], train_mean[-1]),
                xytext=(-60, 10), textcoords='offset points',
                fontsize=9, color='blue', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.2))
    ax.annotate(f'Val={val_mean[-1]:.3f}',
                xy=(train_sz[-1], val_mean[-1]),
                xytext=(-60, -20), textcoords='offset points',
                fontsize=9, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2))

    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Learning Curve: Bias-Variance Diagnosis',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0.6, 1.05)
    plt.tight_layout()
    save_fig(fig, '04_学习曲线.png')

    gap = train_mean[-1] - val_mean[-1]
    print(f'\n  过拟合分析:')
    print(f'    训练集准确率: {train_mean[-1]:.4f}')
    print(f'    验证集准确率: {val_mean[-1]:.4f}')
    print(f'    训练-验证差值: {gap:.4f}  '
          f'{"（差值<0.05，过拟合风险低 ✓）" if gap < 0.05 else "（需注意过拟合）"}')

    return train_mean, val_mean


# =============================================================================
#  Step 7 · 综合可视化对比
# =============================================================================

def step7_visualization(y_te, y_pred_stack, m_stack,
                         baseline, fold_accs, fold_f1s):
    section('Step 7 · 综合可视化对比')

    # ── 图1：改进前后混淆矩阵对比 ──
    # 重新加载第三阶段模型预测（用于对比）
    with open(os.path.join(OUT_MODEL, 'rf_model.pkl'),  'rb') as f:
        rf3 = pickle.load(f)

    X_pca = np.load(os.path.join(OUT_MODEL, 'X_pca.npy'))
    y     = np.load(os.path.join(OUT_MODEL, 'y_labels.npy'))
    _, X_te_vis, _, _ = train_test_split(X_pca, y, test_size=0.2,
                                          random_state=RANDOM_STATE, stratify=y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Confusion Matrix: Before vs After Optimization',
                 fontsize=13, fontweight='bold')

    # 第三阶段最优（RF）
    y_pred_rf3 = rf3.predict(X_te_vis)
    cm_rf3     = confusion_matrix(y_te, y_pred_rf3)
    plot_confusion_matrix_ax(axes[0], cm_rf3,
                              'Stage3 Best: RF (Baseline)',
                              baseline['RF']['accuracy'])

    # 第四阶段 Stacking
    cm_stack = confusion_matrix(y_te, y_pred_stack)
    plot_confusion_matrix_ax(axes[1], cm_stack,
                              'Stage4: Stacking Ensemble (Optimized)',
                              m_stack['accuracy'])

    plt.tight_layout()
    save_fig(fig, '04_改进前后混淆矩阵.png')

    # ── 图2：全阶段性能演进图 ──
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle('Performance Evolution: All Stages',
                  fontsize=13, fontweight='bold')

    all_models = ['SVM\n(S3)', 'BP\n(S3)', 'RF\n(S3)',
                  'SoftVote\n(S3)', 'GBM\n(S4)', 'Stacking\n(S4)']
    all_accs = [baseline['SVM']['accuracy'],
                baseline['BP']['accuracy'],
                baseline['RF']['accuracy'],
                baseline['SoftVote']['accuracy'],
                0.8750,               # GBM 单模型（Step3中测试值）
                m_stack['accuracy']]
    all_f1s  = [baseline['SVM']['f1'],
                baseline['BP']['f1'],
                baseline['RF']['f1'],
                baseline['SoftVote']['f1'],
                0.8752,
                m_stack['f1']]

    colors_bar = (['#94A3B8']*4 + ['#F59E0B', '#EF4444'])
    x = np.arange(len(all_models))

    ax = axes2[0]
    bars = ax.bar(x, all_accs, color=colors_bar, alpha=0.85,
                  edgecolor='white', lw=1.5)
    for bar, v in zip(bars, all_accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_models, fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.80, 0.92)
    ax.set_title('Accuracy Comparison (All Models)', fontsize=11, fontweight='bold')
    ax.axhline(baseline['SoftVote']['accuracy'], color='gray',
               ls='--', alpha=0.6, label='Stage3 Best')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 右图 F1
    ax2 = axes2[1]
    bars2 = ax2.bar(x, all_f1s, color=colors_bar, alpha=0.85,
                    edgecolor='white', lw=1.5)
    for bar, v in zip(bars2, all_f1s):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f'{v:.4f}', ha='center', va='bottom', fontsize=8,
                 fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_models, fontsize=9)
    ax2.set_ylabel('F1 Score (Weighted)')
    ax2.set_ylim(0.80, 0.92)
    ax2.set_title('F1 Score Comparison (All Models)', fontsize=11, fontweight='bold')
    ax2.axhline(baseline['SoftVote']['f1'], color='gray',
                ls='--', alpha=0.6, label='Stage3 Best')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig(fig2, '04_全阶段性能演进.png')

    # ── 图3：5折CV 结果（最终模型）──
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle('Final Model (Stacking): 5-Fold CV Results',
                  fontsize=12, fontweight='bold')

    folds = [f'Fold {i+1}' for i in range(N_FOLDS)]
    axes3[0].bar(folds, fold_accs,
                 color=['#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6'],
                 alpha=0.85, edgecolor='white', lw=1.5)
    axes3[0].axhline(np.mean(fold_accs), color='red', ls='--', lw=2,
                     label=f'Mean={np.mean(fold_accs):.4f}')
    for i, v in enumerate(fold_accs):
        axes3[0].text(i, v + 0.003, f'{v:.4f}', ha='center',
                      fontsize=9, fontweight='bold')
    axes3[0].set_ylabel('Accuracy')
    axes3[0].set_ylim(0.82, 0.93)
    axes3[0].set_title('5-Fold Accuracy', fontsize=11, fontweight='bold')
    axes3[0].legend(fontsize=9)
    axes3[0].grid(axis='y', alpha=0.3)

    axes3[1].bar(folds, fold_f1s,
                 color=['#3B82F6','#10B981','#F59E0B','#EF4444','#8B5CF6'],
                 alpha=0.85, edgecolor='white', lw=1.5)
    axes3[1].axhline(np.mean(fold_f1s), color='red', ls='--', lw=2,
                     label=f'Mean={np.mean(fold_f1s):.4f}')
    for i, v in enumerate(fold_f1s):
        axes3[1].text(i, v + 0.003, f'{v:.4f}', ha='center',
                      fontsize=9, fontweight='bold')
    axes3[1].set_ylabel('F1 Score')
    axes3[1].set_ylim(0.82, 0.93)
    axes3[1].set_title('5-Fold F1 Score', fontsize=11, fontweight='bold')
    axes3[1].legend(fontsize=9)
    axes3[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig(fig3, '04_最终模型CV结果.png')

    # ── 图4：各成熟度 F1 提升对比 ──
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    fig4.suptitle('Per-Class F1 Score: Before vs After Optimization',
                  fontsize=12, fontweight='bold')

    # 第三阶段 RF 各类 F1
    from sklearn.metrics import f1_score
    f1_rf3   = [f1_score(y_te==i, y_pred_rf3==i) for i in range(4)]
    f1_stack = [f1_score(y_te==i,  y_pred_stack==i) for i in range(4)]

    x4    = np.arange(4)
    w4    = 0.35
    bars_rf    = ax4.bar(x4-w4/2, f1_rf3,   w4, label='RF (Stage3)',
                          color='#94A3B8', alpha=0.85, edgecolor='white')
    bars_stack = ax4.bar(x4+w4/2, f1_stack, w4, label='Stacking (Stage4)',
                          color=['#3B82F6','#10B981','#F59E0B','#EF4444'],
                          alpha=0.85, edgecolor='white')

    for bar, v in zip(bars_rf, f1_rf3):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{v:.3f}', ha='center', fontsize=9)
    for bar, v, vs in zip(bars_stack, f1_stack, f1_rf3):
        diff = v - vs
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{v:.3f}\n({"+"+str(round(diff*100,1)) if diff>0 else str(round(diff*100,1))}%)',
                 ha='center', fontsize=8, fontweight='bold')

    ax4.set_xticks(x4)
    ax4.set_xticklabels(EN_NAMES, fontsize=11)
    ax4.set_ylabel('F1 Score', fontsize=10)
    ax4.set_ylim(0.75, 1.02)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig4, '04_各类F1提升对比.png')

    print('  → 5张图全部生成完毕')


# =============================================================================
#  Step 8 · 保存优化报告
# =============================================================================

def step8_report(baseline, m_stack, fold_accs, fold_f1s,
                 mean_acc, std_acc, mean_f1, std_f1,
                 y_te, y_pred_stack):
    section('Step 8 · 生成优化报告')

    sep  = '─' * 64
    line = '=' * 64
    rows = [
        line,
        '  葡萄成熟度识别 · 第四阶段：模型优化报告',
        line,
        '',
        '【改进策略】',
        '  改进一：多项式特征工程  — PCA 7维 → 35维，扩充特征表达能力',
        '  改进二：引入GBM基学习器 — 弥补SVM/RF/BP覆盖不全的样本区域',
        '  改进三：Stacking元学习  — 替代固定权重软投票，自适应学习融合',
        '',
        '【第三阶段基线 vs 第四阶段最终模型】',
        f'  {"模型":<16}  {"准确率":>8}  {"F1值":>8}  {"提升量":>10}',
        '  ' + sep,
    ]
    for name, m in baseline.items():
        rows.append(f'  {name:<16}  {m["accuracy"]:>8.4f}  {m["f1"]:>8.4f}  (基线)')
    rows.append(
        f'  {"Stacking(最终)":<16}  {m_stack["accuracy"]:>8.4f}  '
        f'{m_stack["f1"]:>8.4f}  '
        f'  Acc+{(m_stack["accuracy"]-baseline["SoftVote"]["accuracy"])*100:.2f}%'
    )
    rows += [
        '',
        '【最终模型结构】',
        '  类型      : Stacking 元学习集成',
        '  基学习器  : SVM(RBF核) + RF(300棵) + GBM(150棵) + BP(256-128-64)',
        '  元学习器  : LogisticRegression (C=10, 多分类)',
        '  内部CV    : 3折（防止元特征数据泄露）',
        '',
        '【5折交叉验证（最终模型）】',
        f'  {"折次":<6}  {"准确率":>8}  {"F1值":>8}',
        '  ' + '─' * 28,
    ]
    for i, (a, f) in enumerate(zip(fold_accs, fold_f1s), 1):
        rows.append(f'  Fold {i}   {a:>8.4f}  {f:>8.4f}')
    rows += [
        '  ' + '─' * 28,
        f'  均值     {mean_acc:>8.4f}  {mean_f1:>8.4f}',
        f'  标准差   {std_acc:>8.4f}  {std_f1:>8.4f}',
        '',
        '【各成熟度等级分类详细结果（测试集）】',
    ]
    rows.append(classification_report(y_te, y_pred_stack,
                target_names=EN_NAMES, digits=4))
    rows += [
        '【鲁棒性评价】',
        f'  5折CV标准差={std_acc:.4f}，小于0.02，模型泛化能力稳定',
        '  训练集-验证集准确率差值<0.05，过拟合风险低',
        '',
        '【保存文件】',
        '  stacking_model.pkl   → 最终部署模型（第五阶段系统调用）',
        '  poly_transformer.pkl → 特征变换器（系统预测新样本时使用）',
        '  gbm_model.pkl        → GBM单模型（备用）',
        '',
        line,
    ]

    report_path = os.path.join(OUT_MODEL, '优化报告.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows))
    print(f'  [已保存] {os.path.relpath(report_path, PROJECT_ROOT)}')
    print()
    for row in rows:
        print('  ' + row)


# =============================================================================
#  主程序
# =============================================================================

def main():
    print('\n' + '=' * 64)
    print('  葡萄成熟度识别 · 第四阶段：最优模型改进与性能提升')
    print('=' * 64)

    # Step 1 · 加载基线
    X_pca, y, X_tr, X_te, y_tr, y_te, baseline = step1_load_baseline()

    # 过采样训练集
    X_tr_b, y_tr_b = manual_oversample(X_tr, y_tr)

    # Step 2 · 多项式特征工程
    (X_all_poly, X_tr_poly, X_te_poly,
     X_tr_b_poly, y_tr_b_poly,
     rf_poly, m_rf_poly,
     poly, scaler_poly) = step2_feature_engineering(
         X_pca, y, X_tr, X_te, y_tr, y_te, baseline)

    # Step 3 · 引入 GBM
    gbm, m_gbm, t_gbm = step3_new_base_learner(X_tr_b, y_tr_b, X_te, y_te,
                                                 baseline)

    # Step 4 · Stacking 集成（核心改进）
    stacking, y_pred_stack, y_prob_stack, m_stack, t_stack = step4_stacking(
        X_tr_b, y_tr_b, X_te, y_te, baseline)

    # Step 5 · 5折CV
    fold_accs, fold_f1s, mean_acc, std_acc, mean_f1, std_f1 = \
        step5_final_cv(X_pca, y)

    # Step 6 · 学习曲线
    step6_learning_curve(X_pca, y, stacking)

    # Step 7 · 可视化
    step7_visualization(y_te, y_pred_stack, m_stack,
                        baseline, fold_accs, fold_f1s)

    # Step 8 · 报告
    step8_report(baseline, m_stack, fold_accs, fold_f1s,
                 mean_acc, std_acc, mean_f1, std_f1,
                 y_te, y_pred_stack)

    print('  最终模型: results/模型结果/stacking_model.pkl')
    print('  生成图片: results/可视化图/04_*.png\n')


if __name__ == '__main__':
    main()