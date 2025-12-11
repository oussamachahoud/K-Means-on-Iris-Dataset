"""
الرسومات البيانية للتحليل
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from algorithms.kmeans import KMeansResult
from algorithms.silhouette import SilhouetteResult
from algorithms.cluster_analysis import ClusterAnalysis
from utils.constants import CLUSTER_COLORS, CLUSTER_NAMES, FEATURE_NAMES, DEFAULT_FIGURE_SIZE


def setup_plot_style():
    """إعداد نمط الرسومات"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = DEFAULT_FIGURE_SIZE
    plt.rcParams['font.size'] = 10


def plot_clusters_scatter(
    points: np.ndarray,
    assignments: np.ndarray,
    centers: np.ndarray,
    feature_x: int = 2,
    feature_y: int = 3,
    title: str = "K-Means Clustering Results"
) -> plt.Figure:
    """
    رسم النقاط والمجموعات
    
    Args: 
        points: البيانات
        assignments: تعيينات المجموعات
        centers: المراكز
        feature_x: رقم الخاصية لمحور X
        feature_y: رقم الخاصية لمحور Y
        title: عنوان الرسم
        
    Returns:
        plt.Figure: الشكل
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    
    # رسم النقاط لكل مجموعة
    for cluster_id in range(len(centers)):
        cluster_mask = assignments == cluster_id
        cluster_points = points[cluster_mask]
        
        ax.scatter(
            cluster_points[:, feature_x],
            cluster_points[:, feature_y],
            c=CLUSTER_COLORS[cluster_id],
            label=CLUSTER_NAMES[cluster_id],
            alpha=0.7,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    # رسم المراكز
    ax.scatter(
        centers[:, feature_x],
        centers[:, feature_y],
        c='red',
        marker='X',
        s=200,
        edgecolors='black',
        linewidth=2,
        label='Centers',
        zorder=5
    )
    
    ax.set_xlabel(FEATURE_NAMES[feature_x], fontsize=12)
    ax.set_ylabel(FEATURE_NAMES[feature_y], fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    return fig


def plot_silhouette_scores(
    silhouette_result: SilhouetteResult,
    assignments: np.ndarray,
    title: str = "Silhouette Analysis"
) -> plt.Figure:
    """
    رسم درجات Silhouette
    
    Args:
        silhouette_result: نتائج Silhouette
        assignments: تعيينات المجموعات
        title: عنوان الرسم
        
    Returns:
        plt.Figure: الشكل
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE)
    
    # ترتيب الدرجات حسب المجموعة ثم القيمة
    sorted_indices = np.lexsort((
        -silhouette_result.scores, 
        assignments
    ))
    
    scores_sorted = silhouette_result.scores[sorted_indices]
    assignments_sorted = assignments[sorted_indices]
    
    # رسم الأعمدة
    colors = [CLUSTER_COLORS[c] for c in assignments_sorted]
    bars = ax.bar(range(len(scores_sorted)), scores_sorted, color=colors, width=1.0)
    
    # خط المتوسط
    ax.axhline(
        y=silhouette_result.average_score, 
        color='red', 
        linestyle='--', 
        linewidth=2,
        label=f'Average: {silhouette_result.average_score:.3f}'
    )
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-0.2, 1.0)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_cluster_statistics(
    analyses: List[ClusterAnalysis],
    title: str = "Cluster Statistics Comparison"
) -> plt.Figure:
    """
    رسم إحصائيات المجموعات
    
    Args:
        analyses: قائمة تحليلات المجموعات
        title: عنوان الرسم
        
    Returns:
        plt.Figure: الشكل
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    x = np.arange(len(analyses))
    width = 0.6
    
    for feature_idx, ax in enumerate(axes):
        means = [a.statistics[feature_idx].mean for a in analyses]
        stds = [a.statistics[feature_idx].std for a in analyses]
        
        bars = ax.bar(
            x, means, width,
            color=CLUSTER_COLORS[: len(analyses)],
            yerr=stds,
            capsize=5,
            edgecolor='black',
            linewidth=1
        )
        
        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(FEATURE_NAMES[feature_idx], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(CLUSTER_NAMES[: len(analyses)])
        
        # إضافة القيم فوق الأعمدة
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f'{mean:.2f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_convergence_history(
    history: List[dict],
    title: str = "K-Means Convergence"
) -> plt.Figure:
    """
    رسم تاريخ التقارب
    
    Args: 
        history: سجل التكرارات
        title: عنوان الرسم
        
    Returns:
        plt.Figure: الشكل
    """
    fig, axes = plt.subplots(1, 2, figsize=DEFAULT_FIGURE_SIZE)
    
    iterations = [h['iteration'] for h in history]
    
    # توزيع النقاط في كل تكرار
    ax1 = axes[0]
    cluster_sizes = []
    
    for h in history:
        sizes = [np.sum(h['clusters'] == c) for c in range(3)]
        cluster_sizes.append(sizes)
    
    cluster_sizes = np.array(cluster_sizes)
    
    for c in range(3):
        ax1.plot(
            iterations, 
            cluster_sizes[:, c], 
            color=CLUSTER_COLORS[c],
            marker='o',
            linewidth=2,
            label=CLUSTER_NAMES[c]
        )
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Number of Points', fontsize=12)
    ax1.set_title('Cluster Size Evolution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # حركة المراكز
    ax2 = axes[1]
    
    for c in range(3):
        center_movements = []
        for i in range(1, len(history)):
            old_center = history[i-1]['centers'][c]
            new_center = history[i]['centers'][c]
            movement = np.sqrt(np.sum((new_center - old_center) ** 2))
            center_movements.append(movement)
        
        if center_movements:
            ax2.plot(
                iterations[1:], 
                center_movements, 
                color=CLUSTER_COLORS[c],
                marker='s',
                linewidth=2,
                label=CLUSTER_NAMES[c]
            )
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Center Movement', fontsize=12)
    ax2.set_title('Center Movement per Iteration', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_dataset_overview(
    points: np.ndarray,
    assignments: Optional[np.ndarray] = None,
    title: str = "Iris Dataset Overview"
) -> plt.Figure:
    """
    عرض نظرة عامة على البيانات
    
    Args:
        points: البيانات
        assignments: تعيينات المجموعات (اختياري)
        title: عنوان الرسم
        
    Returns: 
        plt.Figure: الشكل
    """
    fig, axes = plt.subplots(2, 3, figsize=DEFAULT_FIGURE_SIZE)
    axes = axes.flatten()
    
    feature_pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]
    
    for idx, (f1, f2) in enumerate(feature_pairs):
        ax = axes[idx]
        
        if assignments is not None:
            for c in range(3):
                mask = assignments == c
                ax.scatter(
                    points[mask, f1],
                    points[mask, f2],
                    c=CLUSTER_COLORS[c],
                    label=CLUSTER_NAMES[c],
                    alpha=0.7,
                    s=30
                )
        else:
            ax.scatter(
                points[:, f1],
                points[:, f2],
                c='#3b82f6',
                alpha=0.7,
                s=30
            )
        
        ax.set_xlabel(FEATURE_NAMES[f1], fontsize=10)
        ax.set_ylabel(FEATURE_NAMES[f2], fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # إضافة المفتاح
    if assignments is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    return fig


def display_dataset_table(
    points: np.ndarray,
    assignments: np.ndarray,
    n_rows: int = 20
) -> None:
    """
    عرض البيانات في جدول
    
    Args:
        points: البيانات
        assignments: تعيينات المجموعات
        n_rows: عدد الصفوف للعرض
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "IRIS DATASET")
    print("=" * 80)
    print(f"{'#': >4} | {'Sepal L':>8} | {'Sepal W': >8} | {'Petal L':>8} | {'Petal W':>8} | {'Cluster': >10}")
    print("-" * 80)
    
    for i in range(min(n_rows, len(points))):
        cluster_name = CLUSTER_NAMES[assignments[i]]
        print(f"{i+1:>4} | {points[i,0]:>8.1f} | {points[i,1]:>8.1f} | {points[i,2]: >8.1f} | {points[i,3]:>8.1f} | {cluster_name: >10}")
    
    if len(points) > n_rows:
        print(f"... ({len(points) - n_rows} more rows)")
    
    print("=" * 80)