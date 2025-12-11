"""
الملف الرئيسي لتشغيل تحليل K-Means على بيانات Iris
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# استيراد الوحدات
from data.iris_dataset import load_iris_dataset, get_dataset_info
from algorithms.kmeans import run_kmeans
from algorithms.silhouette import calculate_silhouette_scores
from algorithms.cluster_analysis import analyze_all_clusters, get_cluster_comparison
from visualization.plots import (
    setup_plot_style,
    plot_clusters_scatter,
    plot_silhouette_scores,
    plot_cluster_statistics,
    plot_convergence_history,
    plot_dataset_overview,
    display_dataset_table
)
from utils.constants import CLUSTER_NAMES, CLUSTER_COLORS, WINDOW_SIZE


def display_plots_in_tabs(plots):
    """
    عرض الرسومات في واجهة مع علامات تبويب
    
    Args:
        plots: قائمة من tuples (figure, title)
    """
    root = tk.Tk()
    root.title("K-Means Analysis Dashboard")
    root.geometry(WINDOW_SIZE)
    
    # إنشاء علامات التبويب
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    for fig, title in plots:
        # إنشاء إطار لكل تبويب
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        
        # إضافة الرسم البياني
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # تشغيل التطبيق
    root.mainloop()


def print_header():
    """طباعة رأس التطبيق"""
    print("\n" + "=" * 60)
    print(" " * 15 + "K-MEANS IRIS ANALYSIS")
    print(" " * 12 + "Complete Implementation in Python")
    print("=" * 60 + "\n")


def print_section(title: str):
    """طباعة عنوان قسم"""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


def print_dataset_info(data: np.ndarray):
    """طباعة معلومات البيانات"""
    info = get_dataset_info(data)
    
    print_section("📊 DATASET INFORMATION")
    print(f"  Total Samples:   {info['n_samples']}")
    print(f"  Total Features: {info['n_features']}")
    print(f"\n  Features:")
    for i, name in enumerate(info['feature_names']):
        print(f"    {i+1}. {name}")
        print(f"       Range: [{info['min_values'][i]:.1f} - {info['max_values'][i]:.1f}]")
        print(f"       Mean:   {info['mean_values'][i]:.2f}")


def print_kmeans_results(result, data: np.ndarray):
    """طباعة نتائج K-Means"""
    print_section("🔄 K-MEANS RESULTS")
    
    print(f"  Iterations:   {result.n_iterations}")
    print(f"  Converged:   {'✓ Yes' if result.converged else '✗ No'}")
    print(f"\n  Cluster Distribution:")
    
    for c in range(3):
        count = np.sum(result.clusters == c)
        percentage = (count / len(data)) * 100
        print(f"    {CLUSTER_NAMES[c]}: {count} points ({percentage:.1f}%)")
    
    print(f"\n  Cluster Centers:")
    for c in range(3):
        center = result.centers[c]
        print(f"    {CLUSTER_NAMES[c]}: [{', '.join(f'{v:.2f}' for v in center)}]")


def print_silhouette_results(silhouette_result):
    """طباعة نتائج Silhouette"""
    print_section("📈 SILHOUETTE ANALYSIS")
    
    print(f"  Average Score: {silhouette_result.average_score:.4f}")
    print(f"\n  Score Interpretation:")
    print(f"    • Close to  1: Well clustered")
    print(f"    • Close to  0: On cluster border")
    print(f"    • Negative:     Possibly misclassified")
    
    print(f"\n  Scores by Cluster:")
    for c, score in enumerate(silhouette_result.cluster_scores):
        print(f"    {CLUSTER_NAMES[c]}: {score:.4f}")


def print_cluster_analysis(analyses):
    """طباعة تحليل المجموعات"""
    print_section("📋 DETAILED CLUSTER ANALYSIS")
    
    for analysis in analyses:
        print(f"\n  ┌─ {CLUSTER_NAMES[analysis.cluster_id]} ─────────────────────────────┐")
        print(f"  │  Size: {analysis.size} points ({analysis.percentage:.1f}%)")
        print(f"  │  Center: [{', '.join(f'{v:.2f}' for v in analysis.center)}]")
        print(f"  │")
        print(f"  │  Statistics:")
        
        for stat in analysis.statistics:
            print(f"  │    {stat.name}:")
            print(f"  │      Mean: {stat.mean:.2f}, Std: {stat.std:.2f}")
            print(f"  │      Range: [{stat.min:.2f} - {stat.max:.2f}]")
        
        print(f"  └{'─' * 45}┘")


def run_analysis():
    """تشغيل التحليل الكامل"""
    # إعداد
    setup_plot_style()
    print_header()
    
    # تحميل البيانات
    data = load_iris_dataset()
    print_dataset_info(data)
    
    # تشغيل K-Means
    print("\n  Running K-Means algorithm...")
    kmeans_result = run_kmeans(data, k=3)
    print_kmeans_results(kmeans_result, data)
    
    # حساب Silhouette
    print("\n  Calculating Silhouette scores...")
    silhouette_result = calculate_silhouette_scores(
        data, 
        kmeans_result.clusters, 
        n_clusters=3
    )
    print_silhouette_results(silhouette_result)
    
    # تحليل المجموعات
    analyses = analyze_all_clusters(
        data,
        kmeans_result.clusters,
        kmeans_result.centers
    )
    print_cluster_analysis(analyses)
    
    # عرض البيانات في جدول
    display_dataset_table(data, kmeans_result.clusters, n_rows=15)
    
    # إنشاء الرسومات
    print_section("📊 GENERATING VISUALIZATIONS")
    
    # 1. عرض البيانات
    fig1 = plot_dataset_overview(data, kmeans_result.clusters)
    fig1.savefig('output_dataset_overview.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved:  output_dataset_overview.png")
    
    # 2. رسم المجموعات
    fig2 = plot_clusters_scatter(
        data, 
        kmeans_result.clusters, 
        kmeans_result.centers
    )
    fig2.savefig('output_clusters.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: output_clusters.png")
    
    # 3. رسم Silhouette
    fig3 = plot_silhouette_scores(silhouette_result, kmeans_result.clusters)
    fig3.savefig('output_silhouette.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: output_silhouette.png")
    
    # 4. إحصائيات المجموعات
    fig4 = plot_cluster_statistics(analyses)
    fig4.savefig('output_statistics.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: output_statistics.png")
    
    # 5. تاريخ التقارب
    fig5 = plot_convergence_history(kmeans_result.history)
    fig5.savefig('output_convergence.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved:  output_convergence.png")
    
    # عرض جميع الرسومات في واجهة واحدة
    display_plots_in_tabs([
        (fig1, "Dataset Overview"),
        (fig2, "Cluster Scatter"),
        (fig3, "Silhouette Scores"),
        (fig4, "Cluster Statistics"),
        (fig5, "Convergence History")
    ])
    
    print("\n" + "=" * 60)
    print(" " * 18 + "ANALYSIS COMPLETE!")
    print("=" * 60 + "\n")
    
    return {
        'data': data,
        'kmeans_result': kmeans_result,
        'silhouette_result': silhouette_result,
        'analyses': analyses
    }


# نقطة الدخول الرئيسية
if __name__ == "__main__": 
    results = run_analysis()