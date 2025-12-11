"""
تحليل المجموعات وإحصائياتها
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class FeatureStatistics:
    """إحصائيات خاصية واحدة"""
    name: str
    mean: float
    min: float
    max: float
    std: float


@dataclass
class ClusterAnalysis:
    """تحليل مجموعة واحدة"""
    cluster_id: int
    size: int
    percentage: float
    center: np.ndarray
    statistics: List[FeatureStatistics]


FEATURE_NAMES = [
    'Sepal Length',
    'Sepal Width',
    'Petal Length',
    'Petal Width'
]


def calculate_feature_statistics(
    values: np.ndarray, 
    feature_name: str
) -> FeatureStatistics:
    """
    حساب الإحصائيات لخاصية واحدة
    
    Args:
        values: قيم الخاصية
        feature_name: اسم الخاصية
        
    Returns:
        FeatureStatistics: الإحصائيات
    """
    if len(values) == 0:
        return FeatureStatistics(
            name=feature_name,
            mean=0.0,
            min=0.0,
            max=0.0,
            std=0.0
        )
    
    return FeatureStatistics(
        name=feature_name,
        mean=float(np.mean(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        std=float(np.std(values))
    )


def analyze_single_cluster(
    points: np.ndarray,
    assignments: np.ndarray,
    centers: np.ndarray,
    cluster_id: int,
    total_points: int
) -> ClusterAnalysis:
    """
    تحليل مجموعة واحدة
    
    Args:
        points:  جميع النقاط
        assignments: تعيينات المجموعات
        centers:  مراكز المجموعات
        cluster_id: رقم المجموعة
        total_points: إجمالي النقاط
        
    Returns:
        ClusterAnalysis: تحليل المجموعة
    """
    # الحصول على نقاط المجموعة
    cluster_mask = assignments == cluster_id
    cluster_points = points[cluster_mask]
    
    # حساب الإحصائيات لكل خاصية
    statistics = []
    for feature_idx, feature_name in enumerate(FEATURE_NAMES):
        feature_values = cluster_points[:, feature_idx]
        stats = calculate_feature_statistics(feature_values, feature_name)
        statistics.append(stats)
    
    return ClusterAnalysis(
        cluster_id=cluster_id,
        size=len(cluster_points),
        percentage=(len(cluster_points) / total_points) * 100,
        center=centers[cluster_id],
        statistics=statistics
    )


def analyze_all_clusters(
    points: np.ndarray,
    assignments: np.ndarray,
    centers: np.ndarray,
    n_clusters: int = 3
) -> List[ClusterAnalysis]:
    """
    تحليل جميع المجموعات
    
    Args:
        points: البيانات
        assignments: تعيينات المجموعات
        centers: المراكز
        n_clusters: عدد المجموعات
        
    Returns: 
        List[ClusterAnalysis]:  تحليل كل مجموعة
    """
    total_points = len(points)
    
    return [
        analyze_single_cluster(
            points, assignments, centers, cluster_id, total_points
        )
        for cluster_id in range(n_clusters)
    ]


def get_cluster_comparison(analyses: List[ClusterAnalysis]) -> Dict[str, Any]:
    """
    مقارنة بين المجموعات
    
    Args:
        analyses: قائمة تحليلات المجموعات
        
    Returns:
        Dict:  مقارنة بين المجموعات
    """
    comparison = {}
    
    for feature_idx, feature_name in enumerate(FEATURE_NAMES):
        comparison[feature_name] = {
            'means': [
                analysis.statistics[feature_idx].mean 
                for analysis in analyses
            ],
            'stds': [
                analysis.statistics[feature_idx].std 
                for analysis in analyses
            ]
        }
    
    return comparison