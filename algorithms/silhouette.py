"""
خوارزمية Silhouette لتقييم جودة التجميع
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class SilhouetteResult:
    """نتيجة تحليل Silhouette"""
    scores: np.ndarray
    average_score: float
    cluster_scores: List[float]


def calculate_euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    حساب المسافة الإقليدية بين نقطتين
    
    Args: 
        point1: النقطة الأولى
        point2: النقطة الثانية
        
    Returns: 
        float: المسافة الإقليدية
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


def calculate_intra_cluster_distance(
    point: np.ndarray, 
    same_cluster_points: np.ndarray
) -> float:
    """
    حساب متوسط المسافة داخل المجموعة (a)
    
    Args:
        point: النقطة المراد حسابها
        same_cluster_points: نقاط نفس المجموعة
        
    Returns:
        float: متوسط المسافة
    """
    if len(same_cluster_points) == 0:
        return 0.0
    
    distances = [
        calculate_euclidean_distance(point, other_point) 
        for other_point in same_cluster_points
    ]
    
    return np.mean(distances)


def calculate_nearest_cluster_distance(
    point: np.ndarray,
    points: np.ndarray,
    assignments: np.ndarray,
    current_cluster: int,
    n_clusters: int
) -> float:
    """
    حساب أقل متوسط مسافة للمجموعات الأخرى (b)
    
    Args:
        point: النقطة المراد حسابها
        points: جميع النقاط
        assignments: تعيينات المجموعات
        current_cluster: المجموعة الحالية
        n_clusters: إجمالي المجموعات
        
    Returns:
        float: أقل متوسط مسافة
    """
    min_average_distance = np.inf
    
    for cluster_id in range(n_clusters):
        if cluster_id == current_cluster:
            continue
        
        cluster_mask = assignments == cluster_id
        other_cluster_points = points[cluster_mask]
        
        if len(other_cluster_points) == 0:
            continue
        
        distances = [
            calculate_euclidean_distance(point, other_point)
            for other_point in other_cluster_points
        ]
        
        average_distance = np.mean(distances)
        min_average_distance = min(min_average_distance, average_distance)
    
    return min_average_distance


def calculate_point_silhouette_score(
    intra_distance: float, 
    inter_distance: float
) -> float:
    """
    حساب درجة Silhouette لنقطة واحدة
    
    Args:
        intra_distance:  المسافة داخل المجموعة (a)
        inter_distance:  المسافة بين المجموعات (b)
        
    Returns:
        float:  درجة Silhouette
    """
    max_distance = max(intra_distance, inter_distance)
    
    if max_distance == 0:
        return 0.0
    
    return (inter_distance - intra_distance) / max_distance


def calculate_silhouette_scores(
    points: np.ndarray, 
    assignments: np.ndarray,
    n_clusters: int = 3
) -> SilhouetteResult:
    """
    حساب درجات Silhouette لجميع النقاط
    
    Args:
        points: البيانات
        assignments: تعيينات المجموعات
        n_clusters: عدد المجموعات
        
    Returns:
        SilhouetteResult: نتائج التحليل
    """
    n_points = len(points)
    scores = np.zeros(n_points)
    
    for i, point in enumerate(points):
        current_cluster = assignments[i]
        
        # الحصول على نقاط نفس المجموعة (بدون النقطة الحالية)
        same_cluster_mask = (assignments == current_cluster)
        same_cluster_mask[i] = False
        same_cluster_points = points[same_cluster_mask]
        
        # حساب المسافة داخل المجموعة
        intra_distance = calculate_intra_cluster_distance(point, same_cluster_points)
        
        # حساب أقل مسافة للمجموعات الأخرى
        inter_distance = calculate_nearest_cluster_distance(
            point, points, assignments, current_cluster, n_clusters
        )
        
        # حساب درجة Silhouette
        scores[i] = calculate_point_silhouette_score(intra_distance, inter_distance)
    
    # حساب متوسط الدرجات لكل مجموعة
    cluster_scores = []
    for cluster_id in range(n_clusters):
        cluster_mask = assignments == cluster_id
        if np.any(cluster_mask):
            cluster_scores.append(np.mean(scores[cluster_mask]))
        else:
            cluster_scores.append(0.0)
    
    return SilhouetteResult(
        scores=scores,
        average_score=np.mean(scores),
        cluster_scores=cluster_scores
    )