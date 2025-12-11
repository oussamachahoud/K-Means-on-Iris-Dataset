"""
خوارزمية K-Means للتجميع
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class KMeansResult:
    """نتيجة خوارزمية K-Means"""
    clusters: np.ndarray
    centers: np.ndarray
    history: List[Dict[str, Any]]
    n_iterations: int
    converged: bool


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


def calculate_distances_matrix(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    حساب مصفوفة المسافات بين جميع النقاط والمراكز
    
    Args:
        points: مصفوفة النقاط
        centers: مصفوفة المراكز
        
    Returns:
        np.ndarray: مصفوفة المسافات
    """
    n_points = points.shape[0]
    n_centers = centers.shape[0]
    distances = np.zeros((n_points, n_centers))
    
    for i, point in enumerate(points):
        for j, center in enumerate(centers):
            distances[i, j] = calculate_euclidean_distance(point, center)
    
    return distances


def initialize_random_centers(points: np.ndarray, k: int) -> np.ndarray:
    """
    تهيئة المراكز العشوائية
    
    Args:
        points: مصفوفة النقاط
        k:  عدد المجموعات
        
    Returns:
        np.ndarray: المراكز الأولية
    """
    n_samples = points.shape[0]
    random_indices = np.random.choice(n_samples, size=k, replace=False)
    return points[random_indices].copy()


def assign_points_to_clusters(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    تعيين كل نقطة للمركز الأقرب
    
    Args: 
        points: مصفوفة النقاط
        centers: مصفوفة المراكز
        
    Returns:
        np.ndarray: تعيينات المجموعات
    """
    distances = calculate_distances_matrix(points, centers)
    return np.argmin(distances, axis=1)


def calculate_new_centers(
    points: np.ndarray, 
    assignments: np.ndarray, 
    k: int,
    current_centers: np.ndarray
) -> np.ndarray:
    """
    حساب المراكز الجديدة
    
    Args:
        points: مصفوفة النقاط
        assignments:  تعيينات المجموعات
        k: عدد المجموعات
        current_centers: المراكز الحالية
        
    Returns:
        np.ndarray: المراكز الجديدة
    """
    n_features = points.shape[1]
    new_centers = np.zeros((k, n_features))
    
    for cluster_id in range(k):
        cluster_mask = assignments == cluster_id
        cluster_points = points[cluster_mask]
        
        if len(cluster_points) > 0:
            new_centers[cluster_id] = cluster_points.mean(axis=0)
        else:
            # إذا كانت المجموعة فارغة، احتفظ بالمركز القديم
            new_centers[cluster_id] = current_centers[cluster_id]
    
    return new_centers


def check_convergence(
    old_centers: np.ndarray, 
    new_centers: np.ndarray, 
    tolerance: float = 0.0001
) -> bool:
    """
    التحقق من تقارب الخوارزمية
    
    Args:
        old_centers: المراكز القديمة
        new_centers: المراكز الجديدة
        tolerance: حد التقارب
        
    Returns:
        bool: هل تقاربت الخوارزمية
    """
    return np.all(np.abs(old_centers - new_centers) <= tolerance)


def run_kmeans(
    points: np.ndarray, 
    k: int = 3, 
    max_iterations: int = 100,
    tolerance: float = 0.0001
) -> KMeansResult:
    """
    تشغيل خوارزمية K-Means
    
    Args:
        points: البيانات
        k: عدد المجموعات
        max_iterations: الحد الأقصى للتكرارات
        tolerance: حد التقارب
        
    Returns:
        KMeansResult: نتائج الخوارزمية
    """
    # تهيئة المراكز
    centers = initialize_random_centers(points, k)
    history = []
    converged = False
    
    for iteration in range(max_iterations):
        # تعيين النقاط للمجموعات
        assignments = assign_points_to_clusters(points, centers)
        
        # حفظ الحالة الحالية
        history.append({
            'iteration': iteration + 1,
            'centers': centers.copy(),
            'clusters': assignments.copy()
        })
        
        # حساب المراكز الجديدة
        new_centers = calculate_new_centers(points, assignments, k, centers)
        
        # التحقق من التقارب
        if check_convergence(centers, new_centers, tolerance):
            converged = True
            centers = new_centers
            break
        
        centers = new_centers
    
    # التعيين النهائي
    final_assignments = assign_points_to_clusters(points, centers)
    
    return KMeansResult(
        clusters=final_assignments,
        centers=centers,
        history=history,
        n_iterations=len(history),
        converged=converged
    )