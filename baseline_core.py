"""Original reconstruction kernels; cohort selection is provided by the portable CLI."""

from __future__ import annotations

import csv

import os

import glob

import re

import shutil

import time

import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed

from dataclasses import dataclass, replace, astuple

import numpy as np

import open3d as o3d

from scipy.spatial import cKDTree

RNG_SEED = 20240101

DENSITY_PERCENTILE = 22.5

PERCENTILE_SWEEP = [5, 10, 15, 20, 22.5, 25, 30, 35, 40, 50]

BUDGET = "reduced"

MAX_DISTANCE_THRESHOLD = 0.005

MAX_VARIANCE_THRESHOLD = 0.0025

MIN_COVERAGE_RATIO = 0.9

MAX_ITERATIONS = 4

MIN_TARGET_POINTS = 64

MAX_TARGET_POINTS = 4000

POINT_STEP = 400

HOLE_SCORE_PENALTY = 5.0

FIXED_TARGET_POINTS = 3000

FIXED_DEPTH = 9

FIXED_MAX_NN = 30

FIXED_MLS_SPACING_MULT = 2.5

FIXED_SMOOTHING = 1

BPA_RADIUS_MULTIPLIERS = [1.0, 2.0, 4.0]

BPA_MAX_NN = 30

UNSUPPORTED_DISTANCE_MULT = 2.0

VALIDATE_N_LEAVES = 20

TIME_SINGLE_REPEATS = 3

BUDGETS = {
    # Poisson 质量不随采样点数单调变化，因此这里使用少量相对候选，而不是二分查找。
    "full":    dict(target_multipliers=(0.60, 0.80, 1.00, 1.20, 1.40),
                    max_iterations=None, max_candidates=3),
    "reduced": dict(target_multipliers=(0.80, 1.00, 1.20),
                    max_iterations=2, max_candidates=2),
    "minimal": dict(target_multipliers=(1.00,),
                    max_iterations=0, max_candidates=1),
}

_INNER_WORKERS = -1

@dataclass(frozen=True)
class Params:
    target_points: int
    mls_search_radius: float
    max_nn: int
    depth: int
    smoothing_iterations: int
    interpolation_k_neighbors: int = 8
    density_compensation: bool = True

@dataclass
class Metrics:
    has_holes: bool
    hole_count: int
    avg_distance_error: float
    distance_variance: float
    coverage_ratio: float
    overall_score: float

def load_points(path: str) -> np.ndarray:
    try:
        data = np.loadtxt(path, dtype=np.float64)
    except ValueError:
        data = np.loadtxt(path, dtype=np.float64, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError("point cloud needs at least 3 columns")
    points = data[:, :3]
    return points[np.all(np.isfinite(points), axis=1)]

def obj_surface_area(path: str) -> float:
    """计算 OBJ 已有三角形/多边形的表面积；不跨孔洞补面。"""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                indices = []
                for token in line.split()[1:]:
                    raw = int(token.split("/", 1)[0])
                    indices.append(raw - 1 if raw > 0 else len(vertices) + raw)
                if len(indices) >= 3:
                    faces.append(indices)
    if not vertices or not faces:
        raise ValueError(f"OBJ has no usable vertices/faces: {path}")
    verts = np.asarray(vertices, dtype=np.float64)
    area = 0.0
    for face in faces:
        a = verts[face[0]]
        for i in range(1, len(face) - 1):
            area += 0.5 * float(np.linalg.norm(np.cross(verts[face[i]] - a,
                                                       verts[face[i + 1]] - a)))
    return area

def median_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 1e-6
    dist, _ = cKDTree(points).query(points, k=2, workers=_INNER_WORKERS)
    nearest = dist[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    return float(np.median(nearest)) if len(nearest) else 1e-6

def to_cloud(points: np.ndarray, max_nn: int) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(
        knn=min(max_nn, max(4, len(points) - 1))))
    cloud.orient_normals_consistent_tangent_plane(k=min(max_nn, max(4, len(points) - 1)))
    return cloud

def detect_sparse(points: np.ndarray, k_neighbors: int) -> np.ndarray:
    n = len(points)
    if n < k_neighbors + 1:
        return np.zeros(n, dtype=bool)
    dist, _ = cKDTree(points).query(points, k=min(k_neighbors + 1, n), workers=_INNER_WORKERS)
    if dist.ndim == 1:
        dist = dist[:, None]
    density = dist[:, 1:].mean(axis=1)
    finite = density[np.isfinite(density)]
    if len(finite) == 0:
        return np.zeros(n, dtype=bool)
    return density > finite.mean() + 1.5 * finite.std()

def density_compensate(points: np.ndarray, k_neighbors: int, seed: int) -> np.ndarray:
    sparse_mask = detect_sparse(points, k_neighbors)
    sparse_idx = np.where(sparse_mask)[0]
    if len(sparse_idx) == 0:
        return points

    ratio = len(sparse_idx) / len(points)
    factor = max(1, min(3, int(ratio * 10)) if ratio > 0.1 else 2)

    rng = np.random.default_rng(seed)
    _, neighbor_idx = cKDTree(points).query(
        points[sparse_idx], k=min(k_neighbors * 2 + 1, len(points)), workers=_INNER_WORKERS)
    if neighbor_idx.ndim == 1:
        neighbor_idx = neighbor_idx[:, None]

    generated = []
    for row, idx in enumerate(sparse_idx):
        cand = neighbor_idx[row][1:]
        cand = cand[~sparse_mask[cand]][:k_neighbors]
        if len(cand) < 3:
            continue
        dense = points[cand]
        sigma = float(np.linalg.norm(points[idx] - dense.mean(axis=0))) * 0.1
        n_base = min(3, len(dense))
        for _ in range(factor):
            pick = rng.choice(len(dense), n_base, replace=False)
            weights = rng.dirichlet(np.ones(n_base))
            generated.append(dense[pick].T @ weights + rng.normal(0.0, sigma, 3))

    return np.vstack([points, np.asarray(generated)]) if generated else points

def farthest_point_sample(points: np.ndarray, k: int) -> np.ndarray:
    if k >= len(points):
        return points
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if hasattr(cloud, "farthest_point_down_sample"):
        return np.asarray(cloud.farthest_point_down_sample(k).points)

    selected = np.empty(k, dtype=np.int64)
    selected[0] = 0
    distance = np.linalg.norm(points - points[0], axis=1)
    for i in range(1, k):
        selected[i] = int(np.argmax(distance))
        distance = np.minimum(distance, np.linalg.norm(points - points[selected[i]], axis=1))
    return points[selected]

def mls_smooth(points: np.ndarray, radius: float) -> np.ndarray:
    """一阶 MLS：局部加权平面拟合后投影。"""
    if radius <= 0 or len(points) < 8:
        return points
    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_point(points, r=radius, workers=_INNER_WORKERS)
    smoothed = points.copy()
    for i, neighbors in enumerate(neighbor_lists):
        if len(neighbors) < 4:
            continue
        local = points[neighbors]
        delta = local - points[i]
        weight = np.exp(-np.sum(delta ** 2, axis=1) / (radius ** 2 + 1e-18))
        total = weight.sum()
        if total <= 1e-12:
            continue
        centroid = (local * weight[:, None]).sum(axis=0) / total
        centered = local - centroid
        covariance = (centered * weight[:, None]).T @ centered / total
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        smoothed[i] = points[i] - normal * float(np.dot(points[i] - centroid, normal))
    return smoothed

def largest_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    if len(counts) <= 1:
        return mesh
    result = o3d.geometry.TriangleMesh(mesh)
    result.remove_triangles_by_mask(labels != int(np.argmax(counts)))
    result.remove_unreferenced_vertices()
    return result

def n_components(mesh: o3d.geometry.TriangleMesh) -> int:
    if len(mesh.triangles) == 0:
        return 0
    _, counts, _ = mesh.cluster_connected_triangles()
    return int(len(np.asarray(counts)))

def finalize(mesh: o3d.geometry.TriangleMesh, smoothing: int) -> o3d.geometry.TriangleMesh:
    if smoothing > 0 and len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
    return mesh

def poisson_core(points: np.ndarray, params: Params, seed: int):
    """返回 (未裁剪网格, 每顶点密度, 用于重建的采样点)。失败返回 (None, None, None)。"""
    try:
        working = points
        if params.density_compensation:
            working = density_compensate(working, params.interpolation_k_neighbors, seed)
        sampled = farthest_point_sample(working, params.target_points)
        sampled = mls_smooth(sampled, params.mls_search_radius)

        cloud = to_cloud(sampled, params.max_nn)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            cloud, depth=params.depth, linear_fit=False)
        if len(mesh.triangles) == 0:
            return None, None, None
        return mesh, np.asarray(densities), sampled
    except Exception:
        return None, None, None

def crop_by_density(mesh, densities, percentile: float, smoothing: int):
    """按密度分位裁剪出开放片。必须在任何清理/平滑之前做，否则顶点索引会错位。"""
    if mesh is None or densities is None or len(densities) != len(mesh.vertices):
        return None
    cropped = o3d.geometry.TriangleMesh(mesh)
    cropped.remove_vertices_by_mask(densities < np.percentile(densities, percentile))
    cropped.remove_unreferenced_vertices()
    if len(cropped.triangles) == 0:
        return None
    return finalize(largest_component(cropped), smoothing)

def ball_pivoting_core(points: np.ndarray):
    spacing = median_spacing(points)
    radii = [spacing * m for m in BPA_RADIUS_MULTIPLIERS]
    cloud = to_cloud(points, BPA_MAX_NN)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        cloud, o3d.utility.DoubleVector(radii))
    if len(mesh.triangles) == 0:
        return None, radii
    return finalize(mesh, smoothing=0), radii

def unsupported_area_ratio(mesh, points: np.ndarray, spacing: float) -> float:
    """网格中"没有点云支撑"的面积占比 —— 定量刻画外扩超出点云范围的程度。

    对每个三角形取质心，若其到最近输入点的距离超过 UNSUPPORTED_DISTANCE_MULT x
    中位最近邻间距，则该三角形计入外扩。本文方法按构造应接近 0。
    """
    if mesh is None or len(mesh.triangles) == 0:
        return float("nan")
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    corners = vertices[triangles]
    centroids = corners.mean(axis=1)
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    total = areas.sum()
    if total <= 0:
        return float("nan")
    distance, _ = cKDTree(points).query(centroids, k=1, workers=_INNER_WORKERS)
    outside = distance > UNSUPPORTED_DISTANCE_MULT * spacing
    return float(areas[outside].sum() / total)

def internal_hole_count(mesh) -> int:
    """统计开放叶片曲面的内部孔洞数。

    叶片是开放单层曲面，最大的一个边界环是正常的叶片外轮廓，不能用
    ``is_watertight()`` 判孔洞。其余闭合边界环才是内部孔洞。边界分叉、
    非流形边/点也视为至少一个拓扑缺陷。
    """
    if mesh is None or len(mesh.triangles) == 0:
        return 1

    triangles = np.asarray(mesh.triangles)
    edges = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]],
                               triangles[:, [2, 0]]]), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    invalid_topology = bool(np.any(counts > 2))

    # 开放曲面中的边界顶点正常情况下度数都为 2；分叉或断裂属于拓扑缺陷。
    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        a, b = int(a), int(b)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        invalid_topology = True

    # 边界图的连通分量数即边界环数（合法流形边界中）。
    boundary_loops = 0
    unseen = set(adjacency)
    while unseen:
        boundary_loops += 1
        stack = [unseen.pop()]
        while stack:
            vertex = stack.pop()
            for neighbor in adjacency.get(vertex, ()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)

    holes = max(0, boundary_loops - 1)  # 最大边界环是叶片外轮廓
    if invalid_topology:
        holes = max(1, holes)
    return holes

def has_holes(mesh) -> bool:
    return internal_hole_count(mesh) > 0

def evaluate(reference: np.ndarray, mesh, target_points: int) -> Metrics:
    if mesh is None or len(mesh.triangles) == 0:
        return Metrics(True, 1, float("inf"), float("inf"), 0.0, float("inf"))
    holes = internal_hole_count(mesh)
    try:
        sampled = np.asarray(mesh.sample_points_uniformly(
            number_of_points=max(1000, min(4000, max(target_points, len(reference))))).points)
    except Exception:
        sampled = np.asarray(mesh.vertices)
    if len(sampled) == 0 or len(reference) == 0:
        return Metrics(True, max(1, holes), float("inf"), float("inf"), 0.0, float("inf"))

    d_ref, _ = cKDTree(sampled).query(reference, k=1, workers=_INNER_WORKERS)
    d_smp, _ = cKDTree(reference).query(sampled, k=1, workers=_INNER_WORKERS)
    both = np.concatenate([d_ref, d_smp])
    avg, var = float(both.mean()), float(both.var())
    coverage = float(np.count_nonzero(d_ref < MAX_DISTANCE_THRESHOLD * 2) / len(reference))
    score = (avg / MAX_DISTANCE_THRESHOLD + var / MAX_VARIANCE_THRESHOLD
             + (1.0 - coverage) / (1.0 - MIN_COVERAGE_RATIO)
             + holes * HOLE_SCORE_PENALTY)
    return Metrics(holes > 0, holes, avg, var, coverage, score)

def is_good(metrics: Metrics) -> bool:
    return (not metrics.has_holes
            and metrics.avg_distance_error <= MAX_DISTANCE_THRESHOLD
            and metrics.distance_variance <= MAX_VARIANCE_THRESHOLD
            and metrics.coverage_ratio >= MIN_COVERAGE_RATIO)

def chamfer(mesh, points: np.ndarray, n_sample: int = 4000) -> float:
    """双向 Chamfer 距离：开放网格 vs 原始点云。"""
    if mesh is None or len(mesh.triangles) == 0:
        return float("nan")
    try:
        sampled = np.asarray(mesh.sample_points_uniformly(number_of_points=n_sample).points)
    except Exception:
        sampled = np.asarray(mesh.vertices)
    if len(sampled) == 0:
        return float("nan")
    d1, _ = cKDTree(sampled).query(points, k=1, workers=_INNER_WORKERS)
    d2, _ = cKDTree(points).query(sampled, k=1, workers=_INNER_WORKERS)
    return float(np.concatenate([d1, d2]).mean())

def initial_params(points: np.ndarray) -> Params:
    n = len(points)
    rng = np.random.default_rng(RNG_SEED)
    sample_idx = rng.choice(n, min(500, n), replace=False)
    dist, _ = cKDTree(points).query(points[sample_idx], k=min(6, n), workers=_INNER_WORKERS)
    if dist.ndim == 1:
        dist = dist[:, None]
    avg_distance = float(dist[:, 1:min(4, dist.shape[1])].mean())

    # target_points 必须随实际点数变化。旧公式的内层上限不超过 600，
    # 再与 MIN_TARGET_POINTS=2000 取 max，导致所有叶片恒为 2000。
    if n < 300:
        target, mls, depth, max_nn = n, avg_distance * 3.0, 8, 20
    elif n < 800:
        target, mls, depth, max_nn = int(round(n * 0.9)), avg_distance * 2.5, 10, 30
    else:
        target, mls, depth, max_nn = int(round(n * 0.8)), avg_distance * 2.0, 11, 40
    target = int(np.clip(target, MIN_TARGET_POINTS, MAX_TARGET_POINTS))
    return Params(target, mls, max_nn, depth, smoothing_iterations=1)

def candidate_params(base: Params, metrics: Metrics, budget: dict) -> list[Params]:
    candidates = []
    if metrics.has_holes:
        candidates.append(replace(base,
                                  target_points=min(MAX_TARGET_POINTS, base.target_points + POINT_STEP),
                                  mls_search_radius=base.mls_search_radius * 0.8,
                                  max_nn=min(80, base.max_nn + 15),
                                  depth=min(16, base.depth + 1),
                                  smoothing_iterations=max(0, base.smoothing_iterations - 1),
                                  interpolation_k_neighbors=min(12, base.interpolation_k_neighbors + 2)))
        candidates.append(replace(base,
                                  interpolation_k_neighbors=min(16, base.interpolation_k_neighbors + 4)))
    if metrics.avg_distance_error > MAX_DISTANCE_THRESHOLD:
        candidates.append(replace(base,
                                  target_points=min(MAX_TARGET_POINTS, int(base.target_points * 1.2)),
                                  mls_search_radius=base.mls_search_radius * 0.7,
                                  smoothing_iterations=max(0, base.smoothing_iterations - 1)))
    if metrics.distance_variance > MAX_VARIANCE_THRESHOLD:
        candidates.append(replace(base,
                                  mls_search_radius=base.mls_search_radius * 1.1,
                                  smoothing_iterations=min(4, base.smoothing_iterations + 1)))
    if metrics.coverage_ratio < MIN_COVERAGE_RATIO:
        candidates.append(replace(base,
                                  target_points=min(MAX_TARGET_POINTS, int(base.target_points * 1.3)),
                                  mls_search_radius=base.mls_search_radius * 0.9,
                                  max_nn=min(100, base.max_nn + 20),
                                  depth=min(18, base.depth + 2)))
    return candidates[: budget["max_candidates"]]

class Searcher:
    def __init__(self, points: np.ndarray, budget: dict, seed: int):
        self.points, self.budget, self.seed = points, budget, seed
        self.cache: dict = {}
        self.recon_calls = 0

    def run(self, params: Params):
        key = astuple(params)
        if key not in self.cache:
            self.recon_calls += 1
            mesh, densities, sampled = poisson_core(self.points, params, self.seed)
            if mesh is None:
                metrics = Metrics(True, 1, float("inf"), float("inf"), 0.0, float("inf"))
                self.cache[key] = (None, None, None, metrics)
            else:
                # 搜索必须评价最终会导出的密度裁剪开放网格，而不是未裁剪的
                # Poisson 封闭壳；参考数据使用原始点云而不是重采样点。
                evaluated = crop_by_density(
                    mesh, densities, DENSITY_PERCENTILE, params.smoothing_iterations)
                self.cache[key] = (mesh, densities, sampled,
                                   evaluate(self.points, evaluated, params.target_points))
        return self.cache[key]

    def search_target_points(self, base: Params) -> Params:
        # “孔洞是否存在”不随采样点数单调变化，不能二分查找。按基础点数的
        # 相对比例生成少量候选，并用孔洞数和连续质量分数统一排序。
        counts = sorted({
            int(np.clip(round(base.target_points * multiplier),
                        MIN_TARGET_POINTS, MAX_TARGET_POINTS))
            for multiplier in self.budget["target_multipliers"]
        })
        tested = []
        for count in counts:
            params = replace(base, target_points=count)
            *_, metrics = self.run(params)
            tested.append((params, metrics))
        return min(tested, key=lambda item: (
            item[1].has_holes, item[1].hole_count, item[1].overall_score))[0]

    def optimize(self):
        base = initial_params(self.points)
        best = self.search_target_points(base)
        *_, metrics = self.run(best)
        if is_good(metrics):
            return best, metrics

        rounds = self.budget["max_iterations"]
        rounds = MAX_ITERATIONS if rounds is None else rounds
        for _ in range(rounds):
            improved = False
            for params in candidate_params(best, metrics, self.budget):
                *_, new_metrics = self.run(params)
                better = ((new_metrics.has_holes, new_metrics.hole_count,
                           new_metrics.overall_score)
                          < (metrics.has_holes, metrics.hole_count,
                             metrics.overall_score))
                if better:
                    best, metrics, improved = params, new_metrics, True
                    break
            if not improved or is_good(metrics):
                break
        return best, metrics

def process_one(path: str, output_dir: str, method: str, budget_name: str) -> dict:
    started = time.perf_counter()
    record = {"input": path, "leaf": leaf_id(path),
              "method": method, "success": False}
    try:
        points = load_points(path)
        spacing = median_spacing(points)
        record["input_points"] = int(len(points))
        record["median_spacing"] = spacing

        if method == "ball_pivoting":
            mesh, radii = ball_pivoting_core(points)
            record["bpa_radii"] = ";".join(f"{r:.6f}" for r in radii)
            record["n_components"] = n_components(mesh) if mesh is not None else 0
        elif method == "fixed_poisson":
            params = Params(target_points=FIXED_TARGET_POINTS,
                            mls_search_radius=FIXED_MLS_SPACING_MULT * spacing,
                            max_nn=FIXED_MAX_NN, depth=FIXED_DEPTH,
                            smoothing_iterations=FIXED_SMOOTHING,
                            density_compensation=False)
            raw, densities, _ = poisson_core(points, params, RNG_SEED)
            mesh = crop_by_density(raw, densities, DENSITY_PERCENTILE, params.smoothing_iterations)
            record.update({"target_points": params.target_points, "depth": params.depth,
                           "density_percentile": DENSITY_PERCENTILE})
        elif method == "adaptive_poisson":
            searcher = Searcher(points, BUDGETS[budget_name], RNG_SEED)
            params, metrics = searcher.optimize()
            raw, densities, _, _ = searcher.run(params)
            mesh = crop_by_density(raw, densities, DENSITY_PERCENTILE, params.smoothing_iterations)
            record.update({"budget": budget_name, "target_points": params.target_points,
                           "depth": params.depth, "max_nn": params.max_nn,
                           "mls_radius": params.mls_search_radius,
                           "density_percentile": DENSITY_PERCENTILE,
                           "search_hole_count": metrics.hole_count,
                           "search_avg_error": metrics.avg_distance_error,
                           "search_coverage": metrics.coverage_ratio,
                           "n_reconstructions": searcher.recon_calls})
        else:
            raise ValueError(f"unknown METHOD: {method}")

        if mesh is None or len(mesh.triangles) == 0:
            record["error"] = "reconstruction produced no mesh"
            record["seconds"] = time.perf_counter() - started
            return record

        os.makedirs(output_dir, exist_ok=True)
        o3d.io.write_triangle_mesh(
            os.path.join(output_dir, f"{record['leaf']}_{method}.obj"), mesh)

        record.update({
            "success": True,
            "mesh_vertices": int(len(mesh.vertices)),
            "mesh_triangles": int(len(mesh.triangles)),
            "area": float(mesh.get_surface_area()),
            "chamfer": chamfer(mesh, points),
            "unsupported_ratio": unsupported_area_ratio(mesh, points, spacing),
        })
    except Exception as error:
        record["error"] = f"{error} | {traceback.format_exc().splitlines()[-1]}"

    record["seconds"] = time.perf_counter() - started
    return record

def sweep_one(path: str, method: str, budget_name: str) -> list[dict]:
    """一片叶只做一次 Poisson，然后在同一结果上按各分位裁剪。"""
    rows = []
    try:
        points = load_points(path)
        spacing = median_spacing(points)
        leaf = leaf_id(path)

        if method == "fixed_poisson":
            params = Params(FIXED_TARGET_POINTS, FIXED_MLS_SPACING_MULT * spacing,
                            FIXED_MAX_NN, FIXED_DEPTH, FIXED_SMOOTHING,
                            density_compensation=False)
            raw, densities, _ = poisson_core(points, params, RNG_SEED)
        else:
            searcher = Searcher(points, BUDGETS[budget_name], RNG_SEED)
            params, _ = searcher.optimize()
            raw, densities, _, _ = searcher.run(params)

        for percentile in PERCENTILE_SWEEP:
            mesh = crop_by_density(raw, densities, percentile, params.smoothing_iterations)
            rows.append({
                "leaf": leaf, "input": path, "method": method, "percentile": percentile,
                "input_points": int(len(points)), "median_spacing": spacing,
                "area": float(mesh.get_surface_area()) if mesh is not None else None,
                "n_triangles": int(len(mesh.triangles)) if mesh is not None else 0,
                "n_components": n_components(mesh) if mesh is not None else 0,
                "chamfer": chamfer(mesh, points) if mesh is not None else None,
                "unsupported_ratio": (unsupported_area_ratio(mesh, points, spacing)
                                      if mesh is not None else None),
            })
    except Exception as error:
        rows.append({"leaf": leaf_id(path), "input": path, "method": method,
                     "error": f"{error}"})
    return rows

def _init_worker():
    global _INNER_WORKERS
    _INNER_WORKERS = 1
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

FILE_META: dict[str, dict] = {}

BASELINE_METHODS = ("adaptive_poisson", "fixed_poisson", "ball_pivoting")

def leaf_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]
