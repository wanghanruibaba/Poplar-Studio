#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast, scale-invariant reconstruction of individual leaf point clouds.

Pipeline
--------
1. Robust cleanup and scale normalization.
2. PCA projection to a local leaf coordinate system (u, v, w).
3. Concave outer-boundary extraction from a 2-D alpha complex.
4. Boundary-constrained Delaunay triangulation (internal gaps are filled).
5. Local quadratic MLS recovery/smoothing of the height w.
6. Inverse PCA transform to the original, physically calibrated coordinates.

The output is an open, single-layer leaf surface.  Its one boundary loop is the
natural leaf margin; additional boundary loops are reported as internal holes.

python scripts/pca_reconstruct.py -i data/raw -o runs/pca --recursive

Dependencies: NumPy and SciPy only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree


DEFAULT_PATTERN = "*.txt"


@dataclass
class ReconstructionConfig:
    """Scale-free reconstruction settings."""

    min_points: int = 30
    max_vertices: int = 5000
    outlier_k: int = 8
    outlier_mad: float = 8.0
    alpha_factor: float = 2.5
    max_alpha_factor: float = 12.0
    min_outline_coverage: float = 0.97
    max_flatness_ratio: float = 0.15
    height_smooth_k: int = 18
    height_smooth_blend: float = 0.65


@dataclass
class MeshResult:
    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    colors: Optional[np.ndarray]
    metadata: dict


def load_point_cloud(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load XYZ and optional RGB from a whitespace/comma-delimited text file."""
    try:
        data = np.loadtxt(path, dtype=np.float64)
    except ValueError:
        data = np.loadtxt(path, dtype=np.float64, delimiter=",")

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError("Point cloud must contain at least three XYZ columns")

    finite = np.all(np.isfinite(data[:, :3]), axis=1)
    data = data[finite]
    points = data[:, :3]
    colors = detect_rgb_columns(data)
    return points, colors


def detect_rgb_columns(data: np.ndarray) -> Optional[np.ndarray]:
    """Detect RGB in columns 4--6 without mistaking unit normals for colors."""
    if data.shape[1] < 6:
        return None

    candidate = data[:, 3:6]
    finite = candidate[np.all(np.isfinite(candidate), axis=1)]
    if len(finite) == 0:
        return None

    # Project files with ten columns use XYZ RGB label Nx Ny Nz.
    if data.shape[1] >= 10:
        colors = candidate.copy()
    else:
        norms = np.linalg.norm(finite, axis=1)
        looks_like_normals = (
            np.mean((norms > 0.75) & (norms < 1.25)) > 0.8
            and np.min(finite) < -0.02
            and np.max(finite) <= 1.05
        )
        if looks_like_normals:
            return None
        colors = candidate.copy()

    if np.nanmax(colors) > 1.0:
        colors /= 255.0
    return np.clip(colors, 0.0, 1.0)


def robust_outlier_mask(points: np.ndarray, k: int, mad_factor: float) -> np.ndarray:
    """Remove only extreme isolated points while retaining sparse leaf margins."""
    n = len(points)
    if mad_factor <= 0 or n <= k + 2:
        return np.ones(n, dtype=bool)

    extent = np.ptp(points, axis=0)
    scale = max(float(np.linalg.norm(extent)), np.finfo(float).eps)
    normalized = (points - np.median(points, axis=0)) / scale
    k_eff = min(k + 1, n)
    distances, _ = cKDTree(normalized).query(normalized, k=k_eff, workers=-1)
    score = np.mean(distances[:, 1:], axis=1)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    if mad <= np.finfo(float).eps:
        return np.ones(n, dtype=bool)

    robust_sigma = 1.4826 * mad
    # The quantile guard prevents removal of more than 0.5% merely because the
    # natural leaf edge is less dense than the interior.
    cutoff = max(median + mad_factor * robust_sigma, float(np.quantile(score, 0.995)))
    return score <= cutoff


def pca_frame(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Return center, right-handed PCA basis, and normalized local coordinates."""
    center = np.mean(points, axis=0)
    centered = points - center
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    basis = eigenvectors[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1.0

    local = centered @ basis
    planar_extent = np.ptp(local[:, :2], axis=0)
    scale = max(float(np.linalg.norm(planar_extent)), np.finfo(float).eps)
    return center, basis, local / scale, scale, eigenvalues


def aggregate_grid(
    local: np.ndarray,
    colors: Optional[np.ndarray],
    max_vertices: int,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Deterministically consolidate projected samples into at most max_vertices."""
    uv = local[:, :2]
    n = len(uv)

    def aggregate(cell_size: float) -> tuple[np.ndarray, Optional[np.ndarray]]:
        origin = np.min(uv, axis=0)
        if cell_size <= 0:
            keys = np.round(uv, decimals=12)
        else:
            keys = np.floor((uv - origin) / cell_size).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        count = int(inverse.max()) + 1
        counts = np.bincount(inverse, minlength=count).astype(np.float64)
        reduced = np.zeros((count, 3), dtype=np.float64)
        np.add.at(reduced, inverse, local)
        reduced /= counts[:, None]

        reduced_colors = None
        if colors is not None:
            reduced_colors = np.zeros((count, 3), dtype=np.float64)
            np.add.at(reduced_colors, inverse, colors)
            reduced_colors /= counts[:, None]
        return reduced, reduced_colors

    deduplicated, deduplicated_colors = aggregate(0.0)
    if len(deduplicated) <= max_vertices:
        return deduplicated, deduplicated_colors

    uv_extent = np.ptp(uv, axis=0)
    high = max(float(np.max(uv_extent)), 1e-6)
    low = 0.0
    best = (deduplicated, deduplicated_colors)

    for _ in range(24):
        cell = 0.5 * (low + high)
        candidate = aggregate(cell)
        if len(candidate[0]) > max_vertices:
            low = cell
        else:
            high = cell
            best = candidate
    return best


def estimate_spacing(uv: np.ndarray) -> float:
    distances, _ = cKDTree(uv).query(uv, k=2, workers=-1)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if len(nearest) == 0:
        raise ValueError("Projected points are duplicated or degenerate")
    return float(np.median(nearest))


def delaunay_2d(uv: np.ndarray) -> np.ndarray:
    try:
        return Delaunay(uv, qhull_options="Qbb Qc Qz Q12").simplices.astype(np.int64)
    except QhullError:
        return Delaunay(uv, qhull_options="QJ Qbb Qc Q12").simplices.astype(np.int64)


def triangle_circumradii(uv: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p0 = uv[triangles[:, 0]]
    p1 = uv[triangles[:, 1]]
    p2 = uv[triangles[:, 2]]
    edge_1 = p1 - p0
    edge_2 = p2 - p0
    twice_area = np.abs(edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0])
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p2 - p0, axis=1)
    c = np.linalg.norm(p0 - p1, axis=1)
    radius = np.full(len(triangles), np.inf, dtype=np.float64)
    valid = twice_area > 1e-14
    radius[valid] = a[valid] * b[valid] * c[valid] / (2.0 * twice_area[valid])
    return radius, valid


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def largest_triangle_component(triangles: np.ndarray) -> np.ndarray:
    if len(triangles) == 0:
        return triangles
    edge_owner: dict[tuple[int, int], int] = {}
    union_find = UnionFind(len(triangles))
    for triangle_index, triangle in enumerate(triangles):
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            previous = edge_owner.get(edge)
            if previous is None:
                edge_owner[edge] = triangle_index
            else:
                union_find.union(previous, triangle_index)

    roots = np.fromiter((union_find.find(i) for i in range(len(triangles))), dtype=np.int64)
    labels, counts = np.unique(roots, return_counts=True)
    largest = labels[int(np.argmax(counts))]
    return triangles[roots == largest]


def boundary_edges(triangles: np.ndarray) -> list[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            counts[edge] = counts.get(edge, 0) + 1
    return [edge for edge, count in counts.items() if count == 1]


def boundary_loops(edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    adjacency: dict[int, list[int]] = {}
    for left, right in edges:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)

    unused = {tuple(sorted(edge)) for edge in edges}
    loops: list[list[int]] = []
    while unused:
        first_edge = next(iter(unused))
        start, current = first_edge
        previous = start
        loop = [start, current]
        unused.discard(first_edge)

        for _ in range(len(edges) + 1):
            candidates = [
                node for node in adjacency.get(current, [])
                if tuple(sorted((current, node))) in unused
            ]
            if not candidates:
                break
            # A valid planar boundary has degree two.  This deterministic
            # fallback also handles rare alpha-complex junctions.
            next_node = candidates[0] if len(candidates) == 1 else min(candidates)
            unused.discard(tuple(sorted((current, next_node))))
            previous, current = current, next_node
            if current == start:
                loops.append(loop)
                break
            loop.append(current)
    return [loop for loop in loops if len(loop) >= 3]


def polygon_area(polygon: np.ndarray) -> float:
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized ray casting; polygon boundary vertices are handled separately."""
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        crosses = (y1 > y) != (y0 > y)
        denominator = y0 - y1
        denominator = denominator if abs(denominator) > 1e-15 else 1e-15
        x_cross = (x0 - x1) * (y - y1) / denominator + x1
        inside ^= crosses & (x < x_cross)
        x0, y0 = x1, y1
    return inside


def extract_outer_boundary(
    uv: np.ndarray,
    triangles: np.ndarray,
    spacing: float,
    config: ReconstructionConfig,
) -> tuple[list[int], float, float, np.ndarray]:
    """Extract the largest concave alpha-complex loop as the leaf margin."""
    radii, valid = triangle_circumradii(uv, triangles)
    factors = []
    factor = config.alpha_factor
    while factor < config.max_alpha_factor:
        factors.append(factor)
        factor *= 1.35
    factors.append(config.max_alpha_factor)

    best: Optional[tuple[list[int], float, float, np.ndarray]] = None
    for alpha_factor in factors:
        selected = triangles[valid & (radii <= alpha_factor * spacing)]
        selected = largest_triangle_component(selected)
        if len(selected) == 0:
            continue

        selected_boundary_edges = boundary_edges(selected)
        degrees: dict[int, int] = {}
        for left, right in selected_boundary_edges:
            degrees[left] = degrees.get(left, 0) + 1
            degrees[right] = degrees.get(right, 0) + 1
        # Pinched alpha boundaries contain degree > 2 junctions and do not
        # define a valid simple leaf margin.  Increase alpha until every
        # boundary vertex belongs to exactly one closed loop.
        if any(degree != 2 for degree in degrees.values()):
            continue

        loops = boundary_loops(selected_boundary_edges)
        if not loops:
            continue
        loop = max(loops, key=lambda indices: abs(polygon_area(uv[indices])))
        polygon = uv[loop]
        if polygon_area(polygon) < 0:
            loop = list(reversed(loop))
            polygon = uv[loop]

        inside = points_in_polygon(uv, polygon)
        inside[np.asarray(loop, dtype=np.int64)] = True
        coverage = float(np.mean(inside))
        candidate = (loop, alpha_factor, coverage, inside)
        if best is None or coverage > best[2]:
            best = candidate
        if coverage >= config.min_outline_coverage:
            return candidate

    if best is not None and best[2] >= 0.80:
        return best

    # Guaranteed fallback for extremely sparse or irregular clusters.
    hull = ConvexHull(uv)
    loop = hull.vertices.tolist()
    polygon = uv[loop]
    if polygon_area(polygon) < 0:
        loop.reverse()
        polygon = uv[loop]
    inside = points_in_polygon(uv, polygon)
    inside[np.asarray(loop, dtype=np.int64)] = True
    return loop, math.inf, float(np.mean(inside)), inside


def constrained_triangulation(
    uv: np.ndarray,
    triangles: np.ndarray,
    boundary: Sequence[int],
) -> np.ndarray:
    """Keep the Delaunay subcomplex inside an alpha-derived boundary loop.

    The boundary is itself composed of Delaunay edges.  Selecting triangles by
    their centroids therefore preserves those edges as hard constraints while
    filling every interior gap of the alpha complex.
    """
    polygon = uv[np.asarray(boundary, dtype=np.int64)]
    centroids = np.mean(uv[triangles], axis=1)
    inside = points_in_polygon(centroids, polygon)
    result = largest_triangle_component(triangles[inside])
    if len(result) == 0:
        raise ValueError("Constrained triangulation produced no triangles")

    # Ensure consistent counter-clockwise orientation in PCA coordinates.
    p0 = uv[result[:, 0]]
    p1 = uv[result[:, 1]]
    p2 = uv[result[:, 2]]
    edge_1 = p1 - p0
    edge_2 = p2 - p0
    signed_twice_area = edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    result = result[np.abs(signed_twice_area) > 1e-14]
    flip = signed_twice_area[np.abs(signed_twice_area) > 1e-14] < 0
    result[flip, 1], result[flip, 2] = result[flip, 2].copy(), result[flip, 1].copy()
    return result


def point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray, eps: float = 1e-14) -> bool:
    """Return True when point lies inside or on a counter-clockwise triangle."""
    a, b, c = triangle
    cross_ab = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    cross_bc = (c[0] - b[0]) * (point[1] - b[1]) - (c[1] - b[1]) * (point[0] - b[0])
    cross_ca = (a[0] - c[0]) * (point[1] - c[1]) - (a[1] - c[1]) * (point[0] - c[0])
    return cross_ab >= -eps and cross_bc >= -eps and cross_ca >= -eps


def ear_clip_polygon(uv: np.ndarray, loop: Sequence[int]) -> np.ndarray:
    """Triangulate a simple boundary loop without adding new vertices."""
    indices = list(dict.fromkeys(int(index) for index in loop))
    if len(indices) < 3:
        return np.empty((0, 3), dtype=np.int64)
    if polygon_area(uv[indices]) < 0:
        indices.reverse()

    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(indices) > 3 and guard < len(loop) * len(loop) + 10:
        guard += 1
        ear_found = False
        count = len(indices)
        for position in range(count):
            previous = indices[(position - 1) % count]
            current = indices[position]
            following = indices[(position + 1) % count]
            a, b, c = uv[[previous, current, following]]
            signed_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if signed_area <= 1e-14:
                continue

            ear_triangle = np.asarray([a, b, c])
            contains_other = False
            for candidate in indices:
                if candidate in (previous, current, following):
                    continue
                if point_in_triangle_2d(uv[candidate], ear_triangle):
                    contains_other = True
                    break
            if contains_other:
                continue

            triangles.append((previous, current, following))
            del indices[position]
            ear_found = True
            break

        if not ear_found:
            # Remove the least significant nearly-collinear vertex and retry.
            areas = []
            for position in range(count):
                a = uv[indices[(position - 1) % count]]
                b = uv[indices[position]]
                c = uv[indices[(position + 1) % count]]
                areas.append(abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])))
            del indices[int(np.argmin(areas))]

    if len(indices) == 3:
        a, b, c = uv[indices]
        signed_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(signed_area) > 1e-14:
            if signed_area < 0:
                indices[1], indices[2] = indices[2], indices[1]
            triangles.append(tuple(indices))
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def fill_internal_boundary_loops(
    uv: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Fill every boundary loop except the largest natural leaf margin."""
    loops = boundary_loops(boundary_edges(triangles))
    if len(loops) <= 1:
        return triangles, 0

    outer_index = int(np.argmax([abs(polygon_area(uv[loop])) for loop in loops]))
    patches = []
    for loop_index, loop in enumerate(loops):
        if loop_index == outer_index:
            continue
        patch = ear_clip_polygon(uv, loop)
        if len(patch):
            patches.append(patch)

    if not patches:
        return triangles, 0
    repaired = np.vstack([triangles, *patches])
    return repaired, len(patches)


def compact_mesh(
    local: np.ndarray,
    triangles: np.ndarray,
    colors: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    used = np.unique(triangles.ravel())
    remap = np.full(len(local), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    compact_colors = colors[used] if colors is not None else None
    return local[used], remap[triangles], compact_colors


def smooth_heights_quadratic(
    local: np.ndarray,
    k: int,
    blend: float,
) -> np.ndarray:
    """Recover w with a local quadratic MLS fit while retaining leaf curvature."""
    if k < 6 or blend <= 0 or len(local) < 7:
        return local
    uv = local[:, :2]
    z = local[:, 2]
    k_eff = min(max(k, 6), len(local))
    distances, neighbors = cKDTree(uv).query(uv, k=k_eff, workers=-1)
    fitted = z.copy()

    for index in range(len(local)):
        ids = np.atleast_1d(neighbors[index])
        delta = uv[ids] - uv[index]
        distance = np.atleast_1d(distances[index])
        bandwidth = max(float(distance[-1]), 1e-12)
        weight = np.exp(-((distance / bandwidth) ** 2))
        design = np.column_stack(
            (
                np.ones(len(ids)),
                delta[:, 0],
                delta[:, 1],
                delta[:, 0] ** 2,
                delta[:, 0] * delta[:, 1],
                delta[:, 1] ** 2,
            )
        )
        weighted_design = design * np.sqrt(weight)[:, None]
        weighted_height = z[ids] * np.sqrt(weight)
        try:
            coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_height, rcond=None)
            fitted[index] = coefficients[0]
        except np.linalg.LinAlgError:
            fitted[index] = np.average(z[ids], weights=weight)

    result = local.copy()
    result[:, 2] = (1.0 - blend) * z + blend * fitted
    return result


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]
    face_normals = np.cross(p1 - p0, p2 - p0)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-15
    normals[valid] /= lengths[valid, None]
    normals[~valid] = (0.0, 0.0, 1.0)
    return normals


def mesh_topology(triangles: np.ndarray) -> tuple[int, int]:
    loops = boundary_loops(boundary_edges(triangles))
    return len(loops), max(0, len(loops) - 1)


def reconstruct_leaf(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    config: ReconstructionConfig,
) -> MeshResult:
    start = time.perf_counter()
    if len(points) < config.min_points:
        raise ValueError(f"Too few points: {len(points)} < {config.min_points}")

    mask = robust_outlier_mask(points, config.outlier_k, config.outlier_mad)
    filtered_points = points[mask]
    filtered_colors = colors[mask] if colors is not None else None
    if len(filtered_points) < config.min_points:
        raise ValueError("Too few points remain after outlier filtering")

    center, basis, local, scale, eigenvalues = pca_frame(filtered_points)
    planar_variance = max(float(eigenvalues[0] + eigenvalues[1]), 1e-15)
    flatness_ratio = float(eigenvalues[2] / planar_variance)
    if config.max_flatness_ratio > 0 and flatness_ratio > config.max_flatness_ratio:
        raise ValueError(
            f"PCA projection is unreliable (flatness ratio {flatness_ratio:.4f} > "
            f"{config.max_flatness_ratio:.4f}); the cluster may contain multiple leaves "
            "or a strongly folded/overlapping leaf"
        )

    local, filtered_colors = aggregate_grid(local, filtered_colors, config.max_vertices)
    if len(local) < config.min_points:
        raise ValueError("Too few unique projected points after consolidation")

    uv = local[:, :2]
    spacing = estimate_spacing(uv)
    triangles_all = delaunay_2d(uv)
    boundary, alpha_used, outline_coverage, _ = extract_outer_boundary(
        uv, triangles_all, spacing, config
    )
    triangles = constrained_triangulation(uv, triangles_all, boundary)
    triangles, filled_internal_holes = fill_internal_boundary_loops(uv, triangles)
    local, triangles, filtered_colors = compact_mesh(local, triangles, filtered_colors)
    local = smooth_heights_quadratic(
        local, config.height_smooth_k, config.height_smooth_blend
    )

    vertices = center + (local * scale) @ basis.T
    normals = vertex_normals(vertices, triangles)
    loop_count, internal_holes = mesh_topology(triangles)
    if loop_count != 1 or internal_holes != 0:
        raise ValueError(
            f"Mesh topology check failed: boundary_loops={loop_count}, "
            f"internal_holes={internal_holes}"
        )
    elapsed = time.perf_counter() - start

    metadata = {
        "input_points": int(len(points)),
        "filtered_points": int(len(filtered_points)),
        "mesh_vertices": int(len(vertices)),
        "mesh_triangles": int(len(triangles)),
        "boundary_loops": int(loop_count),
        "internal_holes": int(internal_holes),
        "filled_internal_holes": int(filled_internal_holes),
        "outline_coverage": float(outline_coverage),
        "alpha_factor_used": None if math.isinf(alpha_used) else float(alpha_used),
        "convex_hull_fallback": bool(math.isinf(alpha_used)),
        "normalized_spacing": float(spacing),
        "physical_scale": float(scale),
        "pca_flatness_ratio": flatness_ratio,
        "elapsed_seconds": float(elapsed),
        "center": center.tolist(),
        "pca_basis": basis.tolist(),
        "config": asdict(config),
    }
    return MeshResult(vertices, triangles, normals, filtered_colors, metadata)


def write_obj(path: Path, mesh: MeshResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write("# PCA-constrained leaf reconstruction\n")
        if mesh.colors is None:
            for vertex in mesh.vertices:
                file.write(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}\n")
        else:
            for vertex, color in zip(mesh.vertices, mesh.colors):
                file.write(
                    f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f} "
                    f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
                )
        for normal in mesh.normals:
            file.write(f"vn {normal[0]:.9f} {normal[1]:.9f} {normal[2]:.9f}\n")
        for triangle in mesh.triangles + 1:
            file.write(
                f"f {triangle[0]}//{triangle[0]} "
                f"{triangle[1]}//{triangle[1]} {triangle[2]}//{triangle[2]}\n"
            )


def write_ply(path: Path, mesh: MeshResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_color = mesh.colors is not None
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(mesh.vertices)}\n")
        file.write("property double x\nproperty double y\nproperty double z\n")
        file.write("property double nx\nproperty double ny\nproperty double nz\n")
        if has_color:
            file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write(f"element face {len(mesh.triangles)}\n")
        file.write("property list uchar int vertex_indices\nend_header\n")

        if has_color:
            rgb = np.rint(np.clip(mesh.colors, 0.0, 1.0) * 255.0).astype(np.uint8)
            for vertex, normal, color in zip(mesh.vertices, mesh.normals, rgb):
                file.write(
                    f"{vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f} "
                    f"{normal[0]:.9f} {normal[1]:.9f} {normal[2]:.9f} "
                    f"{color[0]} {color[1]} {color[2]}\n"
                )
        else:
            for vertex, normal in zip(mesh.vertices, mesh.normals):
                file.write(
                    f"{vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f} "
                    f"{normal[0]:.9f} {normal[1]:.9f} {normal[2]:.9f}\n"
                )
        for triangle in mesh.triangles:
            file.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")


def output_paths(input_file: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    stem = input_file.stem
    return (
        output_dir / f"{stem}_pca_mesh.obj",
        output_dir / f"{stem}_pca_mesh.ply",
        output_dir / f"{stem}_pca_mesh.json",
    )


def process_file(input_file: Path, output_dir: Path, config: ReconstructionConfig) -> dict:
    record = {"input": str(input_file), "success": False}
    try:
        points, colors = load_point_cloud(input_file)
        mesh = reconstruct_leaf(points, colors, config)
        obj_path, ply_path, json_path = output_paths(input_file, output_dir)
        write_obj(obj_path, mesh)
        write_ply(ply_path, mesh)
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(mesh.metadata, file, indent=2, ensure_ascii=False)
        record.update(mesh.metadata)
        record.update(
            {
                "success": True,
                "obj": str(obj_path),
                "ply": str(ply_path),
                "metadata": str(json_path),
            }
        )
        print(
            f"[OK] {input_file.name}: {len(points)} points -> "
            f"{len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles, "
            f"internal_holes={mesh.metadata['internal_holes']}, "
            f"{mesh.metadata['elapsed_seconds']:.3f} s"
        )
    except Exception as error:
        record["error"] = str(error)
        print(f"[FAILED] {input_file}: {error}", file=sys.stderr)
    return record


def find_input_files(input_path: Path, pattern: str, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    iterator: Iterable[Path] = (
        input_path.rglob(pattern) if recursive else input_path.glob(pattern)
    )
    files = sorted(path for path in iterator if path.is_file())

    # Project convention: prefer clustered-leaf filenames.  If a directory
    # contains none of them, every TXT file in that directory is treated as an
    # individual leaf point cloud.  An explicitly supplied custom pattern is
    # never broadened automatically.
    if not files and pattern == DEFAULT_PATTERN:
        fallback_pattern = "*.txt"
        fallback_iterator: Iterable[Path] = (
            input_path.rglob(fallback_pattern)
            if recursive
            else input_path.glob(fallback_pattern)
        )
        files = sorted(path for path in fallback_iterator if path.is_file())
        if files:
            print(
                f"No files matched {DEFAULT_PATTERN!r}; "
                f"falling back to all {fallback_pattern} files."
            )
    return files


def write_summary(path: Path, records: list[dict]) -> None:
    fields = [
        "input",
        "success",
        "input_points",
        "filtered_points",
        "mesh_vertices",
        "mesh_triangles",
        "boundary_loops",
        "internal_holes",
        "outline_coverage",
        "pca_flatness_ratio",
        "elapsed_seconds",
        "obj",
        "ply",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast PCA + concave-boundary constrained reconstruction of leaf point clouds"
    )
    parser.add_argument("--input", "-i", required=True, help="Input TXT file or directory")
    parser.add_argument("--output", "-o", help="Output directory (default: <input>/pca_mesh_output)")
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=(
            f"Input filename pattern (default: {DEFAULT_PATTERN}; "
            "falls back to *.txt when no default-pattern files exist)"
        ),
    )
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively")
    parser.add_argument("--max-files", type=int, default=0, help="Process only the first N files")
    parser.add_argument("--max-vertices", type=int, default=5000)
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument("--alpha-factor", type=float, default=2.5)
    parser.add_argument("--outline-coverage", type=float, default=0.97)
    parser.add_argument(
        "--max-flatness-ratio",
        type=float,
        default=0.15,
        help="Reject non-2.5-D clusters above this PCA ratio; 0 disables the check",
    )
    parser.add_argument("--smooth-k", type=int, default=18)
    parser.add_argument("--smooth-blend", type=float, default=0.65)
    parser.add_argument("--outlier-mad", type=float, default=8.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input does not exist: {input_path}", file=sys.stderr)
        return 2

    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    elif input_path.is_file():
        output_dir = input_path.parent / "pca_mesh_output"
    else:
        output_dir = input_path / "pca_mesh_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ReconstructionConfig(
        min_points=max(3, args.min_points),
        max_vertices=max(30, args.max_vertices),
        outlier_mad=max(0.0, args.outlier_mad),
        alpha_factor=max(0.5, args.alpha_factor),
        min_outline_coverage=float(np.clip(args.outline_coverage, 0.5, 1.0)),
        max_flatness_ratio=max(0.0, args.max_flatness_ratio),
        height_smooth_k=max(0, args.smooth_k),
        height_smooth_blend=float(np.clip(args.smooth_blend, 0.0, 1.0)),
    )

    files = find_input_files(input_path, args.pattern, args.recursive)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        print(f"No files matched {args.pattern!r} under {input_path}", file=sys.stderr)
        return 1

    print(f"Processing {len(files)} leaf point cloud(s) -> {output_dir}")
    batch_start = time.perf_counter()
    records = []
    for path in files:
        file_output_dir = output_dir
        if input_path.is_dir() and args.recursive:
            relative_parent = path.parent.relative_to(input_path)
            file_output_dir = output_dir / relative_parent
            file_output_dir.mkdir(parents=True, exist_ok=True)
        records.append(process_file(path, file_output_dir, config))
    elapsed = time.perf_counter() - batch_start
    summary_path = output_dir / "pca_reconstruction_summary.csv"
    write_summary(summary_path, records)
    successes = sum(bool(record.get("success")) for record in records)
    holes = sum(int(record.get("internal_holes", 0)) for record in records if record.get("success"))
    print(
        f"Completed: {successes}/{len(records)} succeeded, "
        f"reported internal holes={holes}, total time={elapsed:.3f} s"
    )
    print(f"Summary: {summary_path}")
    return 0 if successes == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
