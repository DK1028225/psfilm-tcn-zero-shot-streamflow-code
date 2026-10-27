"""Standalone Stage23D convex-hull geometry reference for advisor review.

Derived from the user's frozen Stage23D_support_space_same_checkpoint_controls.py
function `project_to_convex_hull` (active-subset enumeration).  The original
Stage23D calls use ATTR_PROJ_TOL = MOD_PROJ_TOL = 1e-9.  This independent
example documents the geometric solver, not the full Stage23D diagnostic
pipeline or trained network evaluation.  Requires numpy only.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np

ATTR_PROJ_TOL = 1e-9
MOD_PROJ_TOL = 1e-9


def project_to_convex_hull(points, target, tol=1e-10):
    """Euclidean projection onto the convex hull of <=5 rows of points.

    Original numerical procedure: examine every non-empty active subset,
    project onto its affine span using float64 least squares (rcond=None),
    reject weights below -tol, zero only tiny negative weights, renormalize,
    and keep a minimum-distance feasible candidate (improvement >1e-14 in d²).
    """
    P = np.asarray(points, dtype=np.float64)
    x = np.asarray(target, dtype=np.float64).reshape(-1)
    if P.ndim != 2 or P.shape[0] < 1 or P.shape[0] > 5:
        raise ValueError('points must be a nonempty [n_source, n_dim] matrix, with n_source <= 5')
    if P.shape[1] != len(x) or not np.isfinite(P).all() or not np.isfinite(x).all():
        raise ValueError('points/target dimension mismatch or non-finite entries')
    if not (np.isfinite(tol) and tol >= 0):
        raise ValueError('invalid tolerance')

    n = P.shape[0]
    best = None
    for k in range(1, n + 1):
        for subset in combinations(range(n), k):
            Q = P[list(subset)]
            if k == 1:
                w = np.array([1.0], dtype=np.float64)
            else:
                q_last = Q[-1]
                D = (Q[:-1] - q_last).T
                u, _, _, _ = np.linalg.lstsq(D, x - q_last, rcond=None)
                w = np.concatenate([u, np.array([1.0 - np.sum(u)], dtype=np.float64)])
            if np.min(w) < -tol:
                continue
            w_clean = np.where(w < 0.0, 0.0, w)
            s = float(np.sum(w_clean))
            if s <= 0.0:
                continue
            w_clean = w_clean / s
            projected = w_clean @ Q
            diff = projected - x
            d2 = float(np.dot(diff, diff))
            if best is None or d2 < best['d2'] - 1e-14:
                full_w = np.zeros(n, dtype=np.float64)
                full_w[list(subset)] = w_clean
                best = {
                    'd2': d2, 'weights': full_w,
                    'projected': projected,
                    'active_indices': tuple(int(i) for i in subset),
                }

    if best is None:
        raise RuntimeError('No feasible active subset for hull projection')
    distance = float(np.sqrt(best['d2']))
    return {
        'weights': best['weights'],
        'projected': best['projected'],
        'distance': distance,
        'normalized_distance': float(distance / max(np.linalg.norm(x), 1.0)),
        'active_indices': best['active_indices'],
        'sum_weight_error': float(abs(np.sum(best['weights']) - 1.0)),
        'min_weight': float(np.min(best['weights'])),
    }


def joint_attribute_projection(source_static_scaled, target_static_scaled):
    """Project standardized 4-D target attributes onto the source attribute hull."""
    P = np.asarray(source_static_scaled, dtype=np.float64)
    z = np.asarray(target_static_scaled, dtype=np.float64).reshape(-1)
    if P.ndim != 2 or P.shape[1] != 4 or z.shape != (4,):
        raise ValueError('Expected [n_source,4] and [4] standardized descriptors')
    return project_to_convex_hull(P, z, tol=ATTR_PROJ_TOL)


def concatenate_gamma_beta(gamma_beta_by_block):
    """For each conditioned block, append [gamma_l, beta_l], in block order.

    Input: ordered sequence of (gamma_vector, beta_vector) pairs.  The frozen
    Stage23D uses blocks 2, 3, 4; their trained coefficients must be obtained
    separately from the same fixed FiLM network and source-only scaler.
    """
    chunks = []
    for gamma, beta in gamma_beta_by_block:
        g = np.asarray(gamma, dtype=np.float64).reshape(-1)
        b = np.asarray(beta, dtype=np.float64).reshape(-1)
        if len(g) != len(b) or len(g) == 0:
            raise ValueError('Invalid per-block gamma/beta shape')
        chunks.extend((g, b))
    if len(chunks) != 6:  # three FiLM-conditioned blocks
        raise ValueError('Expected three conditioned blocks (blocks 2, 3, 4)')
    return np.concatenate(chunks)


def joint_modulation_projection(source_full_vectors, target_full_vector):
    """Project a full target modulation vector onto joint source modulation hull.

    Single convex weight vector applies to all blocks, gamma and beta channels.
    This function does not compute FiLM coefficients from checkpoint weights.
    """
    return project_to_convex_hull(source_full_vectors, target_full_vector, tol=MOD_PROJ_TOL)
