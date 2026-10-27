"""Small synthetic geometric tests; no hydrometeorological records."""
import numpy as np
from joint_support_reference import (
    project_to_convex_hull, joint_attribute_projection,
    concatenate_gamma_beta, joint_modulation_projection,
)


def run():
    P = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    a = project_to_convex_hull(P, [0.2, 0.3], tol=1e-9)
    assert np.allclose(a['projected'], [0.2, 0.3], atol=1e-12)
    assert a['distance'] <= 1e-12
    b = project_to_convex_hull(P, [2, 0], tol=1e-9)
    assert np.allclose(b['projected'], [1, 0], atol=1e-12)
    c = project_to_convex_hull(np.array([[1, 2], [1, 2]]), [3, 4], tol=1e-9)
    assert np.allclose(c['projected'], [1, 2], atol=1e-12)
    d = joint_attribute_projection(np.array([[0, 0, 0, 0], [1, 0, 0, 0]]), [2, 0, 0, 0])
    assert np.allclose(d['projected'], [1, 0, 0, 0], atol=1e-12)
    v = concatenate_gamma_beta([([1, 2], [0, 0]), ([3], [0]), ([4], [1])])
    e = joint_modulation_projection(np.stack([v, v + 1]), v + 2)
    assert np.allclose(e['projected'], v + 1, atol=1e-12)
    for item in (a, b, c, d, e):
        assert abs(item['weights'].sum() - 1) < 1e-8
        assert item['weights'].min() >= -1e-10
    print('[STATUS] JOINT_HULL_REFERENCE_SYNTHETIC_TEST_PASS')


if __name__ == '__main__':
    run()
