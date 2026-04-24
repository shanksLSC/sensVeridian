from sensveridian.augmentation.geometry import scale_for_delta, depth_sort_indices


def test_scale_for_delta_monotonic():
    s1 = scale_for_delta(5.0, 1.0)
    s2 = scale_for_delta(5.0, 2.0)
    assert s1 > s2
    assert 0 < s2 <= 1.0


def test_depth_sort_farther_first():
    idx = depth_sort_indices([5.0, 8.0, 6.0], delta_ft=1.0)
    assert idx == [1, 2, 0]

