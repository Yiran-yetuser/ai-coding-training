"""Day 008 starter: arrange image patches for a ViT-style input."""

import numpy as np


def patchify(images, patch_size):
    # TODO: 将高度和宽度拆成 patch 网格与 patch 内坐标。
    B, H, W = images.shape
    n_h = H // patch_size
    n_w = W // patch_size
    x = images.reshape(B, n_h, patch_size, n_w, patch_size)
    # TODO: 按行优先排列 patch，并展平每个 patch。
    x = x.transpose(0, 1, 3, 2, 4)
    patches = x.reshape(B, n_h * n_w, patch_size * patch_size)
    return patches
    raise NotImplementedError


# 样例 1：检查 patch 网格顺序和每个 patch 内的展平顺序。
image = np.arange(16, dtype=np.uint8).reshape(1, 4, 4)
original = image.copy()
patches = patchify(image, patch_size=2)
expected = np.array(
    [[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]],
    dtype=np.uint8,
)
assert patches.shape == (1, 4, 4)
assert patches.dtype == image.dtype
np.testing.assert_array_equal(patches, expected)
assert np.array_equal(image, original)

# 样例 2：batch 中每张图恰好形成一个 patch。
images = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int64)
patches = patchify(images, patch_size=2)
assert patches.shape == (2, 1, 4)
np.testing.assert_array_equal(patches[:, 0, :], [[1, 2, 3, 4], [5, 6, 7, 8]])
