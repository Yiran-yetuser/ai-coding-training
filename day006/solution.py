"""Day 006 starter: debug uint8 image preprocessing."""

import numpy as np


def preprocess_image(image):
    # TODO: 修复 dtype 转换与缩放的顺序，返回新数组。
    image_float = image.astype(np.float32)
    image_float /= 255.0
    return image_float


# 样例 1：RGB 图，检查 dtype、范围、精度和输入保护。
image = np.array([[[0, 128, 255], [255, 0, 64]]], dtype=np.uint8)
original = image.copy()
processed = preprocess_image(image)
assert processed.shape == image.shape
assert processed.dtype == np.float32
np.testing.assert_allclose(processed, [[[0.0, 128 / 255, 1.0], [1.0, 0.0, 64 / 255]]])
assert np.array_equal(image, original)

# 样例 2：二维灰度图仍保持原 shape，像素范围为 0～1。
grayscale = np.array([[0, 255]], dtype=np.uint8)
processed_gray = preprocess_image(grayscale)
assert processed_gray.shape == grayscale.shape
np.testing.assert_allclose(processed_gray, [[0.0, 1.0]])
