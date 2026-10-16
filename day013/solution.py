"""Day 013：重构 mini-batch 划分；题目与输入约定见 problem.md。"""

import numpy as np


def make_training_batches(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    drop_last: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """按顺序返回特征/标签批次；drop_last 仅控制不足一批的尾批。"""
    # TODO：用按批推进的循环与 NumPy 切片完成重构。
    batches = []
    N = features.shape[0]

    for start in range(0, N, batch_size):
        end = start + batch_size
        x_batch = features[start:end]
        y_batch = labels[start:end]

        # 尾批不足 batch_size 时，drop_last 决定是否保留
        if len(x_batch) < batch_size and drop_last:
            break

        batches.append((x_batch, y_batch))

    return batches

    raise NotImplementedError("请完成 mini-batch 划分")


if __name__ == "__main__":
    features = np.array(
        [[10, 0.1], [20, 0.2], [30, 0.3], [40, 0.4], [50, 0.5]],
        dtype=np.float32,
    )
    labels = np.array([1, 0, 2, 1, 2], dtype=np.int64)
    features_before = features.copy()
    labels_before = labels.copy()

    # 样例 1：保留尾批；最后一个 batch 只有一条样本，仍保留样本轴。
    batches = make_training_batches(features, labels, batch_size=2)
    assert isinstance(batches, list)
    assert len(batches) == 3
    assert [x.shape for x, _ in batches] == [(2, 2), (2, 2), (1, 2)]
    assert [y.shape for _, y in batches] == [(2,), (2,), (1,)]
    assert all(isinstance(batch, tuple) for batch in batches)
    assert all(x.dtype == features.dtype for x, _ in batches)
    assert all(y.dtype == labels.dtype for _, y in batches)
    np.testing.assert_array_equal(
        np.concatenate([x for x, _ in batches]), features_before
    )
    np.testing.assert_array_equal(
        np.concatenate([y for _, y in batches]), labels_before
    )
    np.testing.assert_array_equal(features, features_before)
    np.testing.assert_array_equal(labels, labels_before)

    # 样例 2：只丢弃不足一批的尾批；前两批的内容和对应关系保持不变。
    full_batches = make_training_batches(features, labels, batch_size=2, drop_last=True)
    assert isinstance(full_batches, list)
    assert len(full_batches) == 2
    assert [x.shape for x, _ in full_batches] == [(2, 2), (2, 2)]
    assert [y.shape for _, y in full_batches] == [(2,), (2,)]
    assert all(isinstance(batch, tuple) for batch in full_batches)
    assert all(x.dtype == features.dtype for x, _ in full_batches)
    assert all(y.dtype == labels.dtype for _, y in full_batches)
    np.testing.assert_array_equal(
        np.concatenate([x for x, _ in full_batches]), features_before[:4]
    )
    np.testing.assert_array_equal(
        np.concatenate([y for _, y in full_batches]), labels_before[:4]
    )
    np.testing.assert_array_equal(features, features_before)
    np.testing.assert_array_equal(labels, labels_before)

    print("Day 013：两个验收样例通过。")
