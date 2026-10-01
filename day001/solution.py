def select_predictions(scores, threshold=0.5):
    selected = []
    for index, score in enumerate(scores):
        if score >= threshold:
            selected.append(index)
    return selected


# 样例 1：同时检查遍历、顺序和等于阈值的行为。
assert select_predictions([0.2, 0.8, 0.5, 0.9], 0.5) == [1, 2, 3]

# 样例 2：空输入，也使用默认阈值。
assert select_predictions([]) == []
