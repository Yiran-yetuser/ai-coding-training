# 样例 1：不同长度、空序列，以及原始编号 0。
def pad_batch(sequences, pad_id=0):
    padded = []
    masks = []
    width = max((len(seq) for seq in sequences), default=0)

    for seq in sequences:
        # TODO：构造补齐后的新行，以及对应的 mask 行。
        mask = [1] * len(seq)
        new_seq = seq.copy()
        for i in range(len(seq), width):
            new_seq.append(pad_id)
            mask.append(0)
        # TODO：把两行分别加入 padded 和 masks。
        padded.append(new_seq)
        masks.append(mask)

    return padded, masks


sequences = [[8, 0], [3], []]
padded, masks = pad_batch(sequences)
assert padded == [[8, 0], [3, 0], [0, 0]]
assert masks == [[1, 1], [1, 0], [0, 0]]
assert sequences == [[8, 0], [3], []]

# 样例 2：没有任何序列。
assert pad_batch([]) == ([], [])
