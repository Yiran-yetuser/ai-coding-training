"""Day 011 starter: debug per-sample top-k classification predictions."""

import torch


def topk_predictions(scores, k):
    """Return top-k scores and their class indices for each batch row."""
    # TODO: Debug the axis used to select classes independently per sample.
    top_scores, class_ids = torch.topk(scores, k=k, dim=1)
    return top_scores, class_ids


# Sample 1: each row is a separate sample; select two class scores per row.
scores = torch.tensor([[0.1, 0.9, 0.4, 0.3], [0.8, 0.2, 0.7, 0.1]])
scores_before = scores.clone()
top_scores, class_ids = topk_predictions(scores, k=2)
assert top_scores.shape == class_ids.shape == (2, 2)
torch.testing.assert_close(top_scores, torch.tensor([[0.9, 0.4], [0.8, 0.7]]))
assert class_ids.tolist() == [[1, 2], [0, 2]]
torch.testing.assert_close(scores, scores_before)

# Sample 2: k=1 keeps both dimensions.
one_score, one_id = topk_predictions(torch.tensor([[0.2, 0.5, 0.3]]), k=1)
assert one_score.shape == one_id.shape == (1, 1)
assert one_id.item() == 1
