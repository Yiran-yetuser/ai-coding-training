"""Day 010 starter: mean-pool valid Transformer token states."""

import torch


def masked_mean_pool(hidden_states, attention_mask):
    """Pool (B, T, H) token states into (B, H) sentence vectors."""
    # TODO: Align the token mask with hidden_states and exclude padding tokens.
    B, T, H = hidden_states.shape
    mask = attention_mask.unsqueeze(-1).to(
        hidden_states
    )  # (B, T, 1), 对齐关键步骤！否则无法广播！
    # TODO: Average along the token dimension; an all-padding row returns zeros.
    pooled = torch.sum(hidden_states * mask, dim=1)  # (B, H)
    pooled = pooled / torch.sum(mask, dim=1).clamp(min=1)  # (B, H)
    return pooled
    raise NotImplementedError


# Sample 1: padding is excluded; the second sequence has one valid token.
hidden_states = torch.tensor(
    [
        [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
        [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
    ]
)
mask = torch.tensor([[1, 1, 0], [0, 1, 0]])
hidden_before = hidden_states.clone()
pooled = masked_mean_pool(hidden_states, mask)
assert pooled.shape == (2, 2)
assert pooled.dtype == hidden_states.dtype
assert pooled.device == hidden_states.device
torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0], [6.0, 8.0]]))
torch.testing.assert_close(hidden_states, hidden_before)

# Sample 2: an all-padding row returns a zero vector.
all_padding = torch.zeros((1, 3), dtype=torch.bool)
all_padding_result = masked_mean_pool(torch.ones((1, 3, 2)), all_padding)
torch.testing.assert_close(all_padding_result, torch.zeros((1, 2)))
