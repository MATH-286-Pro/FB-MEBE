import torch

def compute_entropy(
    x: torch.Tensor,
    range_dict: dict,
    bins: int = 20,
    eps: float = 1e-12,
):
    """
    Histogram-based joint entropy estimation for a batch of samples.
    """
    assert x.dim() == 2
    N, D = x.shape
    keys = list(range_dict.keys())

    # ---- discretize each dimension ----
    bin_indices = []
    for d in range(D):
        min_v, max_v = range_dict[keys[d]]
        idx = ((x[:, d] - min_v) / (max_v - min_v) * bins).long()
        idx = idx.clamp(0, bins - 1)
        bin_indices.append(idx)

    bin_indices = torch.stack(bin_indices, dim=1)  # (N, D)

    # ---- map multi-d index to single index ----
    multipliers = (bins ** torch.arange(D)).to(x.device)
    flat_idx = (bin_indices * multipliers).sum(dim=1)

    # ---- count histogram ----
    hist = torch.bincount(flat_idx, minlength=bins**D).float()

    prob = hist / hist.sum()

    entropy = -(prob * torch.log(prob + eps)).sum()

    return entropy.item()