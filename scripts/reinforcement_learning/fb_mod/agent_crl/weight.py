import torch
import torch.nn as nn
from . import fb_meta_module as meta

def weight_init(m):

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, meta.DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()

def parallel_orthogonal_(tensor, gain: float = 1.0):
    """Parallel orthogonal initialization for DenseParallel layers"""
    if tensor.ndimension() == 2:
        return nn.init.orthogonal_(tensor, gain=gain)
    
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")
    
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()

        # Compute the qr factorization
        q, r = torch.linalg.qr(flat_tensor)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor