import torch


def tensors_equal(a: torch.Tensor, b: torch.Tensor, rtol=1e-5, atol=1e-8) -> bool:
    """True if a and b have the same shape and match within floating-point tolerance."""
    return a.shape == b.shape and torch.allclose(a, b, rtol=rtol, atol=atol)
