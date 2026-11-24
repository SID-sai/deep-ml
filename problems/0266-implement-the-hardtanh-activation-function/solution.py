import torch
import torch.nn.functional as F
def hardtanh(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    """
    Compute the Hardtanh activation function.

    Args:
        x: Input tensor
        min_val: Minimum value for the output range (default: -1.0)
        max_val: Maximum value for the output range (default: 1.0)

    Returns:
        The Hardtanh value clipped to [min_val, max_val]
    """
    results = F.hardtanh(x,min_val = min_val,max_val=max_val)
    return results