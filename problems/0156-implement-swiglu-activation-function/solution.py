import torch
import torch.nn.functional as F
def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor of shape (batch_size, 2d)
    Returns:
        torch.Tensor of shape (batch_size, d)
    """
    x1,x2 = x.chunk(2,dim=1)
    return x1*F.silu(x2)
    