import torch
import torch.nn.functional as F
def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the Tanh (hyperbolic tangent) activation function.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: The tanh of the input, rounded to 4 decimal places
    """
    results = F.tanh(x)
    return results