import torch
import torch.nn.functional as F

def selu(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the SELU (Scaled Exponential Linear Unit) activation function.

    Args:
        x: Input tensor

    Returns:
        SELU activation tensor
    """
    alpha = 1.6732632423543772
    scale = 1.0507009873554804
    # Your code here
    results = F.selu(x)
    return results