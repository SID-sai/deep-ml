import torch
import torch.nn.functional as F

def softplus(x: float) -> torch.Tensor:
    """
    Compute the softplus activation function using PyTorch.

    Args:
        x: Input value

    Returns:
        The softplus value: log(1 + e^x) as a torch.Tensor
    """
    # Your code here
    x = torch.tensor(x,dtype = torch.float32)
    results = F.softplus(x)
    return results