import torch
import torch.nn.functional as F
def hard_sigmoid(x: float) -> float:
    """
    Implements the Hard Sigmoid activation function using PyTorch.
    Uses the Keras convention: 0.2*x + 0.5, clamped to [0, 1].

    Args:
        x (float): Input value

    Returns:
        float: The Hard Sigmoid of the input
    """
    # Your code here
    x = torch.tensor(x,dtype = torch.float32)
    results = torch.clamp(0.2*x+0.5,min=0.0,max=1.0)
    return results.item()
