import torch
import torch.nn.functional as F

def leaky_relu(z: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    Implements the Leaky ReLU activation function using PyTorch.
    
    Args:
        z: Input tensor (scalar or any shape)
        alpha: Slope for negative values (default: 0.01)
    
    Returns:
        Output tensor after applying Leaky ReLU
    """
    # Your implementation here
    z = torch.tensor(z,dtype = torch.float32)
    results = F.leaky_relu(z,alpha)
    return results
