import torch
import torch.nn as nn
import torch.nn.functional as F

def relu(z: float) -> torch.Tensor:
    """
    Implements the ReLU activation function using PyTorch.
    
    Args:
        z: A float input value.
    
    Returns:
        A torch.Tensor with ReLU applied (max(0, z)).
    """
    z = torch.tensor(z,dtype = torch.float32)
    results = F.relu(z)
    return results
