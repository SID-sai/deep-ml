import torch
import torch.nn.functional as F
def binary_cross_entropy(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-15) -> float:
    """
    Compute binary cross-entropy loss.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)
        epsilon: Small value for numerical stability
    
    Returns:
        Mean binary cross-entropy loss
    """
    # Your code here
    y_pred = torch.clamp(y_pred,min = epsilon,max = 1.0-epsilon)
    loss = F.binary_cross_entropy(y_pred,y_true)
    return loss.item()