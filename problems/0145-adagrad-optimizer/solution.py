import torch
import torch.optim as optim
def adagrad_optimizer(parameter: torch.Tensor, grad: torch.Tensor, G: torch.Tensor, learning_rate: float = 0.01, epsilon: float = 1e-8) -> tuple:
    """
    Update parameters using the Adagrad optimizer.
    Adapts the learning rate for each parameter based on the historical gradients.
    Args:
        parameter: Current parameter value (torch.Tensor)
        grad: Current gradient (torch.Tensor)
        G: Accumulated squared gradients (torch.Tensor)
        learning_rate: Learning rate (default=0.01)
        epsilon: Small constant for numerical stability (default=1e-8)
    Returns:
        tuple: (updated_parameter, updated_G)
    """
    # Your code here
    updated_g = G+grad**2
    updated_parameter = parameter-(learning_rate/torch.sqrt(updated_g+epsilon))*grad
    return updated_parameter,updated_g