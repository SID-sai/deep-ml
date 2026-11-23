import torch
from typing import Tuple

def momentum_optimizer(parameter: torch.Tensor, grad: torch.Tensor, velocity: torch.Tensor, 
                       learning_rate: float = 0.01, momentum: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update parameters using the momentum optimizer.
    Uses momentum to accelerate learning in relevant directions and dampen oscillations.
    Args:
        parameter: Current parameter value (torch.Tensor)
        grad: Current gradient (torch.Tensor)
        velocity: Current velocity/momentum term (torch.Tensor)
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)
    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
    parameter = parameter.to(dtype=torch.float32)
    grad = grad.to(dtype=torch.float32)
    velocity = velocity.to(dtype=torch.float32)

    v_new = momentum*velocity+learning_rate*grad
    para_new = parameter-v_new
    return torch.round(para_new,decimals=5), torch.round(v_new,decimals=5)