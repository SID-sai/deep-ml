import torch

def square_relu(x: torch.Tensor) -> dict:
    """
    Apply the Square ReLU activation function and compute its derivative.
    
    Args:
        x: Input torch.Tensor of any shape
    
    Returns:
        Dictionary with 'output' and 'derivative' as torch.Tensors
    """
    x = torch.as_tensor(x)
    float_dtype = x.dtype if x.is_floating_point() else torch.float32
    x  = x.to(dtype = float_dtype)
    relu_x = torch.clamp(x,min=0.0)
    results = relu_x**2
    derivative = 2.0*relu_x
    return {
        "output": torch.round(results, decimals=4),
        "derivative": torch.round(derivative, decimals=4),
    }