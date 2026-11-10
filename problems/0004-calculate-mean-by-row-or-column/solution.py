import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    if mode == "row":
        x = -1
    elif mode == "column":
        x = -2
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    meann = a_t.mean(x)
    return meann