import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)

    A_det = torch.linalg.det(A_t)
    T_det = torch.linalg.det(T_t)
    S_det = torch.linalg.det(S_t)

    if T_det==0 and S_det==0:
        z = "not_invertable"
    else:
        z = "invertable"
    if z=="invertable":
        T_inv = torch.linalg.inv(T_t)
        w = torch.matmul(T_inv,A_t)
        return torch.matmul(w,S_t)

