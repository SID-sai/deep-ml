import torch

def matrix_rank(A: torch.Tensor, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n) as a torch.Tensor
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    # Your code here
    rank = torch.linalg.matrix_rank(A)
    return rank.item()