import torch
import torch.nn.functional as F
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    scores = torch.tensor(scores,dtype = torch.float32)
    results = F.log_softmax(scores,dim=0)
    return results
    
