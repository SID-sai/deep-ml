import torch
import torch.nn.functional as F
import torch.nn as nn

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    n_features = len(weights)
    model = nn.Sequential(
        nn.Linear(n_features,1),
        nn.Sigmoid()
    )

    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([weights]))
        model[0].bias.copy_(torch.tensor([bias]))


    loss_fun = nn.MSELoss()

    x = torch.tensor(features,dtype = torch.float32)
    y = torch.tensor(labels, dtype = torch.float32)

    results = model(x).squeeze(1)
    loss = loss_fun(results,y)

    return ([round(p, 4) for p in results.tolist()], round(loss.item(), 4))

    
