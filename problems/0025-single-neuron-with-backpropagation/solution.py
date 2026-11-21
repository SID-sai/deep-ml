import torch
import torch.nn as nn
import torch.optim as optim

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    features = features.to(dtype=torch.float32)

    num_features = features.shape[1]

    labels = labels.to(dtype = torch.float32)
    linear = nn.Linear(num_features,1,bias = True)
    model = nn.Sequential(
        linear,
        nn.Sigmoid()
    )

    with torch.no_grad():
        linear.weight.copy_(initial_weights.to(dtype=torch.float32).view(1, num_features))
        linear.bias.fill_(float(initial_bias))

    loss_fun = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)

    msevalues = []

    labels = labels.to(dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(features)
        loss = loss_fun(output,labels)
        msevalues.append(loss.item())
        loss.backward()
        optimizer.step()


    # with torch.no_grad():
    #     updated_weights = [round(float(w), 4) for w in linear.weight.view(-1)]
    #     updated_bias = round(float(linear.bias.item()), 4)

    with torch.no_grad():
        updated_weights = [round(float(w), 4) for w in linear.weight.view(-1)]
        updated_bias = round(float(linear.bias.item()), 4)

    return updated_weights, updated_bias, msevalues