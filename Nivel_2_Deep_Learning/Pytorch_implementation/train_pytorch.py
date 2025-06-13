import torch
import torch.nn as nn

# --- DEFINITION OF THE 'BLUEPRINT' (Neural Network Architecture) ---
class MyNeuralNetwork(nn.Module):
    def __init__(self, n_x, n_h, n_y):
        super().__init__()
        # Define linear layers
        self.layer_1 = nn.Linear(n_x, n_h)     # First linear transformation: input to hidden
        self.activation_1 = nn.Tanh()          # Non-linear activation function (Tanh)
        self.layer_2 = nn.Linear(n_h, n_y)     # Second linear transformation: hidden to output

    def forward(self, x):
        # Forward pass: input -> hidden layer with activation -> output
        x = self.activation_1(self.layer_1(x))
        x = self.layer_2(x)
        return x
