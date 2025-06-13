# --- IMPORTS ---
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from train_pytorch import MyNeuralNetwork  # Import your custom neural network class

# --- USING THE BLUEPRINT TO BUILD THE MODEL ---
input_size = 2      # n_x: 2 input features (x and y from make_moons)
hidden_size = 4     # n_h: 4 neurons in the hidden layer (you can experiment with this)
output_size = 1     # n_y: 1 output neuron for binary classification (0 or 1)

# Create an instance of the model (call the class and pass sizes to __init__)
model = MyNeuralNetwork(n_x=input_size, n_h=hidden_size, n_y=output_size)
print(model)  # Print model structure

# --- LOAD DATA ---
X, Y = make_moons(n_samples=400, noise=0.20)  # Generate synthetic 2D classification data
Y = Y.reshape(-1, 1)                           # Reshape labels to be column vector
print(X.shape, Y.shape)                       # Print shapes to confirm

# Convert NumPy arrays to PyTorch tensors
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()
print(X.dtype, Y.dtype)  # Print data types for confirmation

# Rename variables for clarity (optional, but improves readability)
X_train = X
y_train = Y

# --- DEFINE LOSS FUNCTION AND OPTIMIZER ---
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with logits (for binary classification)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer with learning rate 0.01

# --- TRAINING LOOP ---
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()                # Step 1: Clear gradients from previous step
    y_pred = model(X_train)              # Step 2: Forward pass - get predictions
    loss = criterion(y_pred, y_train)    # Step 3: Compute loss
    loss.backward()                      # Step 4: Backpropagation - compute gradients
    optimizer.step()                     # Step 5: Update weights

    # Print loss every 100 epochs for monitoring
    if epoch % 100 == 0:
        print(f"Loss (epoch {epoch}): {loss.item():.4f}")


def predict(x):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        prediction = (probs >= 0.5).float()
    model.train()
    return prediction

def plot_decision_boundary(predict_fn, X, y):
    # Step 1: create a mesh grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01  # grid resolution

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Step 2: predict class for each point in the grid
    grid = np.c_[xx.ravel(), yy.ravel()]  # shape (N, 2)
    grid_tensor = torch.from_numpy(grid).float()

    Z = predict_fn(grid_tensor).numpy()
    Z = Z.reshape(xx.shape)

    # Step 3: plot decision boundary and data points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()


plot_decision_boundary(predict, X_train.numpy(), y_train.numpy())
