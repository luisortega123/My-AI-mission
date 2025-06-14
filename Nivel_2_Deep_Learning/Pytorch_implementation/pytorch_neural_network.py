# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- LOAD DATA ---
X, Y = make_moons(n_samples=1000, noise=0.20)
Y = Y.reshape(-1, 1)
print(X.shape, Y.shape)      # Print shapes to confirm

X_tensor = torch.from_numpy(X).float()
Y_tensor = torch.from_numpy(Y).float() 
print(X_tensor.dtype, Y_tensor.dtype)  # Print data types for confirmation

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# --- MODEL PARAMETERS ---
input_size = 2      
hidden_size_1 = 16
hidden_size_2 = 8 
output_size = 1     

# Create an instance of the model 
model = NeuralNetwork(input_size, hidden_size_1, hidden_size_2, output_size)
print(model) 

# Define loss function and optimizer
Loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

# Prediction function using the trained model
def predict_pytorch(model, x):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        prediction = (probs >= 0.5).float()
    model.train()  # Return to training mode
    return prediction

# Function to plot decision boundary
def plot_decision_boundary(predict_fn, X, y):
    # Step 1: Create a mesh grid of points covering the feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01  # Grid resolution

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Step 2: Predict class for each point in the grid
    grid = np.c_[xx.ravel(), yy.ravel()]  # Shape (N, 2)
    grid_tensor = torch.from_numpy(grid).float()

    Z = predict_fn(grid_tensor).numpy()
    Z = Z.reshape(xx.shape)

    # Step 3: Plot decision boundary and original data points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    #plt.show()

# Create directory to save visualizations if it doesn't exist
os.makedirs("visualizations", exist_ok=True)

# --- TRAINING LOOP ---
epochs = 1500
losses = []  # List to store loss values every 100 epochs
epochs_to_loss_01 = None 
start_time = time.time()


for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    y_pred = model(X_tensor)  # Forward pass
    loss = Loss_function(y_pred, Y_tensor)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if epochs_to_loss_01 is None and loss.item() < 0.1:
        epochs_to_loss_01 = epoch
        print(f"¡Loss < 0.1 alcanzado en época {epoch}!")
    # Save and print loss every 100 epochs for monitoring
    if epoch % 100 == 0:
        losses.append(loss.item())
        print(f"Loss (epoch {epoch}): {loss.item():.4f}")

    # Save decision boundary visualization every 300 epochs
    if epoch % 300 == 0:
        plot_decision_boundary(lambda x: predict_pytorch(model, x), X_tensor.numpy(), Y_tensor.numpy())
        plt.title(f"Epoch {epoch}")
        plt.savefig(f"visualizations/epoch_{epoch}.png")
        plt.close()

end_time = time.time()
print(f"Epochs to reach loss < 0.1: {epochs_to_loss_01 if epochs_to_loss_01 else 'Not reached'}")

print(f"Training Time {end_time - start_time:.2f} seconds")

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel("Iterations (x100)")
plt.ylabel("Loss")
plt.title("Loss Function During Training")
plt.grid(True)
plt.savefig('visualizations/loss_curve.png')
plt.show()


plot_decision_boundary(lambda x: predict_pytorch(model, x), X_tensor.numpy(), Y_tensor.numpy())
plt.title('Final Decision Boundary')
plt.savefig('visualizations/final_decision_boundary.png')
plt.show()



