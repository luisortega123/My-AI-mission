# --- IMPORTS ---
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from nn_functions import nn_model, predict

# --- SET RANDOM SEED FOR REPRODUCIBILITY ---
np.random.seed(42)

# --- LOAD DATA ---
X, Y = make_moons(n_samples=400, noise=0.20)  # Generate 2D binary classification data
X = X.T  # Transpose for consistency with the model's input format
Y = Y.reshape(1, -1)  # Reshape labels to a row vector
print(X.shape, Y.shape)

# --- PLOT DATA ---
plt.scatter(X[0, :], X[1, :], c=Y[0, :], cmap='viridis')  
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter plot of make_moons dataset')
plt.show()

# --- TRAIN THE NEURAL NETWORK ---
trained_parameters, costs = nn_model(X, Y, n_h=4, num_iterations=10000, learning_rate=1.2)

# --- PLOT DECISION BOUNDARY ---
def plot_decision_boundary(model, X, y):
    # Set plot range
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    
    # Create a grid of points with 0.01 spacing
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predict values for each point in the grid
    Z = model(np.c_[xx.ravel(), yy.ravel()].T)  # Flatten and stack into prediction input
    Z = Z.reshape(xx.shape)  # Reshape predictions back to match the grid shape
    
    # Draw the decision boundary and overlay the data points
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[0, :], X[1, :], c=y[0, :], cmap=plt.cm.Spectral)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title("Decision Boundary")
    plt.show()

# Pass a lambda to call predict with the trained parameters
plot_decision_boundary(lambda x: predict(trained_parameters, x), X, Y)

# --- PLOT COST FUNCTION ---
plt.figure()
plt.plot(costs)
plt.xlabel("Iterations (per thousands)")
plt.ylabel("Cost")
plt.title("Cost reduction over time")
plt.show()
