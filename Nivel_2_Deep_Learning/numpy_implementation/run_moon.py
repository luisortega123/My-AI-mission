# --- IMPORTS ---
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import time
import os

# --- INITIALIZE PARAMETERS FOR 3 LAYERS ---
def initialize_parameters(n_x, n_h1, n_h2, n_y):
    np.random.seed(42)

    W1 = np.random.randn(n_h1, n_x) * np.sqrt(2.0/n_x)
    b1 = np.zeros((n_h1, 1))

    W2 = np.random.randn(n_h2, n_h1) * np.sqrt(2.0/n_h1)
    b2 = np.zeros((n_h2, 1))

    # Para la última capa con sigmoid, usar Xavier
    W3 = np.random.randn(n_y, n_h2) * np.sqrt(1.0/n_h2)
    b3 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2,
        "W3": W3, "b3": b3,
    }
    return parameters

# --- ACTIVATION FUNCTIONS ---
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# --- FORWARD PROPAGATION ---
def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = W1 @ X + b1
    A1 = relu(Z1)

    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)

    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    cache = {
        "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2,
        "Z3": Z3, "A3": A3,
    }
    return A3, cache

# --- COST FUNCTION ---
def compute_cost(A3, Y):
    m = Y.shape[1]
    eps = 1e-15  # avoid log(0)
    A3 = np.clip(A3, eps, 1 - eps)
    cost = -np.sum(Y * np.log(A3) + (1 - Y) * np.log(1 - A3)) / m
    return cost

# --- BACKWARD PROPAGATION ---
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W2, W3 = parameters["W2"], parameters["W3"]
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]
    Z1, Z2 = cache["Z1"], cache["Z2"]

    dZ3 = A3 - Y
    dW3 = (1/m) * dZ3 @ A2.T
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = W3.T @ dZ3
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T @ dZ2
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW3": dW3, "db3": db3,
        "dW2": dW2, "db2": db2,
        "dW1": dW1, "db1": db1,
    }
    return grads

# --- UPDATE PARAMETERS ---
def update_parameters(parameters, grads, learning_rate):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    parameters["W3"] -= learning_rate * grads["dW3"]
    parameters["b3"] -= learning_rate * grads["db3"]
    return parameters

# --- PLOT DECISION BOUNDARY (modified to optionally save image) ---
def plot_decision_boundary(model, X, y, iteration=None, save_path=None):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[0, :], X[1, :], c=y[0, :], cmap=plt.cm.Spectral)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    title = "Decision Boundary"
    if iteration is not None:
        title += f" after iteration {iteration}"
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# --- TRAIN MODEL (with time measurement and saving decision boundaries) ---
def nn_model(X, Y, n_h1, n_h2, num_iterations, learning_rate):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    parameters = initialize_parameters(n_x, n_h1, n_h2, n_y)
    costs = []
    epochs_to_loss_01 = None 

    # Create directory for saving decision boundaries
    os.makedirs("decision_boundaries", exist_ok=True)

    for i in range(num_iterations):
        A3, cache = forward_propagation(X, parameters)
        cost = compute_cost(A3, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)

        if epochs_to_loss_01 is None and cost < 0.1:
            epochs_to_loss_01 = i
            print(f"Loss < 0.1 reached at epoch {i}!")

        # Save cost every 100 iterations for smoother plot
        if i % 100 == 0:
            costs.append(cost)
            print(f"Loss (epoch {i}): {cost:.4f}")
        # Save decision boundary plot every 300 iterations
        if i % 300 == 0:
            filename = f"decision_boundaries/decision_boundary_iter_{i}.png"
            plot_decision_boundary(lambda x: predict(parameters, x), X, Y, iteration=i, save_path=filename)
    
    print(f"Epochs for loss < 0.1: {epochs_to_loss_01 if epochs_to_loss_01 else 'Not reached'}")
    
    return parameters, costs

# ADD THIS PRINT AT THE END OF THE FUNCTION



# --- PREDICT ---
def predict(parameters, X):
    A3, _ = forward_propagation(X, parameters)
    predictions = (A3 > 0.5).astype(int)
    return predictions

# --- MAIN SCRIPT ---
if __name__ == "__main__":
    np.random.seed(42)
    X, Y = make_moons(n_samples=1000, noise=0.20)
    X = X.T
    Y = Y.reshape(1, -1)

    plt.scatter(X[0, :], X[1, :], c=Y[0, :], cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Scatter plot of make_moons dataset')
    plt.show()

    import time
    start_time = time.time()
    trained_parameters, costs = nn_model(X, Y, n_h1=16, n_h2=8, num_iterations=1500, learning_rate=0.5)
    end_time = time.time()

    print(f"Training time: {end_time - start_time:.4f} seconds")

    # Plot final decision boundary
    plot_decision_boundary(lambda x: predict(trained_parameters, x), X, Y)

    # Plot cost over iterations
    plt.figure()
    plt.plot(np.arange(0, 1500, 100), costs)  # x-axis matches costs saved every 100 iterations
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost reduction over time")
    plt.show()
