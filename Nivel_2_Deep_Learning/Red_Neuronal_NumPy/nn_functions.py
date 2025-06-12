# --- IMPORTS ---
import numpy as np


def initialize_parameters(n_x, n_h, n_y):  # (2,4,1)
    # Layer 1: weights and biases
    W1 = 0.01 * np.random.randn(n_h, n_x)  # (n_h, n_x)
    b1 = np.zeros((n_h, 1))                # (n_h, 1)

    # Layer 2: weights and biases
    W2 = 0.01 * np.random.randn(n_y, n_h)  # (n_y, n_h)
    b2 = np.zeros((n_y, 1))                # (n_y, 1)
    
    # Pack into dictionary
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }

    return parameters

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def forward_propagation(X, parameters):
    # Unpack parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Layer 1
    Z1 = W1 @ X + b1
    A1 = np.tanh(Z1)

    # Layer 2
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    # Store intermediate values
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2,
    }

    return A2, cache


def compute_cost(A2, Y):
    # Compute cross-entropy cost
    m = A2.shape[1]
    cost_sum = np.sum((Y * np.log(A2)) + ((1 - Y) * np.log(1 - A2)))
    cost = - (1 / m) * cost_sum
    return cost


def backward_propagation(parameters, cache, X, Y):
    # Unpack values
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    m = A1.shape[1]  # number of examples

    # Layer 2 gradients
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # Layer 1 gradients
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * (1 - A1**2)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    # Store gradients
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads

def update_parameters(parameters, grads, learning_rate):
    # Unpack current parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Unpack gradients
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update each parameter by taking a step in the opposite direction of the gradient
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    # Store updated parameters back in the dictionary
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    # Return updated parameters
    return parameters

def nn_model(X, Y, n_h, num_iterations, learning_rate):
    # Get the size of input and output layers
    n_x = X.shape[0]
    n_y = Y.shape[0]

    # Initialize parameters with given layer sizes
    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []
    # Loop over the number of iterations to train the model
    for i in range(num_iterations):
        # Forward propagation: compute predictions and cache intermediate values
        A2, cache = forward_propagation(X, parameters)

        # Compute the cost (error) between predictions and true labels
        cost = compute_cost(A2, Y)

        # Backward propagation: compute gradients for parameter updates
        grads = backward_propagation(parameters, cache, X, Y)

        # Update parameters using gradients and learning rate
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print cost every 1000 iterations to monitor training progress
        if i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost}")
            costs.append(cost)

    # Return the trained parameters after all iterations
    return parameters, costs


def predict(parameters, X):
    # Run forward pass to get probabilities
    A2, _ = forward_propagation(X, parameters)
    
    # Apply threshold: if prob > 0.5 → 1, else → 0
    predictions = (A2 > 0.5).astype(int)
    
    # Return binary predictions
    return predictions

