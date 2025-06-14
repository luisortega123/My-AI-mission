# --- IMPORTS ---
import numpy as np


def initialize_parameters(n_x, n_h1, n_h2, n_y):
    W1 = 0.01 * np.random.randn(n_h1, n_x)
    b1 = np.zeros((n_h1, 1))
    
    W2 = 0.01 * np.random.randn(n_h2, n_h1)
    b2 = np.zeros((n_h2, 1))
    
    W3 = 0.01 * np.random.randn(n_y, n_h2)
    b3 = np.zeros((n_y, 1))
    
    parameters = {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2,
        "W3": W3, "b3": b3
    }
    return parameters

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

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
        "Z3": Z3, "A3": A3
    }
    
    return A3, cache



def compute_cost(A2, Y):
    # Compute cross-entropy cost
    epsilon = 1e-8
    A2 = np.clip(A2, epsilon, 1 - epsilon)
    m = A2.shape[1]
    cost_sum = np.sum((Y * np.log(A2)) + ((1 - Y) * np.log(1 - A2)))
    cost = - (1 / m) * cost_sum
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    
    W2, W3 = parameters["W2"], parameters["W3"]
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]
    Z1, Z2 = cache["Z1"], cache["Z2"]
    
    dZ3 = A3 - Y                        # derivada salida sigmoid + cost
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
        "dW1": dW1, "db1": db1
    }
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    parameters["W3"] -= learning_rate * grads["dW3"]
    parameters["b3"] -= learning_rate * grads["db3"]
    
    return parameters
