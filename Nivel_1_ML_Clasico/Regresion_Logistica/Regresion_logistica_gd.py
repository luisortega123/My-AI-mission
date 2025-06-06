# ------ IMPORTS -------
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report



# =========================================
# --- DATA LOADING AND PREPARATION  ---
# =========================================

datos = load_breast_cancer()
X_process = datos.data
y = datos.target
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_process,y,test_size=0.2, random_state=42)

# Feature Scaling
epsilon = 1e-8
mu_train = np.mean(X_train, axis=0)
sigma_train = np.std(X_train, axis=0)

# Scale Training and Test Data
X_train_scaled = ((X_train - mu_train) / (sigma_train + epsilon))
X_test_scaled = ((X_test - mu_train) / (sigma_train + epsilon))

# Add Bias Term to SCALED Data
ones_train_scaled = np.ones((X_train_scaled.shape[0], 1))
X_train_bias_scaled = np.hstack((ones_train_scaled, X_train_scaled))

ones_test_scaled = np.ones((X_test_scaled.shape[0], 1))
X_test_bias_scaled = np.hstack((ones_test_scaled, X_test_scaled))

num_cols_logreg = X_train_bias_scaled.shape[1]
initial_theta = np.zeros((num_cols_logreg, 1))


print("Data preparation complete. All necessary train/test sets are ready.")


# Sigmoid function will give us a value between 0 and 1, which is useful for probabilities

def sigmoid(z):
    exp = np.exp(-z)
    resultado_sigmoide = 1 / (1 + exp)
    return resultado_sigmoide

# Value tests for sigmoid
# Dictionary to store the results
sigmoid_list = {}

# List for z values
z_values = [
    0,
    10,
    100,
    0.1,
    0.5,
    -100,
    -10,
    1000,
    -500,
]

for z in z_values:
    result = sigmoid(z)
    sigmoid_list[z] = result

# Print Results
for z, g_z in sigmoid_list.items():
    print(f"sigmoid({z}) = {g_z}")


# Hypothesis function hθ(x)
# We multiply the matrices
def calculate_hypothesis(X, theta):
    Z_vector = X @ theta
    Z_vector_prob = sigmoid(Z_vector)
    return Z_vector_prob


# Examples
# X_example: 2 observations, 1 feature + column of ones
X_example = np.array([[1.0, 5.0], [1.0, -2.0]])  # Observation 1  # Observation 2
# theta_example: for intercept and 1 feature
theta_example = np.array([[0.5], [-0.1]])  # theta_0  # theta_1

probabilities_example = calculate_hypothesis(X_example, theta_example)
print(f"Example probabilities:\n", probabilities_example)

# Cost function (Binary Cross-Entropy):
# J(θ) = -(1/m) * ∑[ y(i) * log(hθ(x(i))) + (1 - y(i)) * log(1 - hθ(x(i))) ]
def calculate_cost(X, y, theta, lambda_reg):
    epsilon = 1e-8  # A very small value for numerical stability
    m = X.shape[0]  # Number of observations

    # 1. Get predictions (probabilities)
    h_theta_X = calculate_hypothesis(X, theta)
    # 2. Clip predictions to avoid exact log(0) or log(1)
    h_theta_X_clipped = np.clip(h_theta_X, epsilon, 1 - epsilon)
    # 3. Calculate the two terms of the cross-entropy cost function
    # Term when y=1: y * log(h)
    product1 = y * np.log(h_theta_X_clipped)
    # Term when y=0: (1-y) * log(1-h)
    product2 = (1 - y) * np.log(1 - h_theta_X_clipped)
    # 4. Sum the terms for all observations
    sum_products = product1 + product2
    sum_total = np.sum(sum_products)
    # 5. Apply the factor -1/m
    original_cost = - (1 / m) * sum_total
    
    # Part 2: Calculate regularization term here directly
    theta_for_regularization = theta[1:]
    sum_squares_theta = np.sum(theta_for_regularization**2)
    regularization_term = (lambda_reg / (2 * m)) * sum_squares_theta

    total_cost = original_cost + regularization_term

    return total_cost
y_example = np.array([[1],  # True label for observation 1
                    [0]]) # True label for observation 2

example_cost = calculate_cost(X_example, y_example, theta_example, lambda_reg=0)
print(f"Calculated cost:\n", example_cost)

# Gradient Descent function
def gradient_descent(X, y, theta, alpha, num_iterations, lambda_reg):
    y = y.reshape(-1, 1)
    cost_history = []  # List to store cost at each iteration (optional)
    m = X.shape[0]
    
    for i in range(num_iterations):
        # 1. Calculate predictions (h_theta)
        predictions = calculate_hypothesis(X, theta)

        # 2. Compute errors (predictions - y)
        errors = predictions - y

        # 3. Compute the vectorized gradient: (1/m) * X^T * errors
        #    (You need the transpose of X_bias: X_bias.T)
        original_gradient = (1 / m) * X.T @ errors

        # Add regularization term
        theta_penalty = theta.copy()
        theta_penalty[0] = 0  # Do not regularize the bias term
        gradient_penalty = theta_penalty * (lambda_reg / m)
        regularized_gradient = original_gradient + gradient_penalty

        # 4. Update rule for theta: theta = theta - alpha * gradient
        theta = theta - alpha * regularized_gradient

        # 5. Compute and store the cost (optional)
        current_cost = calculate_cost(X, y, theta, lambda_reg)
        cost_history.append(current_cost)

        return theta, cost_history

    # Return the fi

alpha = 0.1
num_iterations = 5000

# Tests with various alphas to determine which one is best
results_history = {}
test_alphas = [
    0.3,
    0.1,
    0.03,
    0.01,
    0.001,
]

for alpha in test_alphas:
    theta_result, cost_history = gradient_descent(X_train_bias_scaled, y_train, initial_theta, alpha, num_iterations, lambda_reg=0)
    results_history[alpha] = cost_history

for alpha, cost_history in results_history.items():
    plt.plot(cost_history, label=f"alpha = {alpha}")

theta_final_logistic, cost_history_logistic = gradient_descent(X_train_bias_scaled, y_train, initial_theta, alpha, num_iterations, lambda_reg=0)

print("Final theta:", theta_final_logistic)
print("Cost history:", cost_history_logistic)

plt.plot(cost_history_logistic)
plt.title("Gradient Descent Convergence for Logistic Regression")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost J(θ)")
plt.show()


# Predict function
def predict(X, theta):
    # Calculate probabilities using the existing hypothesis function
    Z_predict_prob = calculate_hypothesis(X, theta)
    # Apply standard threshold: >= 0.5 predicts class 1, < 0.5 predicts class 0
    predictions = (Z_predict_prob >= 0.5).astype(int)
    return predictions

# Accuracy
y_predictions = predict(X_train_bias_scaled, theta_final_logistic)
correct = y_train == y_predictions
total_correct = np.sum(correct)
print(f"Correct predictions: {total_correct} out of {len(y)}")

percentage = (total_correct / len(y)) * 100
print(f"Accuracy percentage: {percentage:,.2f}%")

# Alternative
# --- Calculate accuracy with the BEST theta ---
print("Calculating Accuracy...")
y_predictions = predict(X_train_bias_scaled, theta_final_logistic)
accuracy = np.mean(y_train == y_predictions)  # Concise way
print(f"Model accuracy: {accuracy * 100:.2f}%")
num_iterations_test = 100
alpha_test = 0.1

lambda_dict = {}
lambda_values = [
    0,
    0.1,
    0.01,
    0.001,
    1,
    10,
    100,
    1000,
]

for lambda_reg in lambda_values:
    print(f"--- Processing lambda = {lambda_reg} ---")
    theta_calculated, cost_history = gradient_descent(
        X_train_bias_scaled, y_train, initial_theta.copy(), alpha_test, num_iterations_test, lambda_reg
    )
    lambda_dict[lambda_reg] = (theta_calculated, cost_history)

for lambda_reg, results_tuple in lambda_dict.items():
    theta_calculated = results_tuple[0]
    print(f"lambda({lambda_reg}) = {results_tuple}")
    print(theta_calculated.T)

    plt.plot(results_tuple[1], label=f"Cost for lambda = {lambda_reg}")

plt.xlabel("Iterations")
plt.ylabel("Cost J(θ)")
plt.title("Cost Curves by Lambda")
plt.legend()
plt.show()

# Initialize lists to store theta values for each coefficient
theta1_values = []
theta2_values = []
theta3_values = []
theta4_values = []
theta5_values = []
theta6_values = []
theta7_values = []
theta8_values = []

for specific_lambda in lambda_values:
    theta_calculated, cost_history = lambda_dict[specific_lambda]
    
    theta1_values.append(theta_calculated[1, 0])
    theta2_values.append(theta_calculated[2, 0])
    theta3_values.append(theta_calculated[3, 0])
    theta4_values.append(theta_calculated[4, 0])
    theta5_values.append(theta_calculated[5, 0])
    theta6_values.append(theta_calculated[6, 0])
    theta7_values.append(theta_calculated[7, 0])
    theta8_values.append(theta_calculated[8, 0])

plt.plot(lambda_values[1:], theta1_values[1:], label=r'$\theta_1$')
plt.plot(lambda_values[1:], theta2_values[1:], label=r'$\theta_2$')
plt.plot(lambda_values[1:], theta3_values[1:], label=r'$\theta_3$')
plt.plot(lambda_values[1:], theta4_values[1:], label=r'$\theta_4$')
plt.plot(lambda_values[1:], theta5_values[1:], label=r'$\theta_5$')
plt.plot(lambda_values[1:], theta6_values[1:], label=r'$\theta_6$')
plt.plot(lambda_values[1:], theta7_values[1:], label=r'$\theta_7$')
plt.plot(lambda_values[1:], theta8_values[1:], label=r'$\theta_8$')

plt.xlabel(r'$\lambda$', fontsize=12)
plt.ylabel(r'Value of coefficients $\theta_j$', fontsize=12)
plt.title(r'Coefficients $\theta_1$ to $\theta_8$ as a function of $\lambda$', fontsize=14)

plt.legend()      # Show legend with labels
plt.grid(True)    # Grid for better visualization
plt.tight_layout()
plt.xscale('log')
plt.show()

# =================================================================
# --- TRAINING AND ANALYSIS WITH GRADIENT DESCENT ---
# =================================================================

num_cols_logreg = X_train_bias_scaled.shape[1]
initial_theta = np.zeros((num_cols_logreg, 1))

alpha = 0.1
num_iterations = 2500
lambda_reg = 0.01

trained_theta_logreg , cost_history_logreg = gradient_descent(X_train_bias_scaled, y_train, initial_theta, alpha, num_iterations, lambda_reg)

# make predictions
y_pred_train_logreg = calculate_hypothesis(X_train_bias_scaled, trained_theta_logreg)
y_pred_test_logreg = calculate_hypothesis(X_test_bias_scaled, trained_theta_logreg)

# Classes for the training set
y_class_pred_train_logreg = (y_pred_train_logreg >= 0.5).astype(int)
y_class_pred_test_logreg = (y_pred_test_logreg >= 0.5 ).astype(int)

# Accuracy score test
accuracy_logreg_test = accuracy_score(y_test, y_class_pred_test_logreg) 
# Accuracy score train
accuracy_logreg_train = accuracy_score(y_train, y_class_pred_train_logreg) 

# Confusion score test
conf_matrix_logreg_test = confusion_matrix(y_test, y_class_pred_test_logreg)
# Confusion matrix train 
conf_matrix_logreg_train = confusion_matrix(y_train, y_class_pred_train_logreg)

# Precision score test
precision_logreg_test = precision_score(y_test, y_class_pred_test_logreg)
# Precision score train
precision_logreg_train = precision_score(y_train, y_class_pred_train_logreg)

# Recall score test
recall_logreg_test = recall_score(y_test, y_class_pred_test_logreg)
# Recall score train
recall_logreg_train = recall_score(y_train, y_class_pred_train_logreg)

# f1 score test 
f1_logreg_test = f1_score(y_test, y_class_pred_test_logreg)
# F1 score train
f1_logreg_train = f1_score(y_train, y_class_pred_train_logreg)



# ====================================
# --- Advanced Metrics for SVC ---
# ====================================
print("\n--- Calculate evaluation metrics ---")

print("--- Accuracy_logreg_test (Test Set) ---")
print(f"Accuracy Score:\n{accuracy_logreg_test:.4f}")
print("--- Accuracy_logreg_train (Train Set) ---")
print(f"Accuracy Score:\n{accuracy_logreg_train:.4f}")

print("--- Matriz de Confusión (Test Set) ---")
print(f"Confusion Matrix:\n{conf_matrix_logreg_test}")
print("--- Matriz de Confusión (Train Set) ---")
print(f"Confusion Matrix:\n{conf_matrix_logreg_train}")


print("--- Precision Score (Test Set) ---")
print(f"Precision Score: {precision_logreg_test:.4f}")
print("--- Precision Score (Train Set) ---")
print(f"Precision Score: {precision_logreg_train:.4f}")

print("--- Recall Score (Test Set) ---")
print(f"Recall Score: {recall_logreg_test:.4f}")
print("--- Recall Score (Train Set) ---")
print(f"Recall Score: {recall_logreg_train:.4f}")

print("--- F1 Score (Test Set) ---")
print(f"F1 Score: {f1_logreg_test:.4f}")
print("--- F1 Score (Train Set) ---")
print(f"F1 Score: {f1_logreg_train:.4f}")

# =========================================================
# --- Classification Report for Logistic Regression ---
# =========================================================

print("\n---  Classification Report (Test Set) ---")
cl_report_test = classification_report(y_test, y_class_pred_test_logreg)
print(cl_report_test)

print("\n---  Clasiification Report (Train Set) ---")
cl_report_train = classification_report(y_train, y_class_pred_train_logreg)
print(cl_report_train)
