# --------- IMPORTS ---------
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
# =================================================================
# --- 1. DATA LOADING AND PREPARATION  ---
# =================================================================
print("Loading and preparing data...")
california_data = fetch_california_housing()
X_orig = california_data.data  # Original features
y_orig = california_data.target.reshape(-1, 1) # Original target, reshaped

# 1a. Split data (ONLY ONCE)
X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)
print(f"Shapes: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

# 1b. Feature Scaling Parameters (from TRAINING data ONLY)
mu_train = np.mean(X_train, axis=0)
sigma_train = np.std(X_train, axis=0)
epsilon = 1e-8

# 1c. Scale Training and Test Data
X_train_scaled = (X_train - mu_train) / (sigma_train + epsilon)
X_test_scaled = (X_test - mu_train) / (sigma_train + epsilon)

# 1d. Add Bias Term to SCALED Data
unos_train_scaled = np.ones((X_train_scaled.shape[0], 1))
X_train_bias_scaled = np.hstack((unos_train_scaled, X_train_scaled))

unos_test_scaled = np.ones((X_test_scaled.shape[0], 1))
X_test_bias_scaled = np.hstack((unos_test_scaled, X_test_scaled))

# 1e. Add Bias Term to UNSCALED Data (for Normal Equation, if using unscaled)
unos_train_unscaled = np.ones((X_train.shape[0], 1)) # Uses X_train (original, unscaled)
X_train_bias_unscaled = np.hstack((unos_train_unscaled, X_train))

unos_test_unscaled = np.ones((X_test.shape[0], 1))   # Uses X_test (original, unscaled)
X_test_bias_unscaled = np.hstack((unos_test_unscaled, X_test))

print("Data preparation complete. All necessary train/test sets (scaled/unscaled, with bias) are ready.")

# --- CORE GRADIENT DESCENT FUNCTIONS ---
def calculate_hypothesis(X_b, current_theta): # Nombres genéricos
    return X_b @ current_theta

def calculate_cost(X_b, y_target, current_theta, lmbda_reg): # Nombres genéricos
    m = X_b.shape[0]
    # --- Part 1: Original Cost (MSE) ---
    predictions = calculate_hypothesis(X_b, current_theta)
    errors = predictions - y_target
    squared_errors = errors**2
    sum_squared_errors = np.sum(squared_errors)
    original_cost = (1 / (2 * m)) * sum_squared_errors

    # Part 2: Directly calculating the Regularization term here
    theta_for_reg = current_theta[1:]  # Exclude theta_0
    sum_squared_theta_reg = np.sum(theta_for_reg**2)
    regularization_term = (lmbda_reg / (2 * m)) * sum_squared_theta_reg

    # --- Part 3: Total Regularized Cost ---
    total_cost = original_cost + regularization_term
    return total_cost

def gradient_descent(X_train_bias_scaled, y_train, initial_theta, alpha, num_iterations, lmbda_reg):
    y_reshaped_train = y_train.reshape(-1, 1)  # Ensure y is a column vector
    m = X_train_bias_scaled.shape[0]  # Number of samples
    theta_train = initial_theta.copy()  # Start with initial_theta (copy to avoid modifying the original)
    cost_history = []  # List to store the cost in each iteration (optional)
    
    for i in range(num_iterations):
        # Iteration starts here
        # 1. Calculate Predictions
        predictions = calculate_hypothesis(X_train_bias_scaled, theta_train)  

        
        # 2. Calculate Errors
        errors = predictions - y_reshaped_train

        # 3. Calculate Original Gradient
        original_gradient = (1 / m) * X_train_bias_scaled.T @ errors

        # --- Gradient Regularization ---
        theta_for_penalty = theta_train.copy()  
        theta_for_penalty[0] = 0 # Do not penalize theta_0
        gradient_penalty = (lmbda_reg / m) * theta_for_penalty
        regularized_gradient = original_gradient + gradient_penalty
        
        # 4. Update Theta
        theta_train = theta_train - alpha * regularized_gradient

        
        # 5. Calculate and store the cost
        current_cost = calculate_cost(X_train_bias_scaled, y_reshaped_train, theta_train, lmbda_reg)  # uses 'theta' (already updated)
        cost_history.append(current_cost)
        
    return theta_train, cost_history


# =================================================================
# --- TRAINING AND ANALYSIS WITH GRADIENT DESCENT ---
# =================================================================

# 1. Initialize theta (zero vector with shape (n+1, 1))
# n_features = X_orig.shape[1]  # Number of original features
#initial_theta = np.zeros((n_features + 1, 1))  # Use np.zeros with the correct shape
num_columns_with_bias = X_train_bias_scaled.shape[1]
initial_theta = np.zeros(( num_columns_with_bias, 1))

# Train on the training set, evaluate on the test set
alpha = 0.1
num_iterations = 2500
lmbda_reg = 0.01

# Call your function Gradient Descent
trained_theta_gd, cost_history_gd = gradient_descent(X_train_bias_scaled, y_train, initial_theta, alpha, num_iterations, lmbda_reg)

# Next step is to make predictions
y_pred_train_gd = calculate_hypothesis(X_train_bias_scaled, trained_theta_gd)
y_pred_test_gd = calculate_hypothesis(X_test_bias_scaled, trained_theta_gd)

# ------------ Calculate evaluation metrics for the gradient descent model ------------
print("\n--- Calculate evaluation metrics for the gradient descent model ---")

print("--- Calculate Mean Squared Error (MSE)... ---")
# Calculate Mean Squared Error (MSE)
mse_train_gd = mean_squared_error(y_train, y_pred_train_gd)
mse_test_gd = mean_squared_error(y_test, y_pred_test_gd)

print("--- Calculate Mean Absolute Error (MAE)... ---")
# Calculate Mean Absolute Error (MAE)
mae_train_gd = mean_absolute_error(y_train, y_pred_train_gd)
mae_test_gd = mean_absolute_error(y_test, y_pred_test_gd)

print("--- Calculate R-squared (R²)... ---")
# Calculate R-squared (R²)
r2_train_gd = r2_score(y_train, y_pred_train_gd)
r2_test_gd = r2_score(y_test, y_pred_test_gd)

print("\n--- Final evaluation metrics for the gradient descent model ---")
# Import results for the gradient descent model
print(f"MSE Train: {mse_train_gd:.4f}, MSE Test: {mse_test_gd:.4f}")
print(f"MAE Train: {mae_train_gd:.4f}, MAE Test: {mae_test_gd:.4f}")
print(f"R2 Train: {r2_train_gd:.4f}, R2 Test: {r2_test_gd:.4f}")


# =================================================
# ------------------ TESTING ------------------
# =================================================
# 2. Comparison of Learning Rates (Alpha)
print("\n--- Comparing Learning Rates (Alpha) ---")
num_iterations_alpha_comp = 2500
test_alphas = [0.3, 0.1, 0.03, 0.01, 0.001]
alpha_history_results = {}

for alpha_test_val in test_alphas:
    print(f"Processing Alpha = {alpha_test_val}...")
    _, alpha_cost_history = gradient_descent(
        X_train_bias_scaled, y_train, initial_theta.copy(), alpha_test_val, num_iterations_alpha_comp, lmbda_reg=0
    )
    alpha_history_results[alpha_test_val] = alpha_cost_history

plt.figure(figsize=(10, 6))
for alpha_val, plot_cost_history in alpha_history_results.items():
    plt.plot(plot_cost_history, label=f"Alpha = {alpha_val}")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Cost J(θ)")
plt.title("Comparison of Convergence by Learning Rate (Scaled Data)")
plt.grid(True)
plt.show()

# 3. Comparison of Gradient Descent WITH vs. WITHOUT Scaling
print("\n--- Comparing GD WITH vs. WITHOUT Feature Scaling ---")
optimal_alpha = 0.1  # Alpha selected as optimal in the original script
num_iterations_scale_comp = 2500 # From original script
base_initial_theta_comp = np.zeros((X_orig.shape[1] + 1, 1))

print("Running GD with SCALED data...")
final_theta_scaled_gd, gd_scaled_cost_history = gradient_descent(
    X_train_bias_scaled, y_train, base_initial_theta_comp.copy(), optimal_alpha, num_iterations_scale_comp, lmbda_reg=0
)
print("Done.")

print("Running GD with UNSCALED data...")
print("WARNING: GD on unscaled data with a high alpha (e.g., 0.1) might diverge or be very slow.")
_, gd_unscaled_cost_history = gradient_descent(
    X_train_bias_scaled, y_train, base_initial_theta_comp.copy(), optimal_alpha, num_iterations_scale_comp, lmbda_reg=0
)
print("Done.")

plt.figure(figsize=(10, 6))
plt.plot(gd_scaled_cost_history, label="With Scaling (GD)")
plt.plot(gd_unscaled_cost_history, label="Without Scaling (GD)")
plt.xlabel("Iterations")
plt.ylabel("Cost J(θ)")
plt.title("GD Convergence with vs. Without Feature Scaling")
plt.legend()
plt.grid(True)
# plt.ylim(top=np.min(gd_scaled_cost_history)*10) # Optional ylim from original, adjust as needed
plt.show()

# 4. Final Gradient Descent run to get final_theta
# Using optimal_alpha and num_iterations_scale_comp defined previously.
# The original script had alpha=0.01, num_iterations=2500 for one run,
# but optimal_alpha=0.1 was used later. I will use optimal_alpha.
alpha_for_final_gd = optimal_alpha 
num_iter_for_final_gd = num_iterations_scale_comp

print(f"\n--- Final Model Training with Gradient Descent (Scaled Data) ---")
print(f"Alpha: {alpha_for_final_gd}, Iterations: {num_iter_for_final_gd}, Lambda: 0")
final_theta, final_gd_cost_history = gradient_descent(
    X_train_bias_scaled, y_train, initial_theta.copy(), alpha_for_final_gd, num_iter_for_final_gd, lmbda_reg=0
)
print("Final theta (Gradient Descent, scaled data):")
print(final_theta.T)

plt.figure(figsize=(10, 6))
plt.plot(final_gd_cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost J(θ)")
plt.title("Final Gradient Descent Convergence (Scaled Data)")
plt.grid(True)
plt.show()

# 5. Prediction Function
def predict(X_new, theta_to_use, calculated_mu, calculated_sigma):
    epsilon_pred = 1e-8
    X_new_scaled = (X_new - calculated_mu) / (calculated_sigma + epsilon_pred)
    ones_pred = np.ones((X_new_scaled.shape[0], 1))
    X_new_Bias = np.hstack((ones_pred, X_new_scaled))
    predictions = X_new_Bias @ theta_to_use
    return predictions

# Example usage of the predict function (optional):
if X_orig.shape[0] > 5:
    samples_to_predict = X_orig[:5] 
    example_predictions = predict(samples_to_predict, final_theta, mu_train, sigma_train)
    print("\nExample predictions for the first 5 samples (using final_theta from GD):")
    for i in range(len(example_predictions)):
        print(f"  Prediction: {example_predictions[i][0]:.2f}, Actual: {y_orig[i][0]:.2f}")

# 6. Analysis of Lambda Regularization Parameter
print("\n--- Analyzing the Effect of Lambda (Regularization) with Gradient Descent ---")
num_iterations_lambda = 100  # Iterations for lambda analysis (from original script)
alpha_lambda = 0.1            # Alpha for lambda analysis (from original script)
lambda_values = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000] # Lambda values to test
lambda_results_list = {}

for lmbda_reg_val in lambda_values:
    print(f"--- Processing lambda = {lmbda_reg_val} ---")
    calculated_theta_lambda, lambda_cost_history = gradient_descent(
        X_train_bias_scaled, y_train, initial_theta.copy(), alpha_lambda, num_iterations_lambda, lmbda_reg_val
    )
    lambda_results_list[lmbda_reg_val] = (calculated_theta_lambda, lambda_cost_history)

plt.figure(figsize=(10, 6))
for lmbda_reg_plot, plot_results_tuple in lambda_results_list.items():
    plt.plot(plot_results_tuple[1], label=f"Cost for lambda = {lmbda_reg_plot}")
plt.xlabel("Iterations")
plt.ylabel("Cost J(θ)")
plt.title("Cost Curves by Lambda (Gradient Descent)")
plt.legend()
plt.grid(True)
plt.show()

# Plotting theta coefficients vs. lambda
print("\n--- Plotting Theta Coefficients vs. Lambda ---")
# Initialize lists to store the values of each theta_j
theta_coeffs_values = [[] for _ in range(initial_theta.shape[0])]

for specific_lambda_val in lambda_values:
    plot_calculated_theta_lambda, _ = lambda_results_list[specific_lambda_val]
    for i in range(plot_calculated_theta_lambda.shape[0]):
        theta_coeffs_values[i].append(plot_calculated_theta_lambda[i, 0])

plt.figure(figsize=(12, 7))
# Plot theta_1 to theta_n (excluding theta_0 as in the original script's specific lists)
for i in range(1, len(theta_coeffs_values)): 
    plt.plot(lambda_values, theta_coeffs_values[i], marker='o', linestyle='-', label=fr'$\theta_{i}$')

plt.xlabel(r'$\lambda$ (Regularization Parameter)', fontsize=12)
plt.ylabel(r'Value of coefficients $\theta_j$', fontsize=12)
plt.title(r'Coefficients $\theta_j$ as a function of $\lambda$ (Gradient Descent)', fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.02)) # Adjust legend position
plt.grid(True)
plt.xscale('log')  # Logarithmic scale for lambda (as in original)
plt.tight_layout()
plt.show()

print("\nGradient Descent script completed.")

# ================================================
#  ----------- NORMAL EQUATION -----------
# ================================================
print("\n" + "="*60)
print("                 STARTING NORMAL EQUATION METHOD")
print("="*60 + "\n")



def calculate_normal_equation(X, y):
    Xt = X.T  # Transpose of X
    XtX = Xt @ X  # Compute X^T * X

    try:
        inv_XtX = np.linalg.inv(XtX)  # Try standard inverse
    except np.linalg.LinAlgError:
        print("Warning: XtX is not invertible. Using pseudo-inverse instead.")
        inv_XtX = np.linalg.pinv(XtX)  # Use pseudo-inverse if singular

    Xty = Xt @ y  # Compute X^T * y
    theta = inv_XtX @ Xty  # Compute theta
    return theta

# =================================================================
# --- TRAINING AND ANALYSIS WITH NORAML EQUATION ---
# =================================================================

trained_theta_ne = calculate_normal_equation(X_train_bias_unscaled, y_train)
y_pred_train_ne = calculate_hypothesis(X_train_bias_unscaled, trained_theta_ne)
y_pred_test_ne = calculate_hypothesis(X_test_bias_unscaled, trained_theta_ne)


# ------------ Calculate evaluation metrics for the Normal Equation model ------------
print("\n--- Calculate evaluation metrics for the Normal Equation model ---")

print("--- Calculate Mean Squared Error (MSE)... ---")
# Calculate Mean Squared Error (MSE)
mse_train_ne = mean_squared_error(y_train, y_pred_train_ne)
mse_test_ne = mean_squared_error(y_test, y_pred_test_ne)

print("--- Calculate Mean Absolute Error (MAE)... ---")
# Calculate Mean Absolute Error (MAE)
mae_train_ne = mean_absolute_error(y_train, y_pred_train_ne)
mae_test_ne = mean_absolute_error(y_test, y_pred_test_ne)

print("--- Calculate R-squared (R²)... ---")
# Calculate R-squared (R²)
r2_train_ne = r2_score(y_train, y_pred_train_ne)
r2_test_ne = r2_score(y_test, y_pred_test_ne)


print("\n--- Final evaluation metrics for the Normal Equation model ---")
# Import results for the gradient descent model
print(f"MSE Train: {mse_train_ne:.7f}, MSE Test: {mse_test_ne:.7f}")
print(f"MAE Train: {mae_train_ne:.7f}, MAE Test: {mae_test_ne:.7f}")
print(f"R2 Train: {r2_train_ne:.7f}, R2 Test: {r2_test_ne:.7f}")

# Calculate theta using the Normal Equation with X
theta_calculated_normal = calculate_normal_equation(X_train_bias_unscaled, y_train)

print(f"\nTheta calculated by Normal Equation:\n{theta_calculated_normal}")
# Calculate theta using the Normal Equation with UNSCALED X
theta_calculated_normal = calculate_normal_equation(X_train_bias_unscaled, y_train)
print(f"Theta calculated by Normal Equation (unscaled):\n{theta_calculated_normal}")
print("Theta GD:", final_theta.T)
print("Theta NE:", theta_calculated_normal.T)  # This is NE with UNSCALED data

# Compare scaled GD with unscaled NE
difference = np.linalg.norm(
    final_theta - theta_calculated_normal
)  # Uses theta_calculated_normal (or _unscaled)
print("Difference (scaled GD vs unscaled NE):", difference)

# Prepare X with the bias term (column of ones at the beginning)
# Calculate theta using the Normal Equation with X
theta_calculated_normal_scaled = calculate_normal_equation(X_train_bias_unscaled, y_train)

print("\nScaled GD Theta:", final_theta.T)  # .T to view as row if long
print("Scaled NE Theta:", theta_calculated_normal_scaled.T)
difference = np.linalg.norm(final_theta - theta_calculated_normal_scaled)
# Comparison: scaled GD vs scaled NE
correct_difference = np.linalg.norm(final_theta - theta_calculated_normal_scaled)
print("Difference (scaled GD vs SCALED NE):", correct_difference)

num_iterations = 100
alpha = 0.1

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
    theta_calculated, cost_history = gradient_descent(X_train_bias_scaled, y_train, initial_theta.copy(), alpha, num_iterations, lambda_reg)
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
    theta1_value = theta_calculated[1, 0]
    
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

if lambda_values[0] == 0:
    for i in range(1, len(theta_coeffs_values)):
        plt.plot(lambda_values[1:], theta_coeffs_values[i][1:], marker='o', linestyle='-', label=fr'$\theta_{i}$')
    plt.xscale('log') # Ahora es seguro porque lambda_values[1:] no contiene 0
else: # Si todos los lambda_values son > 0
    for i in range(1, len(theta_coeffs_values)):
        plt.plot(lambda_values, theta_coeffs_values[i], marker='o', linestyle='-', label=fr'$\theta_{i}$')
    plt.xscale('log')

plt.xlabel(r'$\lambda$', fontsize=12)
plt.ylabel(r'Value of coefficients $\theta_j$', fontsize=12)
plt.title(r'Coefficients $\theta_1$ to $\theta_8$ as a function of $\lambda$', fontsize=14)

plt.legend()      # Show legend with labels
plt.grid(True)    # Grid for better visualization
plt.tight_layout()
plt.xscale('log')
plt.show()
