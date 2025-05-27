from sklearn.svm import SVR
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


# Import data from the Dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Print the Dataset values
print("--- Dimensions of X ---")
print(X.shape)
print("--- Dimensions of y ---")
print(y.shape)
# Print the names
print("--- Column Names ---")
print(data.feature_names)
print("--- Target Variable Name ---")  # Adjusted comment
print(data.target_names)  # ['MedHouseVal'], optional to print

# Bring in our train_test_split function
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print the Shapes
print("Shape of X_train:", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test: ", y_test.shape)


# Bring in the SVR instance
svr_model = SVR()  # The default kernel for SVR is 'rbf', and C is 1.0, among others
svr_model.fit(X_train, y_train)  # Train the model
# Predictions on the set
y_pred_svr_train = svr_model.predict(X_train)
y_pred_svr_test = svr_model.predict(X_test)
# mse real values, y
mse_svr_test = mean_squared_error(y_test, y_pred_svr_test)
mse_svr_train = mean_squared_error(y_train, y_pred_svr_train)

# Feature scaling.
# Identify the features, convert to DataFrame
df_train = pd.DataFrame(X_train, columns=data.feature_names).describe().round(2)
print(f"Description of the features:\n", df_train)
# Scale the function
# Call the scaler
scaler = StandardScaler()
# Here the scaler learns how to transform your data
scaler.fit(X_train)
# Apply a transformation
X_train_scaled = scaler.transform(X_train)  # Apply that transformation to X_train
X_test_scaled = scaler.transform(X_test)  # Apply the same transformation to X_test


# Experimentation stage
# Define Hyperparameters
kernel_list = ["linear", "rbf", "poly"]
C_values = [0.1, 1, 3]
gamma_values = [0.01, 0.1, 1]
degree_values = [2, 3]

result_list = []
for kernel_name in kernel_list:
    for C_val in C_values:
        # --- Block for RBF kernel ---
        if kernel_name == "rbf":
            for gamma_val in gamma_values:
                print(f"Testing kernel: {kernel_name}, C={C_val}, gamma={gamma_val}")
                # Bring in the SVR instance
                model = SVR(kernel=kernel_name, C=C_val, gamma=gamma_val)
                model.fit(X_train_scaled, y_train)
                print("TRAINING COMPLETED.")  # To know when it ends
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_test = mean_squared_error(y_test, y_pred_test)
                print(f"MSE Train: {mse_train:.4f}, MSE Test: {mse_test:.4f}")
                # Calculate  MAE (Mean Absolute Error)
                mae_train = mean_absolute_error(y_train, y_pred_train)
                mae_test = mean_absolute_error(y_test, y_pred_test)
                print(f"MAE Train : {mae_train:.4f}, MAE Test: {mae_test:.4f} ")
                # Calculate and store the R² (coefficient of determination) for each experiment.
                r2_train = model.score(X_train_scaled, y_train)
                r2_test = model.score(X_test_scaled, y_test)
                print(f"R2 Train: {r2_train:.4f}, R2 Test: {r2_test:.4f}")
                # Guide results
                experiment_result = {
                    "kernel": kernel_name,
                    "C": C_val,
                    "gamma": gamma_val,
                    "MSE Train": mse_train,
                    "MSE Test": mse_test,
                    "MAE Train": mae_train,
                    "MAE Test": mae_test,
                    "R2 Train": r2_train,
                    "R2 Test": r2_test,
                }
                result_list.append(experiment_result)
        elif kernel_name == "poly":
            for degree_val in degree_values:
                print(
                    f"Testing kernel = {kernel_name}, C = {C_val}, degree = {degree_val}"
                )
                model = SVR(kernel=kernel_name, C=C_val, degree=degree_val)
                model.fit(X_train_scaled, y_train)
                print("TRAINING COMPLETED.")  # To know when it ends

                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                mse_train = mean_squared_error(y_train, y_pred_train)
                mse_test = mean_squared_error(y_test, y_pred_test)
                print(f"MSE Train: {mse_train:.4f}, MSE Test{mse_test:.4f}")
                # Calculate  MAE (Mean Absolute Error)
                mae_train = mean_absolute_error(y_train, y_pred_train)
                mae_test = mean_absolute_error(y_test, y_pred_test)
                print(f"MAE Train : {mae_train:.4f}, MAE Test: {mae_test:.4f} ")
                r2_train = model.score(X_train_scaled, y_train)
                r2_test = model.score(X_test_scaled, y_test)
                print(f"R2 Train: {r2_train:.4f}, R2 Test: {r2_test:.4f}")
                experiment_result = {
                    "kernel": kernel_name,
                    "C": C_val,
                    "degree": degree_val,
                    "MSE Train": mse_train,
                    "MSE Test": mse_test,
                    "MAE Train": mae_train,
                    "MAE Test": mae_test,
                    "R2 Train": r2_train,
                    "R2 Test": r2_test,
                }
                result_list.append(experiment_result)
        elif kernel_name == "linear":
            print(f"Testing kernel = {kernel_name}, C = {C_val}")
            model = SVR(kernel=kernel_name, C=C_val)
            model.fit(X_train_scaled, y_train)
            print("TRAINING COMPLETED.")  # To know when it ends

            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            mse_train = mean_squared_error(y_train, y_pred_train)
            mse_test = mean_squared_error(y_test, y_pred_test)
            print(f"MSE Train: {mse_train:.4f}, MSE Test: {mse_test:.4f}")
            # Calculate  MAE (Mean Absolute Error)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            print(f"MAE Train : {mae_train:.4f}, MAE Test: {mae_test:.4f} ")
            r2_train = model.score(X_train_scaled, y_train)
            r2_test = model.score(X_test_scaled, y_test)
            print(f"R2 Train: {r2_train:.4f}, R2 Test: {r2_test:.4f}")
            experiment_result = {
                "kernel": kernel_name,
                "C": C_val,
                "MSE Train": mse_train,
                "MSE Test": mse_test,
                "MAE Train": mae_train,
                "MAE Test": mae_test,
                "R2 Train": r2_train,
                "R2 Test": r2_test,
            }
            result_list.append(experiment_result)


# Post-processing: analyze these results to understand which combination of kernel and hyperparameters performed best.
best_mse_score = float("inf")
best_r2_score = -1.0
best_mae_score = float("inf")
best_experiment_mse = None
best_experiment_r2 = None
best_experiment_mae = None

for current_experiment in result_list:
    current_test_mse = current_experiment["MSE Test"]
    current_test_r2 = current_experiment["R2 Test"]
    current_test_mae = current_experiment["MAE Test"]
    if current_test_mse < best_mse_score:
        best_mse_score = current_test_mse
        best_experiment_mse = current_experiment
    if current_test_r2 > best_r2_score:
        best_r2_score = current_test_r2
        best_experiment_r2 = current_experiment
    if current_test_mae < best_mae_score:
        best_mae_score = current_test_mae
        best_experiment_mae = current_experiment

# Print the best results.
print("--- Best Experiment According to MSE (Lowest Value) ---")
if best_experiment_mse:
    print(best_experiment_mse)
    print(f"MSE Test: {best_mse_score:.4f}")


print("\n--- Best Experiment According to R² (Highest Value) ---")
if best_experiment_r2:
    print(best_experiment_r2)
    print(f"R2 Test: {best_r2_score:.4f}")


# Experimentation with pandas to find the experiment with the highest Test Accuracy
# 1. Convert our list into a DataFrame
print("\n--- Selecting the Best Experiment Using pandas ---")

df_results = pd.DataFrame(result_list)
print("--- First 5 rows of the DataFrame ---")
print(df_results.head())
df_sorted_mse = df_results.sort_values(by="MSE Test", ascending=True)
df_sorted_r2 = df_results.sort_values(by="R2 Test", ascending=False)
df_sorted_mae = df_results.sort_values(by="MAE Test", ascending=True)

print("--- The Best Experiment ---")
best_experiment_pd_mse = df_sorted_mse.iloc[0]
best_experiment_pd_r2 = df_sorted_r2.iloc[0]
best_experiment_pd_mae = df_sorted_mae.iloc[0]

print("--- Best Experiment According to MSE (Lowest Value) ---")
print(f"    kernels: {best_experiment_pd_mse["kernel"]}")
print(f"    C : {best_experiment_pd_mse["C"]}")

# To print gamma or degree conditionally:
if best_experiment_pd_mse["kernel"] == "rbf":
    print(f"    Gamma: {best_experiment_pd_mse["gamma"]}")
elif best_experiment_pd_mse["kernel"] == "poly":
    print(f"    Degree: {best_experiment_pd_mse["degree"]}")


print(f"  MSE Test: {best_experiment_pd_mse["MSE Test"]:.4f}")
print(f"  MAE Test: {best_experiment_pd_mse["MAE Test"]:.4f}")
print(f"  R2 Test: {best_experiment_pd_mse["R2 Test"]:.4f}")
print(f"  MSE Train: {best_experiment_pd_mse["MSE Train"]:.4f}")
print(f"  MAE Train: {best_experiment_pd_mse["MAE Train"]:.4f}")
print(f"  R2 Train: {best_experiment_pd_mse["R2 Train"]:.4f}")


print("\n--- Best Experiment According to MAE (Lowest Value) ---")
print(f"    kernels: {best_experiment_pd_mae["kernel"]}")
print(f"    C : {best_experiment_pd_mae["C"]}")
# To print gamma or degree conditionally:
if best_experiment_pd_mae["kernel"] == "rbf":
    print(f"    Gamma: {best_experiment_pd_mae["gamma"]}")
elif best_experiment_pd_mae["kernel"] == "poly":
    print(f"    Degree: {best_experiment_pd_mae["degree"]}")
    
print(f"  MSE Test: {best_experiment_pd_mae["MSE Test"]:.4f}")
print(f"  MAE Test: {best_experiment_pd_mae["MAE Test"]:.4f}")
print(f"  R2 Test: {best_experiment_pd_mae["R2 Test"]:.4f}")
print(f"  MSE Train: {best_experiment_pd_mae["MSE Train"]:.4f}")
print(f"  MAE Train: {best_experiment_pd_mae["MAE Train"]:.4f}")
print(f"  R2 Train: {best_experiment_pd_mae["R2 Train"]:.4f}")


print("\n--- Best Experiment According to R² (Highest Value) ---")

if best_experiment_pd_r2["kernel"] == "rbf":
    print(f"    Gamma: {best_experiment_pd_r2["gamma"]}")
elif best_experiment_pd_r2["kernel"] == "poly":
    print(f"    Degree: {best_experiment_pd_r2["degree"]}")


print(f"  Kernel: {best_experiment_pd_r2['kernel']}")
print(f"  C: {best_experiment_pd_r2['C']}")

print(f"  MSE Test: {best_experiment_pd_r2["MSE Test"]:.4f}")
print(f"  MAE Test: {best_experiment_pd_r2["MAE Test"]:.4f}")
print(f"  R2 Test: {best_experiment_pd_r2["R2 Test"]:.4f}")
print(f"  MSE Train: {best_experiment_pd_r2["MSE Train"]:.4f}")
print(f"  MAE Train: {best_experiment_pd_r2["MAE Train"]:.4f}")
print(f"  R2 Train: {best_experiment_pd_r2["R2 Train"]:.4f}")
