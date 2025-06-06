from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
# Print the results to get an idea of the data
print(f"Dimensions of X: \n{X.shape}")  # Dimensions of X
print(f"Dimensions of y: \n{y.shape}")  # Dimensions of y
print(f"Column names:\n {data.feature_names}")  # Column names (features)
print(f"Class name:\n{data.target_names}")  # Class or label names

# Bring in the train_test_split function, but with our own variables
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # test_size is equal to 20% for testing

# As good practice, we print shapes
print("\nShape of X_train:", X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test: ", y_test.shape)


# Block of repetitive actions
# Step 2 of the methodology: for each experiment we do with different SVM configurations:
# Create an instance of this Classifier
svc_model = SVC(kernel="linear")

# Train the svc model, how? with .fit() and its arguments
svc_model.fit(X_train, y_train)

# Make predictions on our samples
# model_name.predict(X) This method returns the predicted classes
y_pred_test = svc_model.predict(X_test)
# (Optional) Predictions on the training set
y_pred_train = svc_model.predict(X_train)

# Let's get the accuracy with .score()
# Evaluate accuracy on the test set
accuracy_test = svc_model.score(X_test, y_test)
print(f"\nAccuracy on test: {accuracy_test*100:,.2f}%")

# Evaluate accuracy on the training set
accuracy_train = svc_model.score(X_train, y_train)
print(f"Accuracy on train:{accuracy_train*100:,.2f}%")

# Successfully completed our first "Isolated Work Unit".

# Experimentation stage
# Experiment with different kernels ('linear', 'rbf', 'poly').

# 'rbf' kernels
# Create SVC() instance
svc_model_exp = SVC(kernel="rbf", C=10, gamma=0.1)
# Train the model with .fit()
svc_model_exp.fit(X_train, y_train)
# Predictions with .predict()
y_pred_test_exp = svc_model_exp.predict(X_test)
y_pred_train_exp = svc_model_exp.predict(X_train)
# Evaluate accuracy with .score()
accuracy_test_exp = svc_model_exp.score(X_test, y_test)
accuracy_train_exp = svc_model_exp.score(X_train, y_train)
print(f"\nAccuracy on test 'rbf' kernels: {accuracy_test_exp*100:,.2f}%")
print(f"Accuracy on train 'rbf' kernels: {accuracy_train_exp*100:,.2f}%")


# We will iterate over a list of kernel names
kernel_list = ["linear", "rbf", "poly"]
C_values = [0.1, 1, 10, 100]
gamma_values = [0.01, 0.1, 1]
degree_values = [2, 3, 4]


results_list = []
for kernel_name in kernel_list:
    for C_val in C_values:
        # --- Block for RBF kernel ---
        if kernel_name == "rbf":
            for gamma_val in gamma_values:
                print(f"Testing kernel = {kernel_name}, C = {C_val}, gamma = {gamma_val}")
                model = SVC(kernel=kernel_name, C=C_val, gamma=gamma_val)
                model.fit(X_train, y_train)
                acc_train = model.score(X_train, y_train)
                acc_test = model.score(X_test, y_test)
                print(f" Training Accuracy: {acc_train * 100:,.2f}%, Test Accuracy: {acc_test * 100:,.2f}%")
                # Guide Results Handling
                experiment_results = {
                    "kernel": kernel_name,
                    "C": C_val,
                    "gamma": gamma_val, # Specific to RBF
                    "Training Accuracy": acc_train,
                    "Test Accuracy": acc_test,
                }
                results_list.append(experiment_results)

        # --- Block for Polynomial kernel ---
        elif kernel_name == "poly":
            for degree_val in degree_values:
                print(f"Testing kernel = {kernel_name}, C = {C_val}, degree = {degree_val}")
                model = SVC(kernel=kernel_name, C=C_val, degree=degree_val)
                model.fit(X_train, y_train)
                acc_train = model.score(X_train, y_train)
                acc_test = model.score(X_test, y_test)
                print(f" Training Accuracy: {acc_train * 100:,.2f}%, Test Accuracy: {acc_test * 100:,.2f}%")
                # Guide Results Handling
                experiment_results = {
                    "kernel": kernel_name,
                    "C": C_val,
                    "degree": degree_val, # Specific to Poly
                    "Training Accuracy": acc_train,
                    "Test Accuracy": acc_test,
                }
                results_list.append(experiment_results)

        # --- Block for Linear kernel ---
        elif kernel_name == "linear": # Made 'linear' explicit for clarity
            print(f"kernel = {kernel_name}, C = {C_val}")
            model = SVC(kernel=kernel_name, C=C_val)
            model.fit(X_train, y_train)
            acc_train = model.score(X_train, y_train)
            acc_test = model.score(X_test, y_test)
            print(f" Training Accuracy: {acc_train * 100:,.2f}%, Test Accuracy: {acc_test * 100:,.2f}%")
            # Guide Results Handling
            experiment_results = {
                "kernel": kernel_name,
                "C": C_val,
                # Does not include gamma or degree, correct for linear!
                "Training Accuracy": acc_train,
                "Test Accuracy": acc_test,
            }
            results_list.append(experiment_results)

# ===========================================================
# Find the experiment with the highest Test Accuracy
# ===========================================================
best_accuracy = -1
best_experiment = None

# Iterate through 'experiment', which is a dictionary in your list
for current_experiment in results_list:
    # Access the "Test Accuracy" value in the experiment dictionary
    current_test_accuracy = current_experiment["Test Accuracy"]
    # If block to find values greater than best_accuracy
    if current_test_accuracy > best_accuracy:
        # Update the best accuracy
        best_accuracy = current_test_accuracy
        # Save the entire dictionary of the best experiment
        best_experiment = current_experiment
# Print the best result AFTER the loop
print("\n--- The Best Experiment ---")
print(f"  Kernel: {best_experiment['kernel']}")
print(f"  C: {best_experiment['C']}")

# If it's RBF, print gamma
if best_experiment['kernel'] == 'rbf' and 'gamma' in best_experiment:
    print(f"  Gamma: {best_experiment['gamma']}")

# If it's Poly, print degree
if best_experiment['kernel'] == 'poly' and 'degree' in best_experiment:
    print(f"  Degree: {best_experiment['degree']}")

# Print accuracies as percentages
print(f"  Training Accuracy: {best_experiment['Training Accuracy'] * 100:.2f}%")
print(f"  Test Accuracy: {best_experiment['Test Accuracy'] * 100:.2f}%")

# ==========================================================================================
# Experimentation with pandas to find the experiment with the highest Test Accuracy
# ==========================================================================================
# 1. Convert our list into a DataFrame
df_results = pd.DataFrame(results_list)
print("\n--- First 5 rows of the DataFrame ---")
print(df_results.head())
# Sort the DF by the Test Accuracy column
# Sort from largest to smallest by the 'Test Accuracy' column
df_sorted = df_results.sort_values(by="Test Accuracy", ascending=False)
# Select that first row
print("\n--- The Best Experiment ---")
best_experiment_pd = df_sorted.iloc[0]

print(f"  Kernel: {best_experiment_pd['kernel']}")
print(f"  C: {best_experiment_pd['C']}")

# If it's RBF, print gamma
if best_experiment_pd['kernel'] == 'rbf' and 'gamma' in best_experiment_pd:
    print(f"  Gamma: {best_experiment_pd['gamma']}")

# If it's Poly, print degree
if best_experiment_pd['kernel'] == 'poly' and 'degree' in best_experiment_pd:
    print(f"  Degree: {best_experiment_pd['degree']}")

# Format as percentage
print(f"  Test Accuracy: {best_experiment_pd['Test Accuracy'] * 100:.2f}%")
print(f"  Training Accuracy: {best_experiment_pd['Training Accuracy'] * 100:.2f}%")


# ====================================
# --- Advanced Metrics for SVC ---
# ====================================

# Create SVC() instance
best_svc_model = SVC(kernel="linear", C=0.1)
best_svc_model.fit(X_train, y_train)
# Predictions
y_pred_test_best_svc = best_svc_model.predict(X_test)
# Evaluate accuracy with .score()
accuracy_best_test = best_svc_model.score(X_test, y_test)

# Predictions on Training Set
y_pred_train_best_svc = best_svc_model.predict(X_train)
# Evaluate accuracy with .score()
accuracy_best_train = best_svc_model.score(X_train, y_train)


# ============================================================
# ------------ CALCULATE EVALUATION METRICS ------------
# ============================================================

print("\n--- Calculate evaluation metrics ---")

print("--- Accuracy (Test Set) ---")
print(f"Accuracy Score:\n{accuracy_best_test:.4f}")
print("--- Accuracy (Train Set) ---")
print(f"Accuracy Score:\n{accuracy_best_train:.4f}")

print("--- Matriz de Confusión (Test Set) ---")
conf_best_test = confusion_matrix(y_test, y_pred_test_best_svc)
print(f"Confusion Matrix:\n{conf_best_test}")

print("--- Confusion Matrix (Train Set) ---")
conf_best_train = confusion_matrix(y_train, y_pred_train_best_svc)
print(f"Confusion Matrix:\n{conf_best_train}")

print("---  Precision Score (Test Set) ---")
preci_best_test = precision_score(y_test, y_pred_test_best_svc)
print(f"Precision Score: {preci_best_test:.4f}")

print("--- Precision Score (Train Set) ---")
preci_best_train = precision_score(y_train, y_pred_train_best_svc)
print(f"Precision Score: {preci_best_train:.4f}")

print("--- Recall Score (Test Set) ---")
recall_best_test = recall_score(y_test, y_pred_test_best_svc)
print(f"Recall Score: {recall_best_test:.4f}")

print("--- Recall Score (Train Set) ---")
recall_best_train = recall_score(y_train, y_pred_train_best_svc)
print(f"Recall Score: {recall_best_train:.4f}")

print("---  F1 Score (Test Set) ---")
f1_best_test = f1_score(y_test, y_pred_test_best_svc)
print(f"F1 Score: {f1_best_test:.4f}")

print("--- F1 Score (Train Set) ---")
f1_best_train = f1_score(y_train, y_pred_train_best_svc)
print(f"F1 Score: {f1_best_train:.4f}")


# =========================================================
# --- Classification Report for SVC ---
# =========================================================

print("\n---  Classification Report (Test Set) ---")
cl_report_test = classification_report(y_test, y_pred_test_best_svc)
print(cl_report_test)

print("\n---  Clasiification Report (Train Set) ---")
cl_report_train = classification_report(y_train, y_pred_train_best_svc)
print(cl_report_train)
