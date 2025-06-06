# --- IMPORTS ---
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
import pandas as pd

# --- LOAD DATA ---
cancer_data = load_breast_cancer()
print("Feature names:", cancer_data.feature_names)
print("Class names:", cancer_data.target_names)
print("Dataset description:", cancer_data.DESCR)

X_orig = cancer_data.data
y_orig = cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2 ,random_state=42)
print(f"Shapes: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

# Initialize and Train the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# ============================================================
# ------------ CALCULATE EVALUATION METRICS ------------
# ============================================================
print("\n--- Calculate evaluation metrics ---")

print("---  Accuracy (Test Set) ---")
accuracy_test = accuracy_score(y_test,y_pred_test)
print(f"Accuracy Score:\n{accuracy_test:.4f}")
print("---  Accuracy (Train Set) ---")
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Accuracy Score:\n{accuracy_train:.4f}")

print("--- Matriz de Confusión (Test Set) ---")
cm_test = confusion_matrix(y_test, y_pred_test)
print(f"Confusion Matrix:\n{cm_test}")

print("--- Confusion Matrix (Train Set) ---")
cm_train = confusion_matrix(y_train, y_pred_train)
print(f"Confusion Matrix:\n{cm_train}")

print("---  Precision Score (Test Set) ---")
preci_test = precision_score(y_test, y_pred_test)
print(f"Precision Score: {preci_test:.4f}")

print("--- Precision Score (Train Set) ---")
preci_train = precision_score(y_train, y_pred_train)
print(f"Precision Score: {preci_train:.4f}")

print("--- Recall Score (Test Set) ---")
recall_test = recall_score(y_test, y_pred_test)
print(f"Recall Score: {recall_test:.4f}")

print("--- Recall Score (Train Set) ---")
recall_train = recall_score(y_train, y_pred_train)
print(f"Recall Score: {recall_train:.4f}")

print("---  F1 Score (Test Set) ---")
f1_test = f1_score(y_test, y_pred_test)
print(f"F1 Score: {f1_test:.4f}")

print("--- F1 Score (Train Set) ---")
f1_train = f1_score(y_train, y_pred_train)
print(f"F1 Score: {f1_train:.4f}")

# =============================
# --- Classification Report ---
# =============================

print("\n---  Classification Report (Test Set) ---")
cl_report_test = classification_report(y_test, y_pred_test)
print(cl_report_test)

print("\n---  Clasiification Report (Train Set) ---")
cl_report_train = classification_report(y_train, y_pred_train)
print(cl_report_train)



# Starting a loop to test different values
max_depth_options = [3, 5, 7, 10]
n_estimators_options = [50, 100, 150, 200]
result_list = []

for current_n_estimators in n_estimators_options:
    for current_max_depth in max_depth_options:
        print(f"\n>>> Testing: n_estimators = {current_n_estimators}, max_depth = {current_max_depth}")
        rf_model_md = RandomForestClassifier(n_estimators=current_n_estimators, max_depth=current_max_depth, random_state=42)
        rf_model_md.fit(X_train, y_train)
        y_pred_test_md = rf_model_md.predict(X_test)
        y_pred_train_md = rf_model_md.predict(X_train)
        accuracy_test_md = accuracy_score(y_test, y_pred_test_md)
        accuracy_train_md = accuracy_score(y_train, y_pred_train_md)
        experiment_result = {
            "n_estimators" : current_n_estimators,
            "max_depth" : current_max_depth,
            "Test accuracy" : accuracy_test_md,
            "Train accuracy" : accuracy_train_md
            }
        result_list.append(experiment_result)
        print(f"   For n_est={current_n_estimators}, m_depth={current_max_depth} -> Train Acc: {accuracy_train_md:.4f}, Test Acc: {accuracy_test_md:.4f}")

# Convert this list of dictionaries into a Pandas DataFrame.
print("\n--- Selecting the Best Experiment Using pandas ---")
results_df = pd.DataFrame(result_list)
print("\n--- First 5 rows of the DataFrame (unsorted) ---")
print(results_df.head())
df_sorted = results_df.sort_values(by= "Test accuracy", ascending=False)
print("\n--- The Best Experiment (Top row of sorted DataFrame) ---")
best_experiment_pd = df_sorted.iloc[0]
print(best_experiment_pd)

print("\n--- Best Hyperparameters ---")
print(f"  max_depth: {best_experiment_pd['max_depth']}")
print(f"  n_estimators: {best_experiment_pd['n_estimators']}")
print(f"  Test Accuracy: {best_experiment_pd['Test accuracy'] * 100:.2f}%")
print(f"  Training Accuracy: {best_experiment_pd['Train accuracy'] * 100:.2f}%")




