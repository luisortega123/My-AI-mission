# --- IMPORTS ---
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score
from itertools import product
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

# Initialize and Train the XGBClassifier
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_test = xgb_model.predict(X_test)
y_pred_train = xgb_model.predict(X_train)

# ============================================================
# ------------ CALCULATE EVALUATION METRICS ------------
# ============================================================
print("\n--- Calculate evaluation metrics ---")
print("---  Accuracy (Test Set) ---")
acc_test = accuracy_score(y_test, y_pred_test)
print(acc_test)

print("---  Accuracy (Train Set) ---")
acc_train = accuracy_score(y_train, y_pred_train)
print(acc_train)

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
n_estimators_list = [50, 100, 200]
max_depth_list = [2, 3, 4, 5, 6, 7]
learning_rate_list = [0.01, 0.05, 0.1]
result_list = []

# Create the Cartesian product (all combinations)
param_grid = product(n_estimators_list, max_depth_list, learning_rate_list)

# Iterate over each combination
for current_ne, current_md, current_lr in param_grid:
    print(f"n_estimators: {current_ne}, max_depth: {current_md} and learning_rate: {current_lr}")
    xgb_model_xp = XGBClassifier(n_estimators=current_ne, learning_rate=current_lr, max_depth=current_md, eval_metric='logloss', random_state=42)
    xgb_model_xp.fit(X_train, y_train)
    y_pred_test_xp = xgb_model_xp.predict(X_test)
    y_pred_train_xp = xgb_model_xp.predict(X_train)
    acc_test_xp = accuracy_score(y_test, y_pred_test_xp)
    acc_train_xp = accuracy_score(y_train, y_pred_train_xp)
    experiment_result = {
        "n_estimators" : current_ne,
        "max_depth" : current_md,
        "Learning_rate" : current_lr,
        "Test accuracy" : acc_test_xp,
        "Train accuracy" : acc_train_xp,
    }
    result_list.append(experiment_result)

# Convert this list of dictionaries into a Pandas DataFrame.
print("\n--- Selecting the Best Experiment Using pandas ---")
result_df = pd.DataFrame(result_list)
print("\n--- First 5 rows of the DataFrame (unsorted) ---")
print(result_df.head())
df_sorted = result_df.sort_values(by="Test accuracy", ascending=False)
print("\n--- The Best Experiment (Top row of sorted DataFrame) ---")
best_experiment_pd = df_sorted.iloc[0]
print(best_experiment_pd)


print("\n--- Best Hyperparameters ---")
print(f"  max_depth: {best_experiment_pd['max_depth']}")
print(f"  n_estimators: {best_experiment_pd['n_estimators']}")
print(f"  Learning_rate: {best_experiment_pd['Learning_rate']}")
print(f"  Test Accuracy: {best_experiment_pd['Test accuracy'] * 100:.2f}%")
print(f"  Training Accuracy: {best_experiment_pd['Train accuracy'] * 100:.2f}%")



# ====================================================================
# --- CLASSIFICATION REPORT GRADIENT BOOSTING FOR BEST EXPERIMENT ---
# ====================================================================

# Initialize and Train the XGBClassifier
xgb_model_best_experiment = XGBClassifier(n_estimators=50, learning_rate=0.01, max_depth=3, eval_metric='logloss', random_state=42)
xgb_model_best_experiment.fit(X_train, y_train)
y_pred_test_best_experiment = xgb_model_best_experiment.predict(X_test)
y_pred_train_best_experiment= xgb_model_best_experiment.predict(X_train)

print("\n---  Classification Report Best Experiment (Test Set) ---")
cl_report_test_best_experiment = classification_report(y_test, y_pred_test_best_experiment) 
print(cl_report_test_best_experiment)

print("\n---  Classification Report Best Experiment (Train Set) ---") 
cl_report_train_best_experiment = classification_report(y_train, y_pred_train_best_experiment)
print(cl_report_train_best_experiment)