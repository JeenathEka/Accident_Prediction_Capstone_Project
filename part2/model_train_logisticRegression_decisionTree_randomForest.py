import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import numpy as np
import joblib
from pathlib import Path  # For handling file paths reliably

base_dir = Path(__file__).resolve().parent
plot_dir = base_dir / "plots"  # Folder where generated plots will be saved
plot_dir.mkdir(exist_ok=True)  # Create the folder if it does not already exist

# Load the dataset
df = pd.read_csv("../part1/cleaned_data.csv")
df.columns = df.columns.str.strip()  # Strip whitespace from column names
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()  # Strip whitespace from string columns


# Feature and label

feature_cols = ["State Name", "City Name", "Month", "Day of Week", "Weather Conditions"]
target_col = "Accident Location Details"

# value counts

print(df[target_col].value_counts())

x_raw = df[feature_cols]
y_raw = df[target_col]


# encode label

le = LabelEncoder()
y = le.fit_transform(y_raw)

# one hot encode features
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X = ohe.fit_transform(x_raw)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def print_results(name, y_test, y_pred):
    print(f"Results for {name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0,target_names=le.classes_)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Recall Score: {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print("\n")

# smote

CV = 5
min_class_count = min(Counter(y_train).values())
safe_neighbors = int(min_class_count * (CV - 1) / CV) -1
k_neighbors = max(1, min(5, safe_neighbors))
print(f"Using k_neighbors={k_neighbors} for SMOTE based on minimum class count of {min_class_count} and CV={CV}")
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)



# logistic regression

lr_pipe = ImbPipeline([('smote', smote),
                       ('classifier', LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced'))])

lr_params = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'saga']
    }

lr_grid = GridSearchCV(lr_pipe, lr_params, cv=CV, scoring='recall_macro', n_jobs=-1)
lr_grid.fit(X_train, y_train)
lr_pred = lr_grid.predict(X_test)
lr_best = lr_grid.best_estimator_
print("Best parameters for Logistic Regression:", lr_grid.best_params_)
print_results("Logistic Regression", y_test, lr_pred)

# lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
# lr.fit(X_train, y_train)
# lr_pred = lr.predict(X_test)
# print_results("Logistic Regression", y_test, lr_pred)


# decision tree

dt_pipe = ImbPipeline([('smote', smote),
                       ('classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'))])

dt_params = {
    'classifier__max_depth': [5, 10, 15, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(dt_pipe, dt_params, cv=CV, scoring='recall_macro', n_jobs=-1)
dt_grid.fit(X_train, y_train)
dt_pred = dt_grid.predict(X_test)
dt_best = dt_grid.best_estimator_
print("Best parameters for Decision Tree Classifier:", dt_grid.best_params_)
print_results("Decision Tree", y_test, dt_pred)

# dt = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
# dt.fit(X_train, y_train)
# dt_pred = dt.predict(X_test)
# print_results("Decision Tree", y_test, dt_pred)



# random forest

rf_pipe = ImbPipeline([('smote', smote),
                       ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced_subsample', n_jobs=-1))])   
rf_params = {
    'classifier__n_estimators': [300],
    'classifier__max_depth': [5],
    'classifier__min_samples_split': [2],
    'classifier__min_samples_leaf': [2],
    'classifier__max_features': ['sqrt']
}
rf_grid = GridSearchCV(rf_pipe, rf_params, cv=CV, scoring='recall_macro', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
rf_pred = rf_grid.predict(X_test)
rf_best = rf_grid.best_estimator_
print("Best parameters for Random Forest Classifier:", rf_grid.best_params_)
print_results("Random Forest", y_test, rf_pred)

# rf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)
# print_results("Random Forest", y_test, rf_pred)




# model comparison summary
models = ["Logistic Regression", "Decision Tree", "Random Forest"]

for name, pred in zip(models, [lr_pred, dt_pred, rf_pred]):
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"Recall Score: {recall_score(y_test, pred, average='macro', zero_division=0):.4f}")
    print("\n")


# ROC and AUC plots for multiclass classification
n_classes = len(le.classes_)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('ROC Curves - One-vs-Rest Approach', fontsize=16)

models_list = [
    ("Logistic Regression", lr_grid),
    ("Decision Tree", dt_grid),
    ("Random Forest", rf_grid)
]

for idx, (model_name, model) in enumerate(models_list):
    # Get probability predictions
    y_proba = model.predict_proba(X_test)
    
    # Compute ROC curve and ROC area for each class using One-vs-Rest
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        # Create binary labels for One-vs-Rest approach
        binary_labels = (y_test == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(binary_labels, y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute macro-average ROC curve
    fpr_macro = np.linspace(0, 1, 100)
    tpr_macro = np.zeros_like(fpr_macro)
    for i in range(n_classes):
        tpr_macro += np.interp(fpr_macro, fpr[i], tpr[i])
    tpr_macro /= n_classes
    roc_auc_macro = auc(fpr_macro, tpr_macro)
    
    # Plot ROC curves
    ax = axes[idx]
    ax.plot(fpr_macro, tpr_macro,
            label=f'Macro-average ROC (AUC = {roc_auc_macro:.2f})',
            color='deeppink', linestyle=':', linewidth=3)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{le.classes_[i]} (AUC = {roc_auc[i]:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(model_name)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(plot_dir /'roc_auc_plots.png', dpi=300, bbox_inches='tight')
print("ROC/AUC plots saved as 'roc_auc_plots.png'")
plt.close()


# Individual ROC plots for Random Forest (best model)
print("\n--- Generating detailed Random Forest ROC/AUC plot ---")
y_proba_rf = rf_grid.predict_proba(X_test)

fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()

# Use One-vs-Rest for Random Forest
for i in range(n_classes):
    binary_labels = (y_test == i).astype(int)
    fpr_rf[i], tpr_rf[i], _ = roc_curve(binary_labels, y_proba_rf[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

# Macro-average
fpr_rf["macro"] = np.linspace(0, 1, 100)
tpr_rf["macro"] = np.zeros_like(fpr_rf["macro"])
for i in range(n_classes):
    tpr_rf["macro"] += np.interp(fpr_rf["macro"], fpr_rf[i], tpr_rf[i])
tpr_rf["macro"] /= n_classes
roc_auc_rf["macro"] = auc(fpr_rf["macro"], tpr_rf["macro"])

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr_rf["macro"], tpr_rf["macro"],
        label=f'Macro-average ROC (AUC = {roc_auc_rf["macro"]:.3f})',
        color='navy', linestyle=':', linewidth=3)

colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
for i, color in zip(range(n_classes), colors):
    ax.plot(fpr_rf[i], tpr_rf[i], color=color, lw=2,
            label=f'{le.classes_[i]} (AUC = {roc_auc_rf[i]:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Random Forest Classifier - ROC Curves', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(plot_dir /'random_forest_roc_auc_detailed.png', dpi=300, bbox_inches='tight')
print("Detailed Random Forest ROC/AUC plot saved as 'random_forest_roc_auc_detailed.png'")
plt.close()


# Combined ROC/AUC comparison for all models
print("\n--- Generating combined ROC/AUC comparison plot ---")

fig, ax = plt.subplots(figsize=(12, 8))

# Define colors for each model
model_colors = {
    "Logistic Regression": "#FF6B6B",
    "Decision Tree": "#4ECDC4",
    "Random Forest": "#45B7D1"
}

# Get predictions for all models
y_proba_lr = lr_grid.predict_proba(X_test)
y_proba_dt = dt_grid.predict_proba(X_test)
y_proba_rf = rf_grid.predict_proba(X_test)

all_models_data = [
    ("Logistic Regression", y_proba_lr, y_proba_lr),
    ("Decision Tree", y_proba_dt, y_proba_dt),
    ("Random Forest", y_proba_rf, y_proba_rf)
]

# Plot macro-average ROC for each model
for model_name, y_proba, _ in all_models_data:
    fpr_model = dict()
    tpr_model = dict()
    
    # Compute ROC for each class
    for i in range(n_classes):
        binary_labels = (y_test == i).astype(int)
        fpr_model[i], tpr_model[i], _ = roc_curve(binary_labels, y_proba[:, i])
    
    # Compute macro-average
    fpr_macro_model = np.linspace(0, 1, 100)
    tpr_macro_model = np.zeros_like(fpr_macro_model)
    for i in range(n_classes):
        tpr_macro_model += np.interp(fpr_macro_model, fpr_model[i], tpr_model[i])
    tpr_macro_model /= n_classes
    roc_auc_macro_model = auc(fpr_macro_model, tpr_macro_model)
    
    # Plot
    ax.plot(fpr_macro_model, tpr_macro_model,
            label=f'{model_name} (AUC = {roc_auc_macro_model:.3f})',
            color=model_colors[model_name], lw=2.5)

# Plot random classifier
ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison - ROC Curves (Macro-Average)', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
ax.grid(alpha=0.3, linestyle='--')




plt.tight_layout()
plt.savefig(plot_dir /'combined_roc_auc_comparison.png', dpi=300, bbox_inches='tight')
print("Combined ROC/AUC comparison plot saved as 'combined_roc_auc_comparison.png'")
plt.close()


# Save the Random Forest model and encoders
joblib.dump(rf_grid, 'random_forest_model.pkl')
joblib.dump(ohe, 'one_hot_encoder.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Random Forest model and encoders saved successfully!")



# def random_forest_model(state_name, city_name, month, day_of_week, weather_conditions):
    
#     # Create a DataFrame with the input features
#     sample = pd.DataFrame({
#         "State Name": [state_name],
#         "City Name": [city_name],
#         "Month": [month],
#         "Day of Week": [day_of_week],
#         "Weather Conditions": [weather_conditions]
#     })
    
#     # Encode the features using the fitted OneHotEncoder
#     sample_encoded = ohe.transform(sample)
    
#     # Make prediction using the trained Random Forest model
#     pred_encoded = rf_grid.predict(sample_encoded)[0]
    
#     # Decode the predicted label using the fitted LabelEncoder
#     pred_label = le.inverse_transform([pred_encoded])[0]
    
#     return pred_label


# # Example usage
# if __name__ == "__main__":
#     # Test the function
#     result = random_forest_model(
#         state_name="Maharashtra",
#         city_name="Pune",
#         month="January",
#         day_of_week="Monday",
#         weather_conditions="Clear"
#     )
#     print(f"Predicted Accident Location Details: {result}")


