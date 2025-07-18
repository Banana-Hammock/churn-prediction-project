# -*- coding: utf-8 -*-
"""Predicting Human Personality

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/ritwikbhandari/predicting-human-personality.20c3b58b-c13f-4d6d-9f18-737b99e8372a.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20250611/auto/storage/goog4_request%26X-Goog-Date%3D20250611T042511Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3Def8510a6e4c40d6b0fac47fbf73b6ecaed126a142a766583440fd7559e4ee83b8edbad3774e4b9756784200883131ea7704dfa9db5d8136695ccae8354cc3cafffe98676632beaed8cfbee683f9ae6d3cdc10624a97ee7ab61ce2932804d4d79f1249d734d89a84c2a1be7f64b1ceab4d45e516e84b573180f1f3a4546bde9618d5cab5bb436359bd7de4bc4d7f9019110feb10b3525bf60137e70f6cfb3a993f048aa001fc2d01d67f0fb8a5e501dcc529690c4faa829b563e23a2e05b5210d88d7bbfe8c8d1ceba32d4f7aa712946c0313583e95f3cb9c66599e0e03eec59c08f380c2d7d5236664eb3572b8ea3dd994785c2475e7ca3cb35e1178adce541a
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
rakeshkapilavai_extrovert_vs_introvert_behavior_data_path = kagglehub.dataset_download('rakeshkapilavai/extrovert-vs-introvert-behavior-data')

print('Data source import complete.')

"""# Personality Classification Notebook

## Table of Contents
1. Introduction
2. Exploratory Data Analysis
3. Data Preprocessing
4. Feature Engineering
5. Model Training and Evaluation
6. Feature Importance and Interpretability
7. Conclusion

## Introduction
This notebook classifies individuals as Introverts or Extroverts based on social behavior features from the personality dataset. We use advanced preprocessing, feature engineering, and a stacking ensemble with Random Forest, Gradient Boosting, XGBoost, and SVM, optimized with extensive hyperparameter tuning. The goal is to achieve high performance on the competition metric (e.g., F1-score or AUC) and create a polished notebook for Kaggle community engagement to secure a silver medal.
"""

# Commented out IPython magic to ensure Python compatibility.
## Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, f1_score
import os
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

# Set random seed
np.random.seed(42)

"""## Exploratory Data Analysis
We explore the dataset to understand feature distributions, class balance, and correlations, which guide preprocessing and modeling.
"""

# Load dataset
data_path = '/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Please check the file path.")
data = pd.read_csv(data_path)

# Display basic information
print("Dataset Shape:", data.shape)
print("\nDataset Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Define numeric and categorical columns
numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_columns = ['Stage_fear', 'Drained_after_socializing']
target_column = 'Personality'

# Verify categorical values
for col in categorical_columns:
    print(f"\nUnique values in {col}:")
    print(data[col].value_counts(dropna=False))

# Class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x=target_column, data=data)
plt.title('Class Distribution of Personality Types')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.savefig('/kaggle/working/class_distribution.png')
plt.show()

# Numeric feature distributions by class
plt.figure(figsize=(8, 6))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x=target_column, y=col, data=data)
    plt.title(f'{col} by Personality')
plt.tight_layout()
plt.savefig('/kaggle/working/box_plots.png')
plt.show()

# Pair plot for key features
sns.pairplot(data[numeric_columns + [target_column]], hue=target_column, diag_kind='hist')
plt.suptitle('Pair Plot of Numeric Features by Personality', y=1.02)
plt.savefig('/kaggle/working/pair_plot.png')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numeric Features')
plt.savefig('/kaggle/working/correlation_heatmap.png')
plt.show()

"""## Data Preprocessing
We handle missing values, encode categorical features, cap outliers, and apply SMOTE and scaling to prepare the data for modeling.
"""

# Encode target variable
le = LabelEncoder()
data[target_column] = le.fit_transform(data[target_column])
print(f"Encoded classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Handle missing values
# Numeric features: impute with median
numeric_imputer = SimpleImputer(strategy='median')
data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

# Categorical features: impute with mode
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

# Encode categorical features
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Update feature list
encoded_columns = [col for col in data.columns if col != target_column]

# Cap outliers instead of removing
for col in numeric_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

# Train-test split
X = data.drop(columns=[target_column])
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""## Feature Engineering
We create interaction features, polynomial features, and binned features to capture complex relationships.
"""

# Interaction features
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
X_train['Alone_to_Social_Ratio'] = X_train['Time_spent_Alone'] / (X_train['Social_event_attendance'] + 1)
X_test['Alone_to_Social_Ratio'] = X_test['Time_spent_Alone'] / (X_test['Social_event_attendance'] + 1)
X_train['Social_Comfort_Index'] = (X_train['Friends_circle_size'] + X_train['Post_frequency'] - X_train['Stage_fear_Yes']) / 3
X_test['Social_Comfort_Index'] = (X_test['Friends_circle_size'] + X_test['Post_frequency'] - X_test['Stage_fear_Yes']) / 3
X_train['Social_Overload'] = X_train['Drained_after_socializing_Yes'] * X_train['Social_event_attendance']
X_test['Social_Overload'] = X_test['Drained_after_socializing_Yes'] * X_test['Social_event_attendance']

# Binned features
X_train['Time_spent_Alone_Binned'] = pd.qcut(X_train['Time_spent_Alone'], q=3, labels=['Low', 'Medium', 'High'])
X_test['Time_spent_Alone_Binned'] = pd.qcut(X_test['Time_spent_Alone'], q=3, labels=['Low', 'Medium', 'High'])
X_train = pd.get_dummies(X_train, columns=['Time_spent_Alone_Binned'], drop_first=True)
X_test = pd.get_dummies(X_test, columns=['Time_spent_Alone_Binned'], drop_first=True)

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
poly_features_train = poly.fit_transform(X_train[['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']])
poly_features_test = poly.transform(X_test[['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size']])
poly_feature_names = poly.get_feature_names_out(['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size'])
X_train[poly_feature_names] = poly_features_train
X_test[poly_feature_names] = poly_features_test

# Update scaled features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""## Model Training and Evaluation
We train multiple models with hyperparameter tuning and use a stacking ensemble for final predictions.
"""

# Define models with expanded hyperparameter grids
models = {
    'logistic': {
        'model': LogisticRegression(max_iter=1000),
        'use_scaled': True,
        'params': {'C': np.logspace(-4, 4, 20), 'solver': ['lbfgs', 'liblinear']}
    },
    'svm': {
        'model': SVC(probability=True),
        'use_scaled': True,
        'params': {'C': np.logspace(-3, 3, 20), 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto', 0.1, 1]}
    },
    'rf': {
        'model': RandomForestClassifier(random_state=42),
        'use_scaled': False,
        'params': {'n_estimators': [100, 150], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
    },
    'gb': {
        'model': GradientBoostingClassifier(random_state=42),
        'use_scaled': False,
        'params': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'use_scaled': True,
        'params': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
    },
    'dt': {
        'model': DecisionTreeClassifier(random_state=42),
        'use_scaled': False,
        'params': {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10]}
    },
    'xgb': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'use_scaled': False,
        'params': {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5], 'subsample': [0.8]}
    }
}

# Hyperparameter tuning
results = []
for model_name, mp in models.items():
    clf = RandomizedSearchCV(mp['model'], mp['params'], n_iter=20, cv=3, scoring='f1_weighted', n_jobs=-1, random_state=42)
    X_train_current = X_train_scaled if mp['use_scaled'] else X_train
    clf.fit(X_train_current, y_train)
    results.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

# Display model performance
df_results = pd.DataFrame(results)
print("\nModel Performance (Cross-Validation F1-Weighted Scores):")
print(df_results)

# Stacking ensemble
estimators = [
    ('rf', RandomForestClassifier(**df_results.loc[df_results['model'] == 'rf']['best_params'].iloc[0], random_state=42)),
    ('gb', GradientBoostingClassifier(**df_results.loc[df_results['model'] == 'gb']['best_params'].iloc[0], random_state=42)),
    ('xgb', XGBClassifier(**df_results.loc[df_results['model'] == 'xgb']['best_params'].iloc[0], random_state=42, eval_metric='logloss')),
    ('svm', SVC(**df_results.loc[df_results['model'] == 'svm']['best_params'].iloc[0], probability=True))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1)
stacking_model.fit(X_train_scaled, y_train)

# Cross-validation for stacking model
stacking_scores = cross_val_score(stacking_model, X_train_scaled, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
print(f"\nStacking Model Cross-Validation F1-Weighted Score: {stacking_scores.mean():.3f} ± {stacking_scores.std():.3f}")

# Evaluate on test set
y_pred = stacking_model.predict(X_test_scaled)
print("\nStacking Model Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Stacking Ensemble')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('/kaggle/working/confusion_matrix.png')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, stacking_model.predict_proba(X_test_scaled)[:, 1])
roc_auc = roc_auc_score(y_test, stacking_model.predict_proba(X_test_scaled)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Stacking Ensemble')
plt.legend()
plt.savefig('/kaggle/working/roc_curve.png')
plt.show()

"""## Feature Importance and Interpretability
We analyze feature importance using Random Forest and SHAP to understand model predictions.
"""

# Random Forest feature importance
rf_model = RandomForestClassifier(**df_results.loc[df_results['model'] == 'rf']['best_params'].iloc[0], random_state=42)
rf_model.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
feature_importance.to_csv('/kaggle/working/feature_importance.csv', index=False)
print("\nFeature Importance (Random Forest):")
print(feature_importance)

# Feature importance bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Random Forest Feature Importance')
plt.savefig('/kaggle/working/feature_importance_plot.png')
plt.show()

# SHAP values (using a sample for efficiency)
X_test_sample = X_test.sample(frac=0.1, random_state=42)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_sample)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_sample, feature_names=X_test.columns, show=False)
plt.title('SHAP Feature Importance for Random Forest')
plt.savefig('/kaggle/working/shap_summary.png')
plt.show()

# SHAP summary for both classes
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[1], X_test_sample, feature_names=X_test.columns, show=False)
plt.title('SHAP Values for Extrovert Class')
plt.savefig('/kaggle/working/shap_extrovert.png')
plt.show()

"""## Conclusion
This notebook demonstrates a robust approach to personality classification using advanced preprocessing, feature engineering, and a stacking ensemble with Random Forest, Gradient Boosting, XGBoost, and SVM. Key features like `Time_spent_Alone` and `Friends_circle_size` were highly influential, as shown by feature importance and SHAP analyses. The stacking model achieved a cross-validation F1-weighted score of approximately {stacking_scores.mean():.3f}, indicating strong performance. For competition submissions, compare the test set performance to the leaderboard. For community engagement, this notebook includes comprehensive visualizations and explanations to attract upvotes.
"""

# Save model and predictions
joblib.dump(stacking_model, '/kaggle/working/stacking_model.pkl')
pd.DataFrame({'Actual': le.inverse_transform(y_test), 'Predicted': le.inverse_transform(y_pred)}).to_csv('/kaggle/working/predictions.csv', index=False)

"""# Please upvote the notebook if you like it.
![](https://image.petmd.com/files/styles/863x625/public/CANS_dogsmiling_379727605.jpg)
"""