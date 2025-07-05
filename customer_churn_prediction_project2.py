# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
import os

warnings.filterwarnings("ignore")

# ------------------ MongoDB Setup ------------------
client = MongoClient("mongodb://localhost:27017")  # Update if using MongoDB Atlas
db = client["churn_db"]
collection = db["customers"]

# Check if collection is empty before uploading
if collection.count_documents({}) == 0:
    print("Uploading CSV data to MongoDB...")
    csv_path = "C:\\Users\\Ritwik Bhandari\\Downloads\\Churn_Modelling.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError("CSV file not found. Please check the path.")

    df = pd.read_csv(csv_path)
    collection.insert_many(df.to_dict("records"))
    print("Upload completed.")

# ------------------ Load Data from MongoDB ------------------
print("Loading data from MongoDB...")
data = pd.DataFrame(list(collection.find()))

# Drop MongoDB’s default _id field
if '_id' in data.columns:
    data = data.drop(columns=['_id'])

# ------------------ Summary Statistics and Visuals ------------------
print("\nSummary Statistics:")
print(data.describe(include='all'))

# Target variable count
plt.figure(figsize=(6, 4))
sns.countplot(x='Exited', data=data)
plt.title("Customer Churn Count")
plt.xlabel("Exited (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
corr = data.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Distribution of key numeric features
numeric_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Boxplots by churn status
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Exited', y=col, data=data)
    plt.title(f"{col} vs Exited")
    plt.xlabel("Exited")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# Categorical count plots (e.g., Geography, Gender)
categorical_cols = ['Geography', 'Gender']
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue='Exited', data=data)
    plt.title(f"{col} Distribution by Exited")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# ------------------ Data Preprocessing ------------------
# Drop unnecessary columns
data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Split features and target
x = data.drop('Exited', axis=1)
y = data['Exited']

# Drop rows where target is NaN
combined = pd.concat([x, y], axis=1)
combined = combined.dropna(subset=['Exited'])

# Separate again
x = combined.drop('Exited', axis=1)
y = combined['Exited']

# Encode categorical features
x = pd.get_dummies(x)

# Fill numeric NaNs with mean (if any)
x = x.fillna(x.mean(numeric_only=True))

# Fill categorical NaNs with mode (if any)
x = x.fillna(x.mode().iloc[0])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# ------------------ Random Forest ------------------
print("\nRandom Forest Classifier")
model_rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
model_rf.fit(x_train, y_train)
y_pred_rf = model_rf.predict(x_test)
print(f"Accuracy: {model_rf.score(x_test, y_test) * 100:.2f}%")
print("Unique predictions:", np.unique(y_pred_rf))
print(classification_report(y_test, y_pred_rf, labels=[0, 1]))

# ------------------ PCA + Random Forest ------------------
print("\nPCA + Random Forest")
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

pca = PCA(0.9)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

model_pca = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
model_pca.fit(x_train_pca, y_train)
y_pred_pca = model_pca.predict(x_test_pca)
print(f"Accuracy: {model_pca.score(x_test_pca, y_test) * 100:.2f}%")
print("Unique predictions:", np.unique(y_pred_pca))
print(classification_report(y_test, y_pred_pca, labels=[0, 1]))

# ------------------ Logistic Regression ------------------
print("\nLogistic Regression")
model_log = LogisticRegression(max_iter=1000)
model_log.fit(x_train, y_train)
y_pred_log = model_log.predict(x_test)
print(f"Accuracy: {model_log.score(x_test, y_test) * 100:.2f}%")
print("Unique predictions:", np.unique(y_pred_log))
print(classification_report(y_test, y_pred_log, labels=[0, 1]))


import pickle

with open("model_pca.pkl", "wb") as f:
    pickle.dump(model_pca, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("pca.pkl", "wb") as f:
    pickle.dump(pca, f)

with open("features.pkl", "wb") as f:
    pickle.dump(list(x.columns), f)


print("Model, scaler, and PCA saved.")
