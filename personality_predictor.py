import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# Load dataset
df = pd.read_csv("C:/Users/Ritwik Bhandari/Downloads/personality_dataset.csv")

# Encode target
le = LabelEncoder()
df['Personality'] = le.fit_transform(df['Personality'])

# Columns
numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
                      'Friends_circle_size', 'Post_frequency']
categorical_features = ['Stage_fear', 'Drained_after_socializing']
target = 'Personality'

# Impute missing values
df[numerical_features] = SimpleImputer(strategy='median').fit_transform(df[numerical_features])
df[categorical_features] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_features])

# Encode categorical features
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Scale numeric
df[numerical_features] = StandardScaler().fit_transform(df[numerical_features])

# Train-test split
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVC (Probabilistic)": SVC(probability=True)
}

# Evaluation
plt.figure(figsize=(14, 20))
for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    plt.subplot(len(models), 2, 2*i+1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.subplot(len(models), 2, 2*i+2)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{name} - ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

plt.tight_layout()
plt.show()



import joblib

# Using just the logistic regression model for prediction
final_model = LogisticRegression()
final_model.fit(X_train, y_train)

# Save model
joblib.dump(final_model, "model.pkl")
