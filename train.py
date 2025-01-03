import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading dataset...")
df = pd.read_csv('heart_disease_uci.csv')
print(f"Initial dataset shape: {df.shape}")

print("\nPreprocessing data...")

df = df.dropna()

categorical_mappings = {
    'sex': {'Male': 1, 'Female': 0},
    'fbs': {True: 1, False: 0},
    'exang': {True: 1, False: 0},
    'cp': {'typical angina': 0, 'atypical angina': 1, 'non-anginal': 2, 'asymptomatic': 3},
    'restecg': {'normal': 0, 'st-t abnormality': 1, 'lv hypertrophy': 2},
    'slope': {'upsloping': 0, 'flat': 1, 'downsloping': 2},
    'thal': {'normal': 0, 'fixed defect': 1, 'reversable defect': 2}
}

for col, mapping in categorical_mappings.items():
    print(f"\nUnique values in {col} before mapping: {df[col].unique()}")
    df[col] = df[col].map(mapping)
    print(f"Unique values in {col} after mapping: {df[col].unique()}")

df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
           
print("\nChecking data types:")
print(df[features].dtypes)

for feature in features:
    if df[feature].dtype == 'object':
        df[feature] = pd.to_numeric(df[feature], errors='coerce')

df = df.dropna(subset=features + ['target'])

print(f"\nFinal dataset shape after preprocessing: {df.shape}")

X = df[features]
y = df['target']

if len(X) == 0:
    raise ValueError("Dataset is empty after preprocessing!")

print(f"Number of samples for training: {len(X)}")
print(f"Feature distribution:\n{X.describe()}")

print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

print("\nTraining and evaluating models...")
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if roc_auc > best_score:
        best_score = roc_auc
        best_model = model

print(f"\nSaving the best model with ROC AUC score of {best_score:.4f}")
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

df.to_csv('heart.csv', index=False)

print("\nGenerating visualizations...")

plt.figure(figsize=(12, 8))
sns.heatmap(df[features + ['target']].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='age', hue='target', common_norm=False)
plt.title('Age Distribution by Heart Disease Status')
plt.savefig('age_distribution.png')
plt.close()

if isinstance(best_model, RandomForestClassifier):
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("\nTraining process completed successfully!")