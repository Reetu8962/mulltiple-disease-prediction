import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Step 1: Load data
df = pd.read_csv('../datasets/kidney.csv')

# Step 2: Clean nulls & convert categories
df = df.dropna()

# Convert all categorical string values to numeric using label encoding
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].astype('category').cat.codes

# Step 3: Split features and labels
X = df.drop(columns=['classification'])
y = df['classification'].apply(lambda x: 1 if x == 1 else 0)  # already converted above

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 7: Save model
os.makedirs('../saved_models', exist_ok=True)
joblib.dump(model, 'C:/Users/Ritu Kunvar/Desktop/multiple disease prediction/saved_models/kidney.pkl')
print("✅ Kidney model saved successfully!")