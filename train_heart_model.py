import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Step 1: Load data
df = pd.read_csv('../datasets/heart.csv')

# Step 2: Split features & labels
X = df.drop(columns=['target'])
y = df['target']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 6: Save model
os.makedirs('../saved_models', exist_ok=True)
with open('../saved_models/heart.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Heart model saved successfully!")