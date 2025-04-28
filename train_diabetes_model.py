import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Step 1: Dataset load karo (ye file same folder me honi chahiye)
df = pd.read_csv("datasets/diabetes.csv")

# Step 2: Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model train karo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Save model
# Folder bana lo agar nahi hai
os.makedirs("saved_models", exist_ok=True)

# Model save karo
with open("saved_models/diabetes.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Diabetes model trained and saved successfully!")