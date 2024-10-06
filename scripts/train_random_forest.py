import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

print(os.getcwd()) 
# Load the dataset
data = pd.read_csv('app/Data/Crop_recommendation.csv')
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Create the models directory if not exists
os.makedirs('app/models', exist_ok=True)

# Save the trained model
joblib.dump(model, 'app/models/RandomForest.joblib')
print("Model saved successfully!")