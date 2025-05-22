import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('boston.csv')

# Split features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
print("Training model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
print("Saving model and scaler...")
with open('regmodel.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler have been saved successfully!")

# Print model performance
y_pred = model.predict(X_test_scaled)
mse = np.mean((y_test - y_pred) ** 2)
rmse = np.sqrt(mse)
print(f"\nModel Performance:")
print(f"Root Mean Squared Error: {rmse:.2f}") 