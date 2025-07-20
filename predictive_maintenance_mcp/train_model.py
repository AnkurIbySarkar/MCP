import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Generate (or load) labeled sensor data
# Columns: temperature (°C), vibration (units), failure (0/1)
data = pd.DataFrame({
    "temperature": np.random.normal(70, 10, 1000),
    "vibration": np.random.normal(0.02, 0.01, 1000),
    "failure": np.random.binomial(1, 0.1, 1000)
})

X = data[["temperature", "vibration"]].values
y = data["failure"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
preds = model.predict(X_test)
print(classification_report(y_test, preds))

joblib.dump(model, "rf_model.joblib")
