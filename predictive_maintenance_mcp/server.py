from fastmcp import FastMCP
import pandas as pd
import numpy as np
import joblib
import os

mcp = FastMCP("Industry-Grade Predictive Maintenance")

@mcp.tool()
def get_sensor_data(equipment_id: int) -> list:
    # In production, query from sensor DB/stream
    data = pd.DataFrame({
        "timestamp": pd.date_range("now", periods=10, freq="5min").astype(str),
        "temperature": np.random.normal(70, 2, 10),
        "vibration": np.random.normal(0.02, 0.01, 10),
        "equipment_id": [equipment_id] * 10
    })
    return data.to_dict(orient="records")

@mcp.tool()
def predict_failure(temperature: float, vibration: float) -> float:
    model_path = os.path.join(os.path.dirname(__file__), "rf_model.joblib")
    model = joblib.load(model_path)
    features = [[temperature, vibration]]
    proba = model.predict_proba(features)[0][1]
    return float(proba)
    
if __name__ == "__main__":
    mcp.run()
