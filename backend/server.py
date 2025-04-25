from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    productionAmount: float
    predictionType: str
    modelType: str

# Load and prepare data
def load_data():
    pollution_data = pd.read_csv('Preprocessed_Data/global-plastics-production-pollution.csv')
    waste_data = pd.read_csv('Preprocessed_Data/global-plastics-production-waste.csv')
    
    pollution_x = pollution_data['Global plastics production (million tonnes)'].values.reshape(-1, 1)
    pollution_y = pollution_data['Global plastics production (million tonnes)'].values
    
    waste_x = waste_data['Global plastics production (million tonnes)'].values.reshape(-1, 1)
    waste_y = waste_data['Global plastics production (million tonnes)'].values
    
    return pollution_x, pollution_y, waste_x, waste_y

# Initialize models
pollution_x, pollution_y, waste_x, waste_y = load_data()

# Linear regression models
pollution_model = LinearRegression().fit(pollution_x, pollution_y)
waste_model = LinearRegression().fit(waste_x, waste_y)

# Polynomial models (degree 5)
def fit_polynomial(x, y, degree=5):
    coeffs = np.polyfit(x.flatten(), y, degree)
    return coeffs

pollution_poly_coeffs = fit_polynomial(pollution_x, pollution_y)
waste_poly_coeffs = fit_polynomial(waste_x, waste_y)

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    try:
        production = np.array([[request.productionAmount]])
        
        if request.predictionType == "pollution":
            x, y = pollution_x, pollution_y
            linear_model = pollution_model
            poly_coeffs = pollution_poly_coeffs
        else:  # waste
            x, y = waste_x, waste_y
            linear_model = waste_model
            poly_coeffs = waste_poly_coeffs
        
        # Make prediction
        if request.modelType == "linear":
            prediction = float(linear_model.predict(production)[0])
            y_pred = linear_model.predict(x)
        else:  # polynomial
            prediction = float(np.polyval(poly_coeffs, request.productionAmount))
            y_pred = np.polyval(poly_coeffs, x.flatten())
        
        # Prepare graph data
        graph_data = [
            {"production": float(x[i][0]), 
             "original": float(y[i]), 
             "predicted": float(y_pred[i])}
            for i in range(len(x))
        ]
        
        # Add user input point
        graph_data.append({
            "production": float(request.productionAmount),
            "userInput": float(prediction)
        })
        
        return {
            "prediction": prediction,
            "graphData": graph_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)