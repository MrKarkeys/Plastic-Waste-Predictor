from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

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
    # Use os.path to handle file paths correctly
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xdata = np.genfromtxt('Preprocessed_Data/global-plastics-production-pollution.csv', delimiter=',', skip_header=1)
    pollution_x = xdata[:, 1]

    ydata = np.genfromtxt('Preprocessed_Data/plastic-fate.csv', delimiter=',', skip_header=1)
    pollution_y = ydata[:, 1]
    
    xdata = np.genfromtxt('Preprocessed_Data/global-plastics-production-waste.csv', delimiter=',', skip_header=1)
    waste_x = xdata[:, 1]

    ydata = np.genfromtxt('Preprocessed_Data/plastic-waste-by-sector.csv', delimiter=',', skip_header=1)
    waste_y = ydata[:, 1]
    
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
        
        # Get environmental impact analysis from OpenAI
        impact_prompt = f"What are the impacts of {prediction:.2f} million tonnes of plastic {request.predictionType} and what are ways to combat it? Please provide a concise response with two sections: 1) Environmental Impacts and 2) Solutions"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": impact_prompt}],
                max_tokens=300,
                temperature=0.7
            )
            impact_analysis = response.choices[0].message.content
        except Exception as e:
            impact_analysis = "Unable to generate environmental impact analysis at this time."
        
        return {
            "prediction": prediction,
            "graphData": graph_data,
            "impactAnalysis": impact_analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)