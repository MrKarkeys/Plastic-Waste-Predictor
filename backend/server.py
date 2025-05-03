from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
import google.generativeai as genai

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_api_key)

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xdata = np.genfromtxt(os.path.join(base_dir, 'Preprocessed_Data/global-plastics-production-pollution.csv'), delimiter=',', skip_header=1)
    pollution_x = xdata[:, 1]

    ydata = np.genfromtxt(os.path.join(base_dir, 'Preprocessed_Data/plastic-fate.csv'), delimiter=',', skip_header=1)
    pollution_y = ydata[:, 1]

    xdata = np.genfromtxt(os.path.join(base_dir, 'Preprocessed_Data/global-plastics-production-waste.csv'), delimiter=',', skip_header=1)
    waste_x = xdata[:, 1]

    ydata = np.genfromtxt(os.path.join(base_dir, 'Preprocessed_Data/plastic-waste-by-sector.csv'), delimiter=',', skip_header=1)
    waste_y = ydata[:, 1]

    return pollution_x, pollution_y, waste_x, waste_y

def calcRSS(y, y_pred):
    return np.sum(np.power(y - y_pred, 2))

def calcAIC(RSS, n, k):
    return n*np.log(RSS/n)+2*(k+1)

def linear_regression():
    pollution_x, pollution_y, waste_x, waste_y = load_data()
    pollution_x = pollution_x.reshape(-1, 1)
    waste_x = waste_x.reshape(-1, 1)
    predict_pollution_model = LinearRegression().fit(pollution_x, pollution_y)
    predict_waste_model = LinearRegression().fit(waste_x, waste_y)
    return predict_pollution_model, predict_waste_model

def pick_best_model(AICvals):
    if not AICvals:
        return None
    best = 0
    for i in range(len(AICvals)):
        if AICvals[i] <= AICvals[best]-2:
            best = i
    return best

def linear_fit2(x, y):
    return np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))

def poly_fit_helper(x, y, num_degrees, AICvals, RSSvals):
    results = []
    for i in range(1, num_degrees+1):
        xdata = np.linspace(min(x), max(x), 500)
        new_x = np.vander(x, i+1)
        coefficients = linear_fit2(new_x, y)
        final_y = np.polyval(coefficients, xdata)
        RSS = calcRSS(y, np.polyval(coefficients, x))
        AIC = calcAIC(RSS, len(x), len(coefficients))
        RSSvals.append(RSS)
        AICvals.append(AIC)
        results.append((i, coefficients, RSS, AIC, xdata, final_y))
    return results

def polynomial_fit():
    pollution_x, pollution_y, waste_x, waste_y = load_data()
    degrees = 5
    AICvals_pollution, RSSvals_pollution = [], []
    pollution_results = poly_fit_helper(pollution_x, pollution_y, degrees, AICvals_pollution, RSSvals_pollution)
    pollution_stored_coefficients = [res[1] for res in pollution_results]
    result_summary_arr_pollution = [f"DEGREE: {res[0]} COEFFICIENTS: {res[1]} \n RSS: {res[2]} \n AIC: {res[3]}" for res in pollution_results]

    AICvals_waste, RSSvals_waste = [], []
    waste_results = poly_fit_helper(waste_x, waste_y, degrees, AICvals_waste, RSSvals_waste)
    waste_stored_coefficients = [res[1] for res in waste_results]
    result_summary_arr_waste = [f"DEGREE: {res[0]} COEFFICIENTS: {res[1]} \n RSS: {res[2]} \n AIC: {res[3]}" for res in waste_results]

    return result_summary_arr_pollution, result_summary_arr_waste, AICvals_pollution, RSSvals_pollution, AICvals_waste, RSSvals_waste, pollution_stored_coefficients, waste_stored_coefficients

def random_forest_model_fit():
    pollution_x, pollution_y, waste_x, waste_y = load_data()
    pollution_x = pollution_x.reshape(-1, 1)
    waste_x = waste_x.reshape(-1, 1)
    predict_pollution_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(pollution_x, pollution_y)
    predict_waste_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(waste_x, waste_y)
    return predict_pollution_model, predict_waste_model

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    try:
        production = np.array([[request.productionAmount]])
        pollution_x, pollution_y, waste_x, waste_y = load_data()

        if request.predictionType == "pollution":
            x, y = pollution_x.reshape(-1, 1), pollution_y
            linear_model = linear_regression()[0]
            random_forest_model = random_forest_model_fit()[0]
            result_summary_arr, _, AICvals, RSSvals, _, _, stored_coeffs, _ = polynomial_fit()
        else:
            x, y = waste_x.reshape(-1, 1), waste_y
            linear_model = linear_regression()[1]
            random_forest_model = random_forest_model_fit()[1]
            _, result_summary_arr, _, _, AICvals, RSSvals, _, stored_coeffs = polynomial_fit()

        model_num = pick_best_model(AICvals)
        AIC_poly = AICvals[model_num]
        RSS_poly = RSSvals[model_num]
        COEFFICIENTS = stored_coeffs[model_num]

        model_info = ""

        if request.modelType == "linear":
            prediction = float(linear_model.predict(production)[0])
            y_pred = linear_model.predict(x)
            RSS = calcRSS(y, y_pred)
            AIC = calcAIC(RSS, len(y), len(linear_model.coef_))
            COEFFICIENTS = [linear_model.coef_[0], linear_model.intercept_]
            model_info = f"\nModel Type: Linear Regression\nRSS: {RSS:.4f}\nAIC: {AIC:.4f}\nCoefficients: {COEFFICIENTS}"
        elif request.modelType == "polynomial":
            prediction = float(np.polyval(COEFFICIENTS, request.productionAmount))
            y_pred = np.polyval(COEFFICIENTS, x)
            RSS = RSS_poly
            AIC = AIC_poly
            model_info = f"\nModel Type: Polynomial Regression\nRSS: {RSS:.4f}\nAIC: {AIC:.4f}\nCoefficients: {COEFFICIENTS}"
        else:
            prediction = float(random_forest_model.predict(production)[0])
            y_pred = random_forest_model.predict(x)
            RSS = calcRSS(y, y_pred)
            model_info = f"\nModel Type: Random Forest\nRSS: {RSS:.4f}"

        graph_data = [{"production": float(x[i][0]), "original": float(y[i])} for i in range(len(x))]
        graph_data.append({"production": float(request.productionAmount), "userInput": float(prediction)})
        graph_data = sorted(graph_data, key=lambda d: d["production"])

        impact_prompt = f"""
You are an environmental analyst. Given the plastic {request.predictionType} of {prediction:.2f} million tonnes, provide the following:

**1) Environmental Impacts:**  
List several key environmental concerns, each starting with a bolded label (e.g., **Ocean Pollution:**).

**2) Solutions:**  
List practical and policy-based solutions, each also beginning with a bolded label.

Model Information: {model_info}

Use markdown formatting for readability (bold, line breaks, and bullet points if needed).
"""

        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(impact_prompt)
            impact_analysis = response.text
            print(impact_analysis)
        except Exception as e:
            print(f"Gemini API Error: {e}")
            impact_analysis = "Environmental impact analysis is currently unavailable."

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
