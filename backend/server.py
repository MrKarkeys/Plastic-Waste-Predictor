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
model_info = ""

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

#currying for RSS and AIC
def RSS_y(y):
    def RSS_y_pred(y_pred):
        return np.sum(np.power(y - y_pred, 2))
    return RSS_y_pred

def AIC_RSS(RSS):
    def AIC_n(n):
        def AIC_k(k):
            return n*np.log(RSS/n)+2*(k+1)
        return AIC_k
    return AIC_n

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
        
        #calculate RSS
        RSS_w_y = RSS_y(y)
        RSS = RSS_w_y(np.polyval(coefficients, x))

        #calculate AIC
        AIC_w_RSS = AIC_RSS(RSS)
        AIC_w_n = AIC_w_RSS(len(x)) 
        AIC = AIC_w_n(len(coefficients))
        
        #add RSS to arrays
        RSSvals.append(RSS)
        AICvals.append(AIC)

        results.append((i, coefficients, RSS, AIC, xdata, final_y))
    return results

def polynomial_fit():
    pollution_x, pollution_y, waste_x, waste_y = load_data()
    degrees = 5
    AICvals_pollution, RSSvals_pollution = [], []
    pollution_results = poly_fit_helper(pollution_x, pollution_y, degrees, AICvals_pollution, RSSvals_pollution)

    # using functional programming (map and lambdas)
    pollution_stored_coefficients = list(map(lambda res: res[1], pollution_results))
    result_summary_arr_pollution = list(map(lambda res: f"DEGREE: {res[0]} COEFFICIENTS: {res[1]} \n RSS: {res[2]} \n AIC: {res[3]}", pollution_results))

    AICvals_waste, RSSvals_waste = [], []
    waste_results = poly_fit_helper(waste_x, waste_y, degrees, AICvals_waste, RSSvals_waste)

    # using functional programming (map and lambdas)
    waste_stored_coefficients = list(map(lambda res: res[1], waste_results))
    result_summary_arr_waste = list(map(lambda res: f"DEGREE: {res[0]} COEFFICIENTS: {res[1]} \n RSS: {res[2]} \n AIC: {res[3]}", waste_results))

    return result_summary_arr_pollution, result_summary_arr_waste, AICvals_pollution, RSSvals_pollution, AICvals_waste, RSSvals_waste, pollution_stored_coefficients, waste_stored_coefficients


# def polynomial_fit():
#     pollution_x, pollution_y, waste_x, waste_y = load_data()
#     degrees = 5
#     AICvals_pollution, RSSvals_pollution = [], []
#     pollution_results = poly_fit_helper(pollution_x, pollution_y, degrees, AICvals_pollution, RSSvals_pollution)
#     pollution_stored_coefficients = [res[1] for res in pollution_results]
#     result_summary_arr_pollution = [f"DEGREE: {res[0]} COEFFICIENTS: {res[1]} \n RSS: {res[2]} \n AIC: {res[3]}" for res in pollution_results]

#     AICvals_waste, RSSvals_waste = [], []
#     waste_results = poly_fit_helper(waste_x, waste_y, degrees, AICvals_waste, RSSvals_waste)
#     waste_stored_coefficients = [res[1] for res in waste_results]
#     result_summary_arr_waste = [f"DEGREE: {res[0]} COEFFICIENTS: {res[1]} \n RSS: {res[2]} \n AIC: {res[3]}" for res in waste_results]

#     return result_summary_arr_pollution, result_summary_arr_waste, AICvals_pollution, RSSvals_pollution, AICvals_waste, RSSvals_waste, pollution_stored_coefficients, waste_stored_coefficients

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
            print(result_summary_arr)
        else:
            x, y = waste_x.reshape(-1, 1), waste_y
            linear_model = linear_regression()[1]
            random_forest_model = random_forest_model_fit()[1]
            _, result_summary_arr, _, _, AICvals, RSSvals, _, stored_coeffs = polynomial_fit()
            print(result_summary_arr)

        model_num = pick_best_model(AICvals)
        AIC_poly = AICvals[model_num]
        RSS_poly = RSSvals[model_num]
        COEFFICIENTS = stored_coeffs[model_num]

        if request.modelType == "linear":
            prediction = float(linear_model.predict(production)[0])
            y_pred = linear_model.predict(x)

            #calculate RSS
            RSS_w_y_linear = RSS_y(y)
            RSS = RSS_w_y_linear(y_pred)

            #calculate AIC
            AIC_w_RSS_linear = AIC_RSS(RSS)
            AIC_w_n_linear = AIC_w_RSS_linear(len(y)) 
            AIC = AIC_w_n_linear(len(linear_model.coef_))

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

            #calculate RSS
            RSS_w_y_linear = RSS_y(y)
            RSS = RSS_w_y_linear(y_pred)

            model_info = f"\nModel Type: Random Forest\nRSS: {RSS:.4f}"

        graph_data = list(map(lambda i: {"production": float(x[i][0]), "original": float(y[i])}, range(len(x))))
        graph_data.append({"production": float(request.productionAmount), "userInput": float(prediction)})
        graph_data = sorted(graph_data, key=lambda d: d["production"])

        return {
            "prediction": prediction,
            "graphData": graph_data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/environmental-impacts")
async def get_environmental_impacts(request: PredictionRequest):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        impact_prompt = f"""
            You are an environmental analyst. Given the predicted plastic {request.predictionType} of **{request.productionAmount:.2f} million tonnes**, generate a detailed analysis of environmental impacts.

            Focus on:
            - Begin with a short summary sentence
            - List several key environmental concerns, each starting with a **bolded label** (e.g., **Ocean Pollution:**)
            - Use bullet points for clarity
            - Focus only on environmental impacts, not solutions

            Format the output in clean markdown using:
            - Bolded labels within bullet points
            - Clear line breaks between points
            """
        
        response = model.generate_content(impact_prompt)
        return {"environmentalImpacts": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/solutions")
async def get_solutions(request: PredictionRequest):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        solutions_prompt = f"""
            You are an environmental analyst. Given the predicted plastic {request.predictionType} of **{request.productionAmount:.2f} million tonnes**, generate practical and policy-based solutions.

            Focus on:
            - Begin with a transition sentence
            - List both practical and policy-based solutions
            - Each solution should start with a **bolded label**
            - Use bullet points for clarity
            - Focus only on solutions, not impacts

            Format the output in clean markdown using:
            - Bolded labels within bullet points
            - Clear line breaks between points
            """
        
        response = model.generate_content(solutions_prompt)
        return {"solutions": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model-info")
async def get_model_info(request: PredictionRequest):
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

        if request.modelType == "linear":
            y_pred = linear_model.predict(x)
            RSS = calcRSS(y, y_pred)
            AIC = calcAIC(RSS, len(y), len(linear_model.coef_))
            COEFFICIENTS = [linear_model.coef_[0], linear_model.intercept_]
            model_info = f"\nModel Type: Linear Regression\nRSS: {RSS:.4f}\nAIC: {AIC:.4f}\nCoefficients: {COEFFICIENTS}"
        elif request.modelType == "polynomial":
            y_pred = np.polyval(COEFFICIENTS, x)
            RSS = RSS_poly
            AIC = AIC_poly
            model_info = f"\nModel Type: Polynomial Regression\nRSS: {RSS:.4f}\nAIC: {AIC:.4f}\nCoefficients: {COEFFICIENTS}"
        else:
            y_pred = random_forest_model.predict(x)
            RSS = calcRSS(y, y_pred)
            model_info = f"\nModel Type: Random Forest\nRSS: {RSS:.4f}"

        model = genai.GenerativeModel("gemini-2.0-flash")
        model_prompt = f"""
            You are an environmental analyst. Given the predicted plastic {request.predictionType} of **{request.productionAmount:.2f} million tonnes**, analyze the following model information:

            {model_info}

            Please provide a clear interpretation of:
            What the model type means for this prediction
            The significance of the RSS and AIC values
            What the coefficients tell us about the relationship between production and {request.predictionType}
            Any limitations or considerations of the model

            Format the output in clean markdown using:
            - Bolded labels within bullet points
            - Clear line breaks between points
            """
        
        response = model.generate_content(model_prompt)
        return {"modelInfo": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
