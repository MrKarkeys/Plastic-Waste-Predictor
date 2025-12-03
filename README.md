# Plastic-Waste-Predictor

A web application that predicts plastic waste and pollution based on production amounts using machine learning and mathematical models.

## Features

- Predict plastic waste and pollution based on production amounts
- Interactive visualization with historical data
- Environmental impact analysis using Google Gemini AI
- Three different prediction models: Linear, Polynomial, and Random Forest

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm (Node Package Manager)

## Installation

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
   
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a .env file and add your Gemini API Key in this format:
   ```bash
   GEMINI_API_KEY={Add Your Key Here}
   ```
5. Start the backend server:
   ```bash
   python server.py
   ```
   The server will run on http://localhost:8000

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The application will be available at http://localhost:5173

## Usage

1. Open your web browser and navigate to http://localhost:5173
2. Choose between Waste or Pollution prediction
3. Select Model 1 (Linear), Model 2 (Polynomial), Model 3 (Random Forest)
4. Enter the production amount in million tonnes
5. Click 'Predict' to see the results
6. View the prediction on the interactive graph and read the environmental impacts, solutions, and model information
