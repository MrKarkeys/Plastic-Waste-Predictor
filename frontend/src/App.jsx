import { useState } from 'react'
import './App.css'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

function App() {
  const [predictionType, setPredictionType] = useState('waste')
  const [modelType, setModelType] = useState('model1')
  const [productionAmount, setProductionAmount] = useState('')
  const [result, setResult] = useState(null)
  const [graphData, setGraphData] = useState(null)
  const [error, setError] = useState(null)
  const [impactAnalysis, setImpactAnalysis] = useState(null)

  const handleSubmit = async () => {
    try {
      if (!productionAmount) {
        setError('Please enter a production amount')
        return
      }
      
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          productionAmount: parseFloat(productionAmount) * 1000000,
          predictionType: predictionType,
          modelType: modelType === 'model1' ? 'linear' : 'polynomial'
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data.prediction.toFixed(2))
      setGraphData(data.graphData)
      setImpactAnalysis(data.impactAnalysis)
      setError(null)
    } catch (error) {
      console.error('Error:', error)
      setError('Error occurred while making prediction')
      setResult(null)
      setGraphData(null)
    }
  }

  return (
    <div className="container">
      <header>
        <div className="recycling-icon">
          ♻️
        </div>
        <h1>Plastic Waste Predictor</h1>
      </header>

      <div className="controls">
        <div className="toggle-group">
          <button 
            style={{
              backgroundColor: predictionType === 'waste' ? '#4CAF50' : '#ffffff',
              color: predictionType === 'waste' ? '#ffffff' : '#000000',
              border: '1px solid #1a1a1a'
            }}
            onClick={() => {
            setPredictionType('waste')
            setProductionAmount('')
            setResult(null)
            setGraphData(null)
            setImpactAnalysis(null)
            setError(null)
          }}
          >
            Waste
          </button>
          <button 
            style={{
              backgroundColor: predictionType === 'pollution' ? '#4CAF50' : '#ffffff',
              color: predictionType === 'pollution' ? '#ffffff' : '#000000',
              border: '1px solid #1a1a1a'
            }}
            onClick={() => {
            setPredictionType('pollution')
            setProductionAmount('')
            setResult(null)
            setGraphData(null)
            setImpactAnalysis(null)
            setError(null)
          }}
          >
            Pollution
          </button>
        </div>

        <div className="toggle-group">
          <button 
            style={{
              backgroundColor: modelType === 'model1' ? '#4CAF50' : '#ffffff',
              color: modelType === 'model1' ? '#ffffff' : '#000000',
              border: '1px solid #1a1a1a'
            }}
            onClick={() => {
            setModelType('model1')
            setProductionAmount('')
            setResult(null)
            setGraphData(null)
            setImpactAnalysis(null)
            setError(null)
          }}
          >
            Model 1
          </button>
          <button 
            style={{
              backgroundColor: modelType === 'model2' ? '#4CAF50' : '#ffffff',
              color: modelType === 'model2' ? '#ffffff' : '#000000',
              border: '1px solid #1a1a1a'
            }}
            onClick={() => {
            setModelType('model2')
            setProductionAmount('')
            setResult(null)
            setGraphData(null)
            setImpactAnalysis(null)
            setError(null)
          }}
          >
            Model 2
          </button>
        </div>

        <div className="input-group">
          <input
            type="number"
            value={productionAmount}
            onChange={(e) => setProductionAmount(e.target.value)}
            placeholder="Enter production amount"
          />
          <button onClick={handleSubmit} className="submit-btn">
            Predict
          </button>
        </div>
      </div>

      <div className="results">
        {error ? (
          <div className="error-display">
            <p className="error-message">{error}</p>
          </div>
        ) : result && (
          <>
            <div className="result-display">
              <h2>Prediction Result</h2>
              <p>
                <strong>{predictionType === 'waste' ? 'Predicted Waste' : 'Predicted Pollution'}:</strong>
                <span style={{ marginLeft: '10px', color: '#4CAF50' }}>{(result / 1000000).toFixed(2)}</span>
                <span style={{ marginLeft: '5px' }}>million tonnes</span>
              </p>
            </div>
            {graphData && (
              <>
                <div className="visualization" style={{ width: '100%', height: '500px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={graphData}
                      margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="production"
                        label={{ value: 'Production (million tonnes)', position: 'insideBottom', offset: -10 }}
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => value.toLocaleString()}
                      />
                      <YAxis
                        label={{
                          value: predictionType === 'waste' ? 'Waste (million tonnes)' : 'Pollution (million tonnes)',
                          angle: -90,
                          position: 'insideLeft',
                          offset: -15
                        }}
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => value.toLocaleString()}
                        width={80}
                      />
                      <Tooltip
                        formatter={(value) => value.toLocaleString()}
                        labelFormatter={(value) => `Production: ${value.toLocaleString()}`}
                      />
                      <Legend verticalAlign="top" height={36} />
                      <Line
                        type="monotone"
                        dataKey="original"
                        stroke="#8884d8"
                        name="Historical Data"
                        dot
                        strokeWidth={2}
                      />
                      {graphData.find(point => point.userInput !== undefined) && (
                        <Line
                          type="monotone"
                          dataKey="userInput"
                          stroke="#ff7300"
                          name="Your Prediction"
                          dot={{
                            r: 6,
                            cx: graphData.find(point => point.userInput !== undefined).production,
                            cy: graphData.find(point => point.userInput !== undefined).userInput
                          }}
                          strokeWidth={2}
                        />
                      )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              {impactAnalysis && (
                <div className="environmental-impact">
                  <h2>Environmental Impact Analysis</h2>
                  <p>{impactAnalysis}</p>
                </div>
              )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default App