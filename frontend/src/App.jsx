import { useState } from 'react'
import './App.css'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import ReactMarkdown from 'react-markdown'

function App() {
  // State variables
  const [predictionType, setPredictionType] = useState('waste')
  const [modelType, setModelType] = useState('model1')
  const [productionAmount, setProductionAmount] = useState('')
  const [result, setResult] = useState(null)
  const [graphData, setGraphData] = useState(null)
  const [error, setError] = useState(null)
  const [environmentalImpacts, setEnvironmentalImpacts] = useState(null)
  const [solutions, setSolutions] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  // Handle submit button click
  const handleSubmit = async () => {
    try {
      if (!productionAmount) {
        setError('Please enter a production amount')
        return
      }
      setIsLoading(true)
      setError(null)
      
      // Make prediction API call
      const predictionResponse = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          productionAmount: parseFloat(productionAmount) * 1000000,
          predictionType: predictionType,
          modelType: modelType === 'model1' ? 'linear' : modelType === 'model2' ? 'polynomial' : 'randomforest'
        })
      })

      if (!predictionResponse.ok) {
        throw new Error(`HTTP error! status: ${predictionResponse.status}`)
      }

      const predictionData = await predictionResponse.json()
      setResult(predictionData.prediction.toFixed(2))
      setGraphData(predictionData.graphData)

      // Make environmental impacts API call
      const impactsResponse = await fetch('http://localhost:8000/api/environmental-impacts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          productionAmount: parseFloat(productionAmount) * 1000000,
          predictionType: predictionType,
          modelType: modelType === 'model1' ? 'linear' : modelType === 'model2' ? 'polynomial' : 'randomforest'
        })
      })

      if (!impactsResponse.ok) {
        throw new Error(`HTTP error! status: ${impactsResponse.status}`)
      }

      const impactsData = await impactsResponse.json()
      setEnvironmentalImpacts(impactsData.environmentalImpacts)

      // Make model info API call
      const modelInfoResponse = await fetch('http://localhost:8000/api/model-info', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          productionAmount: parseFloat(productionAmount) * 1000000,
          predictionType: predictionType,
          modelType: modelType === 'model1' ? 'linear' : modelType === 'model2' ? 'polynomial' : 'randomforest'
        })
      })

      if (!modelInfoResponse.ok) {
        throw new Error(`HTTP error! status: ${modelInfoResponse.status}`)
      }

      const modelInfoData = await modelInfoResponse.json()
      setModelInfo(modelInfoData.modelInfo)

      // Make solutions API call
      const solutionsResponse = await fetch('http://localhost:8000/api/solutions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          productionAmount: parseFloat(productionAmount) * 1000000,
          predictionType: predictionType,
          modelType: modelType === 'model1' ? 'linear' : modelType === 'model2' ? 'polynomial' : 'randomforest'
        })
      })

      if (!solutionsResponse.ok) {
        throw new Error(`HTTP error! status: ${solutionsResponse.status}`)
      }

      const solutionsData = await solutionsResponse.json()
      setSolutions(solutionsData.solutions)
      setError(null)
    } catch (error) {
      console.error('Error:', error)
      setError('Error occurred while making prediction')
    } finally {
      setIsLoading(false)
    }
  }

  // Render the main component
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
              // Update prediction type
              setPredictionType('waste')
              // Reset other state variables
              setProductionAmount('')
              setResult(null)
              setGraphData(null)
              setEnvironmentalImpacts(null)
              setSolutions(null)
            setModelInfo(null)
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
              // Update prediction type
              setPredictionType('pollution')
              // Reset other state variables
              setProductionAmount('')
              setResult(null)
              setGraphData(null)
              setEnvironmentalImpacts(null)
              setSolutions(null)
              setModelInfo(null)
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
              // Update model type
              setModelType('model1')
              // Reset other state variables
              setProductionAmount('')
              setResult(null)
              setGraphData(null)
              setEnvironmentalImpacts(null)
              setSolutions(null)
              setModelInfo(null)
              setError(null)
          }}
          >
            Linear Model
          </button>
          <button 
            style={{
              backgroundColor: modelType === 'model2' ? '#4CAF50' : '#ffffff',
              color: modelType === 'model2' ? '#ffffff' : '#000000',
              border: '1px solid #1a1a1a'
            }}
            onClick={() => {
              // Update model type
              setModelType('model2')
              // Reset other state variables
              setProductionAmount('')
              setResult(null)
              setGraphData(null)
              setEnvironmentalImpacts(null)
              setSolutions(null)
              setModelInfo(null)
              setError(null)
          }}
          >
            Polynomial Model
          </button>
          <button 
            style={{
              backgroundColor: modelType === 'model3' ? '#4CAF50' : '#ffffff',
              color: modelType === 'model3' ? '#ffffff' : '#000000',
              border: '1px solid #1a1a1a'
            }}
            onClick={() => {
              // Update model type
              setModelType('model3')
              // Reset other state variables
              setProductionAmount('')
              setResult(null)
              setGraphData(null)
              setEnvironmentalImpacts(null)
              setSolutions(null)
              setModelInfo(null)
            setError(null)
          }}
          >
            Random Forest Model
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
        {isLoading ? (
          <div className="result-display">
            <p>Loading prediction...</p>
          </div>
        ) : error ? (
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

            {/* Graph data visualization */}
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

                {/* Environmental impact analysis */}
                <div className="environmental-impact">
                  <h2>Environmental Impact Analysis</h2>
                  <div className="impact-sections">
                    {environmentalImpacts && (
                      <div className="impact-section impacts">
                        <h3>Environmental Impacts</h3>
                        <div className="impact-content">
                          <ReactMarkdown>{environmentalImpacts}</ReactMarkdown>
                        </div>
                      </div>
                    )}

                    {solutions && (
                      <div className="impact-section solutions">
                        <h3>Solutions</h3>
                        <div className="impact-content">
                          <ReactMarkdown>{solutions}</ReactMarkdown>
                        </div>
                      </div>
                    )}

                    {modelInfo && (
                      <div className="impact-section model-info">
                        <h3>Model Information</h3>
                        <div className="impact-content">
                          <ReactMarkdown>{modelInfo}</ReactMarkdown>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}
// Export the main component
export default App