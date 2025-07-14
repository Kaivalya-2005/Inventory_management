import React, { useState } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';
import { 
  Download,
  Calendar,
  TrendingUp,
  TrendingDown,
  BarChart3
} from 'lucide-react';
import { format, subDays } from 'date-fns';

const HistoricalData = () => {
  const [selectedPeriod, setSelectedPeriod] = useState('30');
  const [selectedProduct, setSelectedProduct] = useState('all');

  // Mock historical data
  const generateHistoricalData = (days) => {
    const data = [];
    for (let i = days - 1; i >= 0; i--) {
      const date = subDays(new Date(), i);
      const baseDemand = Math.floor(Math.random() * 100) + 50;
      const temperature = Math.floor(Math.random() * 20) + 15;
      
      data.push({
        date: format(date, 'MMM dd'),
        predicted: baseDemand + Math.floor(Math.random() * 20),
        actual: baseDemand + Math.floor(Math.random() * 30),
        temperature: temperature,
        error: Math.floor(Math.random() * 15),
        accuracy: Math.floor(Math.random() * 20) + 80
      });
    }
    return data;
  };

  const historicalData = generateHistoricalData(parseInt(selectedPeriod));

  const products = [
    { id: 'all', name: 'All Products' },
    { id: 'bananas', name: 'Organic Bananas' },
    { id: 'milk', name: 'Fresh Milk' },
    { id: 'bread', name: 'Bread' },
    { id: 'water', name: 'Bottled Water' }
  ];

  const accuracyData = [
    { product: 'Organic Bananas', accuracy: 94, trend: 'up' },
    { product: 'Fresh Milk', accuracy: 88, trend: 'up' },
    { product: 'Bread', accuracy: 92, trend: 'down' },
    { product: 'Bottled Water', accuracy: 96, trend: 'up' },
    { product: 'Ice Cream', accuracy: 85, trend: 'up' }
  ];

  const errorAnalysis = [
    { type: 'Weather Changes', percentage: 35, impact: 'High' },
    { type: 'Unexpected Events', percentage: 25, impact: 'Medium' },
    { type: 'Seasonal Variations', percentage: 20, impact: 'Low' },
    { type: 'Data Quality', percentage: 15, impact: 'Medium' },
    { type: 'Model Limitations', percentage: 5, impact: 'Low' }
  ];

  const handleExport = () => {
    // Mock export functionality
    const csvContent = "data:text/csv;charset=utf-8," + 
      "Date,Predicted,Actual,Temperature,Error,Accuracy\n" +
      historicalData.map(row => 
        `${row.date},${row.predicted},${row.actual},${row.temperature},${row.error},${row.accuracy}`
      ).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `inventory_data_${selectedPeriod}days.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getAverageAccuracy = () => {
    return Math.round(historicalData.reduce((sum, item) => sum + item.accuracy, 0) / historicalData.length);
  };

  const getAverageError = () => {
    return Math.round(historicalData.reduce((sum, item) => sum + item.error, 0) / historicalData.length);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Historical Data</h1>
          <p className="text-gray-600">Analyze past predictions and model performance</p>
        </div>
        <button
          onClick={handleExport}
          className="btn-primary flex items-center"
        >
          <Download className="h-4 w-4 mr-2" />
          Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="md:w-48">
            <label className="block text-sm font-medium text-gray-700 mb-1">Time Period</label>
            <select
              className="input-field"
              value={selectedPeriod}
              onChange={(e) => setSelectedPeriod(e.target.value)}
            >
              <option value="7">Last 7 days</option>
              <option value="30">Last 30 days</option>
              <option value="90">Last 90 days</option>
              <option value="365">Last year</option>
            </select>
          </div>
          <div className="md:w-48">
            <label className="block text-sm font-medium text-gray-700 mb-1">Product</label>
            <select
              className="input-field"
              value={selectedProduct}
              onChange={(e) => setSelectedProduct(e.target.value)}
            >
              {products.map(product => (
                <option key={product.id} value={product.id}>
                  {product.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Summary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card text-center">
          <div className="text-2xl font-bold text-primary-600">{getAverageAccuracy()}%</div>
          <div className="text-sm text-gray-600">Average Accuracy</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-warning-600">{getAverageError()}%</div>
          <div className="text-sm text-gray-600">Average Error</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-success-600">
            {historicalData.length}
          </div>
          <div className="text-sm text-gray-600">Data Points</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-gray-600">
            {Math.round(historicalData.reduce((sum, item) => sum + item.temperature, 0) / historicalData.length)}°C
          </div>
          <div className="text-sm text-gray-600">Avg Temperature</div>
        </div>
      </div>

      {/* Main Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Prediction vs Actual */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Predicted vs Actual Demand</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={2} name="Predicted" />
              <Line type="monotone" dataKey="actual" stroke="#22c55e" strokeWidth={2} name="Actual" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Accuracy Trend */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Accuracy Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={historicalData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[70, 100]} />
              <Tooltip />
              <Line type="monotone" dataKey="accuracy" stroke="#f59e0b" strokeWidth={2} name="Accuracy %" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Error Analysis and Product Accuracy */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Error Analysis */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Error Analysis</h3>
          <div className="space-y-4">
            {errorAnalysis.map((error, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-900">{error.type}</span>
                    <span className="text-sm text-gray-500">{error.percentage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-primary-600 h-2 rounded-full" 
                      style={{ width: `${error.percentage}%` }}
                    ></div>
                  </div>
                  <div className="flex items-center mt-1">
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      error.impact === 'High' ? 'bg-danger-100 text-danger-800' :
                      error.impact === 'Medium' ? 'bg-warning-100 text-warning-800' :
                      'bg-success-100 text-success-800'
                    }`}>
                      {error.impact} Impact
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Product Accuracy */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Product Accuracy</h3>
          <div className="space-y-3">
            {accuracyData.map((product, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center">
                  <BarChart3 className="h-4 w-4 text-gray-400 mr-3" />
                  <span className="text-sm font-medium text-gray-900">{product.product}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-bold text-gray-900">{product.accuracy}%</span>
                  {product.trend === 'up' ? (
                    <TrendingUp className="h-4 w-4 text-success-600" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-danger-600" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Temperature Correlation */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Temperature vs Demand Correlation</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={historicalData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Bar yAxisId="left" dataKey="actual" fill="#3b82f6" name="Demand" />
            <Line yAxisId="right" type="monotone" dataKey="temperature" stroke="#f59e0b" strokeWidth={2} name="Temperature (°C)" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Data Table */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Data</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="table-header">Date</th>
                <th className="table-header">Predicted</th>
                <th className="table-header">Actual</th>
                <th className="table-header">Temperature</th>
                <th className="table-header">Error</th>
                <th className="table-header">Accuracy</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {historicalData.slice(-10).map((row, index) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="table-cell">{row.date}</td>
                  <td className="table-cell">{row.predicted}</td>
                  <td className="table-cell">{row.actual}</td>
                  <td className="table-cell">{row.temperature}°C</td>
                  <td className="table-cell">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      row.error < 5 ? 'bg-success-100 text-success-800' :
                      row.error < 10 ? 'bg-warning-100 text-warning-800' :
                      'bg-danger-100 text-danger-800'
                    }`}>
                      {row.error}%
                    </span>
                  </td>
                  <td className="table-cell">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      row.accuracy >= 90 ? 'bg-success-100 text-success-800' :
                      row.accuracy >= 80 ? 'bg-warning-100 text-warning-800' :
                      'bg-danger-100 text-danger-800'
                    }`}>
                      {row.accuracy}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default HistoricalData; 