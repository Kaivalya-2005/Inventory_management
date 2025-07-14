import React, { useState } from 'react';
import { 
  PieChart, 
  Pie, 
  Cell, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer
} from 'recharts';
import { 
  Brain, 
  TrendingUp, 
  BarChart3,
  Lightbulb,
  Target,
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap
} from 'lucide-react';

const ModelInsights = () => {


  // Mock feature importance data
  const featureImportance = [
    { feature: 'Temperature', importance: 35, impact: 'positive', description: 'Higher temperatures increase demand for cold beverages and frozen items' },
    { feature: 'Historical Sales', importance: 28, impact: 'positive', description: 'Past sales patterns are strong predictors of future demand' },
    { feature: 'Day of Week', importance: 18, impact: 'neutral', description: 'Weekend vs weekday patterns affect shopping behavior' },
    { feature: 'Special Events', importance: 12, impact: 'positive', description: 'Festivals, holidays, and promotions significantly impact demand' },
    { feature: 'Seasonal Trends', importance: 7, impact: 'neutral', description: 'Long-term seasonal patterns influence product preferences' }
  ];

  const confidenceLevels = [
    { level: 'High (90-100%)', count: 45, color: '#22c55e' },
    { level: 'Medium (70-89%)', count: 32, color: '#f59e0b' },
    { level: 'Low (50-69%)', count: 18, color: '#ef4444' },
    { level: 'Very Low (<50%)', count: 5, color: '#dc2626' }
  ];

  const decisionFactors = [
    {
      product: 'Organic Bananas',
      factors: [
        { name: 'Temperature', weight: 40, reason: 'High temperature increases fruit consumption' },
        { name: 'Weekend Effect', weight: 30, reason: 'Higher weekend sales for fresh produce' },
        { name: 'Previous Sales', weight: 20, reason: 'Consistent demand pattern' },
        { name: 'Seasonal Trend', weight: 10, reason: 'Summer fruit preference' }
      ],
      confidence: 95,
      recommendation: 'Increase stock by 25% due to hot weather forecast'
    },
    {
      product: 'Fresh Milk',
      factors: [
        { name: 'Daily Essential', weight: 50, reason: 'Consistent daily consumption pattern' },
        { name: 'Expiration Date', weight: 25, reason: 'Short shelf life affects ordering' },
        { name: 'Temperature', weight: 15, reason: 'Slight increase in hot weather' },
        { name: 'Historical Data', weight: 10, reason: 'Stable demand over time' }
      ],
      confidence: 88,
      recommendation: 'Maintain current stock levels with slight increase'
    }
  ];

  const modelPerformance = [
    { metric: 'Overall Accuracy', value: 87, target: 90, status: 'warning' },
    { metric: 'Temperature Correlation', value: 92, target: 85, status: 'success' },
    { metric: 'Event Prediction', value: 78, target: 80, status: 'warning' },
    { metric: 'Seasonal Accuracy', value: 94, target: 90, status: 'success' },
    { metric: 'Real-time Updates', value: 96, target: 95, status: 'success' }
  ];

  const recentDecisions = [
    {
      id: 1,
      product: 'Ice Cream',
      decision: 'Increase stock by 40%',
      confidence: 92,
      factors: ['High temperature (32°C)', 'Weekend forecast', 'Previous hot day sales'],
      timestamp: '2 hours ago',
      status: 'implemented'
    },
    {
      id: 2,
      product: 'Bottled Water',
      decision: 'Increase stock by 60%',
      confidence: 89,
      factors: ['Heat wave warning', 'Event weekend', 'Historical demand spike'],
      timestamp: '4 hours ago',
      status: 'pending'
    },
    {
      id: 3,
      product: 'Bread',
      decision: 'Maintain current levels',
      confidence: 85,
      factors: ['Stable demand pattern', 'No special events', 'Normal temperature'],
      timestamp: '6 hours ago',
      status: 'implemented'
    }
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'success': return 'text-success-600 bg-success-50';
      case 'warning': return 'text-warning-600 bg-warning-50';
      case 'danger': return 'text-danger-600 bg-danger-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success': return CheckCircle;
      case 'warning': return AlertTriangle;
      case 'danger': return AlertTriangle;
      default: return Clock;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Insights</h1>
        <p className="text-gray-600">Understanding how the AI model makes decisions and predictions</p>
      </div>

      {/* Model Overview */}
      <div className="card">
        <div className="flex items-center space-x-4 mb-6">
          <div className="p-3 bg-primary-100 rounded-lg">
            <Brain className="h-8 w-8 text-primary-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">AI Model Overview</h3>
            <p className="text-gray-600">Machine learning model trained on historical data with 87% accuracy</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-primary-600">87%</div>
            <div className="text-sm text-gray-600">Overall Accuracy</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-success-600">1.2M</div>
            <div className="text-sm text-gray-600">Data Points</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-warning-600">5</div>
            <div className="text-sm text-gray-600">Key Features</div>
          </div>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={featureImportance} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} />
              <YAxis dataKey="feature" type="category" width={100} />
              <Tooltip />
              <Bar dataKey="importance" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Confidence Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={confidenceLevels}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="count"
                label={({ level, count }) => `${level}: ${count}`}
              >
                {confidenceLevels.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Decision Factors for Products */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Decision Factors by Product</h3>
        <div className="space-y-6">
          {decisionFactors.map((item, index) => (
            <div key={index} className="border border-gray-200 rounded-lg p-4">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h4 className="font-medium text-gray-900">{item.product}</h4>
                  <p className="text-sm text-gray-600">{item.recommendation}</p>
                </div>
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  item.confidence >= 90 ? 'bg-success-100 text-success-800' :
                  item.confidence >= 80 ? 'bg-warning-100 text-warning-800' :
                  'bg-danger-100 text-danger-800'
                }`}>
                  {item.confidence}% Confidence
                </span>
              </div>
              
              <div className="space-y-3">
                {item.factors.map((factor, factorIndex) => (
                  <div key={factorIndex} className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-gray-900">{factor.name}</span>
                        <span className="text-sm text-gray-500">{factor.weight}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-primary-600 h-2 rounded-full" 
                          style={{ width: `${factor.weight}%` }}
                        ></div>
                      </div>
                      <p className="text-xs text-gray-600 mt-1">{factor.reason}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model Performance Metrics */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Performance Metrics</h3>
        <div className="space-y-4">
          {modelPerformance.map((metric, index) => {
            const Icon = getStatusIcon(metric.status);
            return (
              <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <Icon className={`h-5 w-5 ${getStatusColor(metric.status).split(' ')[0]}`} />
                  <span className="font-medium text-gray-900">{metric.metric}</span>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <div className="text-lg font-bold text-gray-900">{metric.value}%</div>
                    <div className="text-xs text-gray-500">Target: {metric.target}%</div>
                  </div>
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(metric.status)}`}>
                    {metric.status.charAt(0).toUpperCase() + metric.status.slice(1)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Recent Decisions */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent AI Decisions</h3>
        <div className="space-y-4">
          {recentDecisions.map((decision) => (
            <div key={decision.id} className="border border-gray-200 rounded-lg p-4">
              <div className="flex justify-between items-start mb-3">
                <div>
                  <h4 className="font-medium text-gray-900">{decision.product}</h4>
                  <p className="text-sm text-gray-600">{decision.decision}</p>
                </div>
                <div className="text-right">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    decision.confidence >= 90 ? 'bg-success-100 text-success-800' :
                    decision.confidence >= 80 ? 'bg-warning-100 text-warning-800' :
                    'bg-danger-100 text-danger-800'
                  }`}>
                    {decision.confidence}% Confidence
                  </span>
                  <div className="text-xs text-gray-500 mt-1">{decision.timestamp}</div>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="text-sm font-medium text-gray-900">Key Factors:</div>
                <div className="flex flex-wrap gap-2">
                  {decision.factors.map((factor, index) => (
                    <span key={index} className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-primary-100 text-primary-800">
                      <Zap className="h-3 w-3 mr-1" />
                      {factor}
                    </span>
                  ))}
                </div>
              </div>
              
              <div className="mt-3 flex items-center justify-between">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                  decision.status === 'implemented' ? 'bg-success-100 text-success-800' : 'bg-warning-100 text-warning-800'
                }`}>
                  {decision.status.charAt(0).toUpperCase() + decision.status.slice(1)}
                </span>
                <button className="text-sm text-primary-600 hover:text-primary-700">
                  View Details
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Model Transparency */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Transparency</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="flex items-start space-x-3">
              <Lightbulb className="h-5 w-5 text-primary-600 mt-0.5" />
              <div>
                <h4 className="font-medium text-gray-900">How Decisions Are Made</h4>
                <p className="text-sm text-gray-600">The model analyzes multiple factors including temperature, historical sales, events, and seasonal patterns to predict demand.</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <Target className="h-5 w-5 text-success-600 mt-0.5" />
              <div>
                <h4 className="font-medium text-gray-900">Confidence Levels</h4>
                                 <p className="text-sm text-gray-600">High confidence predictions (&gt;90%) are based on strong patterns, while lower confidence indicates uncertainty.</p>
              </div>
            </div>
          </div>
          <div className="space-y-4">
            <div className="flex items-start space-x-3">
              <AlertTriangle className="h-5 w-5 text-warning-600 mt-0.5" />
              <div>
                <h4 className="font-medium text-gray-900">Limitations</h4>
                <p className="text-sm text-gray-600">The model may struggle with unprecedented events or rapid market changes. Human oversight is recommended.</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <Clock className="h-5 w-5 text-gray-600 mt-0.5" />
              <div>
                <h4 className="font-medium text-gray-900">Continuous Learning</h4>
                <p className="text-sm text-gray-600">The model improves over time as it learns from new data and prediction outcomes.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelInsights; 