const mockData = {
  dashboard: {
    totalProducts: 1250,
    lowStockItems: 23,
    outOfStockItems: 5,
    totalValue: 1250000,
    accuracy: 94.2,
    savings: 125000,
    recentAlerts: [
      { id: 1, type: 'low_stock', message: 'Organic Bananas running low', severity: 'medium' },
      { id: 2, type: 'out_of_stock', message: 'Whole Milk 2% out of stock', severity: 'high' },
      { id: 3, type: 'prediction', message: 'High demand predicted for bread', severity: 'low' }
    ],
    quickActions: [
      { id: 1, title: 'Review Suggestions', icon: 'suggestions', count: 15 },
      { id: 2, title: 'Check Alerts', icon: 'alerts', count: 3 },
      { id: 3, title: 'View Reports', icon: 'reports', count: 8 }
    ]
  },
  stockSuggestions: [
    {
      id: 1,
      productName: "Organic Bananas",
      currentStock: 45,
      predictedDemand: 120,
      confidence: 0.92,
      suggestedOrder: 75,
      urgency: "high",
      category: "Produce",
      supplier: "Fresh Farms Inc.",
      lastOrderDate: "2024-01-15",
      leadTime: 2
    },
    {
      id: 2,
      productName: "Whole Milk 2%",
      currentStock: 12,
      predictedDemand: 85,
      confidence: 0.88,
      suggestedOrder: 73,
      urgency: "medium",
      category: "Dairy",
      supplier: "Dairy Co.",
      lastOrderDate: "2024-01-18",
      leadTime: 1
    },
    {
      id: 3,
      productName: "Whole Grain Bread",
      currentStock: 8,
      predictedDemand: 95,
      confidence: 0.85,
      suggestedOrder: 87,
      urgency: "high",
      category: "Bakery",
      supplier: "Bakery Fresh",
      lastOrderDate: "2024-01-20",
      leadTime: 1
    }
  ],
  historicalData: {
    predictions: [
      { date: '2024-01-01', predicted: 100, actual: 98, accuracy: 98 },
      { date: '2024-01-02', predicted: 105, actual: 102, accuracy: 97 },
      { date: '2024-01-03', predicted: 95, actual: 97, accuracy: 98 },
      { date: '2024-01-04', predicted: 110, actual: 108, accuracy: 98 },
      { date: '2024-01-05', predicted: 115, actual: 112, accuracy: 97 }
    ],
    accuracyTrends: [
      { month: 'Jan', accuracy: 94.2 },
      { month: 'Feb', accuracy: 95.1 },
      { month: 'Mar', accuracy: 93.8 },
      { month: 'Apr', accuracy: 96.2 },
      { month: 'May', accuracy: 95.7 }
    ],
    productAccuracy: [
      { product: 'Bananas', accuracy: 96.5 },
      { product: 'Milk', accuracy: 94.2 },
      { product: 'Bread', accuracy: 92.8 },
      { product: 'Eggs', accuracy: 97.1 },
      { product: 'Cheese', accuracy: 93.5 }
    ]
  },
  stockDetails: [
    {
      id: 1,
      productName: "Organic Bananas",
      category: "Produce",
      currentStock: 45,
      reorderPoint: 50,
      maxStock: 200,
      unitCost: 0.89,
      totalValue: 40.05,
      supplier: {
        name: "Fresh Farms Inc.",
        contact: "John Smith",
        phone: "(555) 123-4567",
        email: "john@freshfarms.com"
      },
      demandTrend: [
        { day: 'Mon', demand: 25 },
        { day: 'Tue', demand: 30 },
        { day: 'Wed', demand: 28 },
        { day: 'Thu', demand: 35 },
        { day: 'Fri', demand: 40 },
        { day: 'Sat', demand: 45 },
        { day: 'Sun', demand: 20 }
      ]
    }
  ],
  modelFactors: {
    temperature: {
      current: 72,
      forecast: [68, 70, 72, 75, 73, 71, 69],
      correlation: 0.85
    },
    demandForecast: {
      current: 120,
      trend: 'increasing',
      factors: ['seasonal', 'promotional', 'weather']
    },
    specialEvents: [
      { name: 'Holiday Sale', date: '2024-02-15', impact: 'high' },
      { name: 'Local Festival', date: '2024-02-20', impact: 'medium' }
    ],
    weatherData: {
      current: 'Sunny',
      forecast: 'Clear skies',
      humidity: 65,
      windSpeed: 8
    }
  },
  modelInsights: {
    featureImportance: [
      { feature: 'Historical Sales', importance: 0.35 },
      { feature: 'Weather', importance: 0.25 },
      { feature: 'Seasonality', importance: 0.20 },
      { feature: 'Promotions', importance: 0.15 },
      { feature: 'Events', importance: 0.05 }
    ],
    confidenceDistribution: [
      { range: '90-100%', count: 45 },
      { range: '80-89%', count: 32 },
      { range: '70-79%', count: 18 },
      { range: '60-69%', count: 8 },
      { range: '50-59%', count: 2 }
    ],
    recentDecisions: [
      { date: '2024-01-20', decision: 'Increase order for bananas', confidence: 92, factors: ['weather', 'historical'] },
      { date: '2024-01-19', decision: 'Reduce milk order', confidence: 88, factors: ['demand trend', 'seasonality'] },
      { date: '2024-01-18', decision: 'Maintain bread levels', confidence: 95, factors: ['promotions', 'events'] }
    ]
  },
  orders: [
    {
      id: 1,
      productName: "Organic Bananas",
      quantity: 75,
      supplier: "Fresh Farms Inc.",
      status: "pending",
      priority: "high",
      orderDate: "2024-01-20",
      expectedDelivery: "2024-01-22",
      totalCost: 66.75
    },
    {
      id: 2,
      productName: "Whole Milk 2%",
      quantity: 73,
      supplier: "Dairy Co.",
      status: "confirmed",
      priority: "medium",
      orderDate: "2024-01-19",
      expectedDelivery: "2024-01-21",
      totalCost: 87.60
    }
  ]
};

module.exports = mockData; 