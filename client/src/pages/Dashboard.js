import React from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  Package, 
  TrendingUp, 
  AlertTriangle, 
  Thermometer,
  ShoppingCart,
  Clock
} from 'lucide-react';

const Dashboard = () => {
  // Mock data for charts
  const demandData = [
    { day: 'Mon', predicted: 120, actual: 115 },
    { day: 'Tue', predicted: 135, actual: 140 },
    { day: 'Wed', predicted: 150, actual: 145 },
    { day: 'Thu', predicted: 165, actual: 170 },
    { day: 'Fri', predicted: 180, actual: 175 },
    { day: 'Sat', predicted: 200, actual: 195 },
    { day: 'Sun', predicted: 160, actual: 165 },
  ];

  const stockLevels = [
    { name: 'In Stock', value: 65, color: '#22c55e' },
    { name: 'Low Stock', value: 20, color: '#f59e0b' },
    { name: 'Out of Stock', value: 15, color: '#ef4444' },
  ];

  const temperatureData = [
    { time: '6AM', temp: 18 },
    { time: '9AM', temp: 22 },
    { time: '12PM', temp: 28 },
    { time: '3PM', temp: 30 },
    { time: '6PM', temp: 25 },
    { time: '9PM', temp: 20 },
  ];

  const metrics = [
    {
      title: 'Total Stock',
      value: '12,450',
      change: '+5.2%',
      changeType: 'positive',
      icon: Package,
      color: 'bg-blue-500',
    },
    {
      title: 'Predicted Orders',
      value: '1,234',
      change: '+12.1%',
      changeType: 'positive',
      icon: TrendingUp,
      color: 'bg-green-500',
    },
    {
      title: 'Out of Stock',
      value: '23',
      change: '-8.3%',
      changeType: 'negative',
      icon: AlertTriangle,
      color: 'bg-red-500',
    },
    {
      title: 'Current Temp',
      value: '28°C',
      change: '+2.1°C',
      changeType: 'neutral',
      icon: Thermometer,
      color: 'bg-orange-500',
    },
  ];

  const recentAlerts = [
    {
      id: 1,
      type: 'warning',
      message: 'Product "Organic Bananas" is running low on stock',
      time: '2 minutes ago',
    },
    {
      id: 2,
      type: 'info',
      message: 'Weather forecast predicts high demand for cold beverages',
      time: '15 minutes ago',
    },
    {
      id: 3,
      type: 'success',
      message: 'Auto-order placed for "Fresh Milk" - 500 units',
      time: '1 hour ago',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600">Overview of your inventory and predictions</p>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric) => (
          <div key={metric.title} className="card">
            <div className="flex items-center">
              <div className={`p-2 rounded-lg ${metric.color}`}>
                <metric.icon className="h-6 w-6 text-white" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">{metric.title}</p>
                <p className="text-2xl font-bold text-gray-900">{metric.value}</p>
              </div>
            </div>
            <div className="mt-4">
              <span className={`text-sm font-medium ${
                metric.changeType === 'positive' ? 'text-green-600' : 
                metric.changeType === 'negative' ? 'text-red-600' : 'text-gray-600'
              }`}>
                {metric.change}
              </span>
              <span className="text-sm text-gray-500 ml-1">from last week</span>
            </div>
          </div>
        ))}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Demand Prediction Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Demand Prediction (7 Days)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={demandData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="predicted" fill="#3b82f6" name="Predicted" />
              <Bar dataKey="actual" fill="#22c55e" name="Actual" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Stock Levels Pie Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Current Stock Levels</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={stockLevels}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}%`}
              >
                {stockLevels.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Temperature and Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Temperature Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Temperature Trend</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={temperatureData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="temp" stroke="#f59e0b" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Alerts */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Alerts</h3>
          <div className="space-y-3">
            {recentAlerts.map((alert) => (
              <div key={alert.id} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                <div className={`w-2 h-2 rounded-full mt-2 ${
                  alert.type === 'warning' ? 'bg-warning-500' :
                  alert.type === 'info' ? 'bg-primary-500' : 'bg-success-500'
                }`} />
                <div className="flex-1">
                  <p className="text-sm text-gray-900">{alert.message}</p>
                  <p className="text-xs text-gray-500">{alert.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <button className="flex flex-col items-center p-4 bg-primary-50 rounded-lg hover:bg-primary-100 transition-colors">
            <ShoppingCart className="h-8 w-8 text-primary-600 mb-2" />
            <span className="text-sm font-medium text-primary-900">View Orders</span>
          </button>
          <button className="flex flex-col items-center p-4 bg-success-50 rounded-lg hover:bg-success-100 transition-colors">
            <Package className="h-8 w-8 text-success-600 mb-2" />
            <span className="text-sm font-medium text-success-900">Stock Details</span>
          </button>
          <button className="flex flex-col items-center p-4 bg-warning-50 rounded-lg hover:bg-warning-100 transition-colors">
            <TrendingUp className="h-8 w-8 text-warning-600 mb-2" />
            <span className="text-sm font-medium text-warning-900">Predictions</span>
          </button>
          <button className="flex flex-col items-center p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
            <Clock className="h-8 w-8 text-gray-600 mb-2" />
            <span className="text-sm font-medium text-gray-900">History</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 