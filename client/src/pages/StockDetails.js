import React, { useState } from 'react';
import { 
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { 
  Search, 
  Package, 
  TrendingUp, 
  AlertTriangle, 
  Clock, 
  DollarSign,
  Phone,
  Mail,
  BarChart3
} from 'lucide-react';

const StockDetails = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  // Mock stock data
  const stockData = [
    {
      id: 1,
      sku: 'BAN-001',
      productName: 'Organic Bananas',
      category: 'Fruits',
      currentStock: 50,
      safetyStock: 25,
      reorderPoint: 30,
      maxStock: 200,
      unitPrice: 0.89,
      supplier: {
        name: 'Fresh Farms Inc.',
        phone: '+1 (555) 123-4567',
        email: 'orders@freshfarms.com',
        address: '123 Farm Road, California'
      },
      demandTrend: [
        { day: 'Mon', demand: 45 },
        { day: 'Tue', demand: 52 },
        { day: 'Wed', demand: 48 },
        { day: 'Thu', demand: 61 },
        { day: 'Fri', demand: 67 },
        { day: 'Sat', demand: 78 },
        { day: 'Sun', demand: 55 }
      ],
      lastOrderDate: '2024-01-15',
      nextOrderDate: '2024-01-22',
      status: 'low'
    },
    {
      id: 2,
      sku: 'MIL-002',
      productName: 'Fresh Milk',
      category: 'Dairy',
      currentStock: 75,
      safetyStock: 50,
      reorderPoint: 60,
      maxStock: 150,
      unitPrice: 3.99,
      supplier: {
        name: 'Dairy Co.',
        phone: '+1 (555) 234-5678',
        email: 'supply@dairyco.com',
        address: '456 Dairy Lane, Wisconsin'
      },
      demandTrend: [
        { day: 'Mon', demand: 35 },
        { day: 'Tue', demand: 38 },
        { day: 'Wed', demand: 42 },
        { day: 'Thu', demand: 40 },
        { day: 'Fri', demand: 45 },
        { day: 'Sat', demand: 50 },
        { day: 'Sun', demand: 38 }
      ],
      lastOrderDate: '2024-01-18',
      nextOrderDate: '2024-01-25',
      status: 'normal'
    },
    {
      id: 3,
      sku: 'ICE-003',
      productName: 'Vanilla Ice Cream',
      category: 'Frozen',
      currentStock: 25,
      safetyStock: 30,
      reorderPoint: 35,
      maxStock: 100,
      unitPrice: 4.99,
      supplier: {
        name: 'Frozen Delights',
        phone: '+1 (555) 345-6789',
        email: 'orders@frozendelights.com',
        address: '789 Cold Street, Alaska'
      },
      demandTrend: [
        { day: 'Mon', demand: 15 },
        { day: 'Tue', demand: 18 },
        { day: 'Wed', demand: 22 },
        { day: 'Thu', demand: 25 },
        { day: 'Fri', demand: 30 },
        { day: 'Sat', demand: 35 },
        { day: 'Sun', demand: 28 }
      ],
      lastOrderDate: '2024-01-20',
      nextOrderDate: '2024-01-27',
      status: 'critical'
    }
  ];

  const categories = ['all', 'Fruits', 'Dairy', 'Frozen', 'Bakery', 'Beverages'];

  const filteredStock = stockData.filter(item => {
    const matchesSearch = item.productName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         item.sku.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || item.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });



  const getStatusColor = (status) => {
    switch (status) {
      case 'critical': return 'text-danger-600 bg-danger-50';
      case 'low': return 'text-warning-600 bg-warning-50';
      case 'normal': return 'text-success-600 bg-success-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getStockPercentage = (current, max) => {
    return Math.round((current / max) * 100);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Stock Details</h1>
        <p className="text-gray-600">Detailed view of individual product stock information</p>
      </div>

      {/* Search and Filters */}
      <div className="card">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
              <input
                type="text"
                placeholder="Search by product name or SKU..."
                className="input-field pl-10"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>
          <div className="md:w-48">
            <select
              className="input-field"
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
            >
              {categories.map(category => (
                <option key={category} value={category}>
                  {category === 'all' ? 'All Categories' : category}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Stock Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {filteredStock.map((item) => (
          <div key={item.id} className="card">
            {/* Header */}
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">{item.productName}</h3>
                <p className="text-sm text-gray-500">SKU: {item.sku}</p>
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 mt-1">
                  {item.category}
                </span>
              </div>
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(item.status)}`}>
                {item.status.charAt(0).toUpperCase() + item.status.slice(1)} Stock
              </span>
            </div>

            {/* Stock Levels */}
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600">Current Stock</span>
                  <span className="font-medium">{item.currentStock} units</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${
                      item.status === 'critical' ? 'bg-danger-500' :
                      item.status === 'low' ? 'bg-warning-500' : 'bg-success-500'
                    }`}
                    style={{ width: `${getStockPercentage(item.currentStock, item.maxStock)}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0</span>
                  <span>{item.maxStock}</span>
                </div>
              </div>

              {/* Stock Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="text-lg font-bold text-primary-600">{item.safetyStock}</div>
                  <div className="text-xs text-gray-600">Safety Stock</div>
                </div>
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="text-lg font-bold text-warning-600">{item.reorderPoint}</div>
                  <div className="text-xs text-gray-600">Reorder Point</div>
                </div>
              </div>

              {/* Price */}
              <div className="flex items-center justify-between p-3 bg-primary-50 rounded-lg">
                <div className="flex items-center">
                  <DollarSign className="h-4 w-4 text-primary-600 mr-2" />
                  <span className="text-sm font-medium text-gray-900">Unit Price</span>
                </div>
                <span className="text-lg font-bold text-primary-600">${item.unitPrice}</span>
              </div>

              {/* Demand Trend Chart */}
              <div>
                <h4 className="text-sm font-medium text-gray-900 mb-2">Weekly Demand Trend</h4>
                <ResponsiveContainer width="100%" height={120}>
                  <LineChart data={item.demandTrend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                    <XAxis dataKey="day" fontSize={10} />
                    <YAxis fontSize={10} />
                    <Tooltip />
                    <Line type="monotone" dataKey="demand" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Supplier Info */}
              <div className="border-t pt-4">
                <h4 className="text-sm font-medium text-gray-900 mb-2">Supplier Information</h4>
                <div className="space-y-2">
                  <div className="flex items-center text-sm">
                    <Package className="h-4 w-4 text-gray-400 mr-2" />
                    <span className="text-gray-900">{item.supplier.name}</span>
                  </div>
                  <div className="flex items-center text-sm">
                    <Phone className="h-4 w-4 text-gray-400 mr-2" />
                    <span className="text-gray-600">{item.supplier.phone}</span>
                  </div>
                  <div className="flex items-center text-sm">
                    <Mail className="h-4 w-4 text-gray-400 mr-2" />
                    <span className="text-gray-600">{item.supplier.email}</span>
                  </div>
                </div>
              </div>

              {/* Order Dates */}
              <div className="border-t pt-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="flex items-center text-sm mb-1">
                      <Clock className="h-4 w-4 text-gray-400 mr-2" />
                      <span className="text-gray-600">Last Order</span>
                    </div>
                    <div className="text-sm font-medium text-gray-900">{item.lastOrderDate}</div>
                  </div>
                  <div>
                    <div className="flex items-center text-sm mb-1">
                      <TrendingUp className="h-4 w-4 text-gray-400 mr-2" />
                      <span className="text-gray-600">Next Order</span>
                    </div>
                    <div className="text-sm font-medium text-gray-900">{item.nextOrderDate}</div>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="border-t pt-4">
                <div className="flex space-x-2">
                  <button className="flex-1 btn-primary text-sm">
                    Place Order
                  </button>
                  <button className="flex-1 btn-secondary text-sm">
                    View Details
                  </button>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filteredStock.length === 0 && (
        <div className="card text-center py-12">
          <AlertTriangle className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No products found</h3>
          <p className="mt-1 text-sm text-gray-500">Try adjusting your search criteria.</p>
        </div>
      )}

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card text-center">
          <div className="text-2xl font-bold text-primary-600">{filteredStock.length}</div>
          <div className="text-sm text-gray-600">Total Products</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-danger-600">
            {filteredStock.filter(item => item.status === 'critical').length}
          </div>
          <div className="text-sm text-gray-600">Critical Stock</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-warning-600">
            {filteredStock.filter(item => item.status === 'low').length}
          </div>
          <div className="text-sm text-gray-600">Low Stock</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-success-600">
            {filteredStock.filter(item => item.status === 'normal').length}
          </div>
          <div className="text-sm text-gray-600">Normal Stock</div>
        </div>
      </div>
    </div>
  );
};

export default StockDetails; 