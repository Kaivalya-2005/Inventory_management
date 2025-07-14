import React, { useState } from 'react';
import { 
  Search, 
  Check, 
  X, 
  Edit, 
  AlertTriangle,
  TrendingUp,
  Package
} from 'lucide-react';
import toast from 'react-hot-toast';

const StockSuggestions = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedUrgency, setSelectedUrgency] = useState('all');

  // Mock data for stock suggestions
  const suggestions = [
    {
      id: 1,
      productName: 'Organic Bananas',
      category: 'Fruits',
      suggestedQuantity: 500,
      confidenceLevel: 95,
      factors: ['High temperature', 'Weekend demand', 'Previous sales trend'],
      urgency: 'high',
      currentStock: 50,
      supplier: 'Fresh Farms Inc.',
      price: '$0.89/lb'
    },
    {
      id: 2,
      productName: 'Fresh Milk',
      category: 'Dairy',
      suggestedQuantity: 200,
      confidenceLevel: 88,
      factors: ['Daily essential', 'Expiration date', 'Customer demand'],
      urgency: 'medium',
      currentStock: 75,
      supplier: 'Dairy Co.',
      price: '$3.99/gallon'
    },
    {
      id: 3,
      productName: 'Ice Cream',
      category: 'Frozen',
      suggestedQuantity: 150,
      confidenceLevel: 92,
      factors: ['High temperature', 'Summer season', 'Weekend sales'],
      urgency: 'high',
      currentStock: 25,
      supplier: 'Frozen Delights',
      price: '$4.99/pint'
    },
    {
      id: 4,
      productName: 'Bread',
      category: 'Bakery',
      suggestedQuantity: 300,
      confidenceLevel: 85,
      factors: ['Daily essential', 'Short shelf life', 'Breakfast demand'],
      urgency: 'medium',
      currentStock: 100,
      supplier: 'Local Bakery',
      price: '$2.49/loaf'
    },
    {
      id: 5,
      productName: 'Bottled Water',
      category: 'Beverages',
      suggestedQuantity: 400,
      confidenceLevel: 90,
      factors: ['High temperature', 'Hydration demand', 'Event weekend'],
      urgency: 'high',
      currentStock: 30,
      supplier: 'Pure Water Co.',
      price: '$0.99/bottle'
    }
  ];

  const categories = ['all', 'Fruits', 'Dairy', 'Frozen', 'Bakery', 'Beverages'];
  const urgencyLevels = ['all', 'high', 'medium', 'low'];

  const filteredSuggestions = suggestions.filter(suggestion => {
    const matchesSearch = suggestion.productName.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         suggestion.category.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || suggestion.category === selectedCategory;
    const matchesUrgency = selectedUrgency === 'all' || suggestion.urgency === selectedUrgency;
    
    return matchesSearch && matchesCategory && matchesUrgency;
  });

  const handleAccept = (id) => {
    toast.success(`Order accepted for suggestion #${id}`);
  };

  const handleReject = (id) => {
    toast.error(`Order rejected for suggestion #${id}`);
  };

  const handleModify = (id) => {
    toast.info(`Modify order for suggestion #${id}`);
  };

  const getConfidenceColor = (level) => {
    if (level >= 90) return 'text-success-600 bg-success-50';
    if (level >= 80) return 'text-warning-600 bg-warning-50';
    return 'text-danger-600 bg-danger-50';
  };

  const getUrgencyColor = (urgency) => {
    switch (urgency) {
      case 'high': return 'text-danger-600 bg-danger-50';
      case 'medium': return 'text-warning-600 bg-warning-50';
      case 'low': return 'text-success-600 bg-success-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Stock Suggestions</h1>
        <p className="text-gray-600">AI-powered recommendations for stock replenishment</p>
      </div>

      {/* Filters and Search */}
      <div className="card">
        <div className="flex flex-col md:flex-row gap-4">
          {/* Search */}
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
              <input
                type="text"
                placeholder="Search products or categories..."
                className="input-field pl-10"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          </div>

          {/* Category Filter */}
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

          {/* Urgency Filter */}
          <div className="md:w-48">
            <select
              className="input-field"
              value={selectedUrgency}
              onChange={(e) => setSelectedUrgency(e.target.value)}
            >
              {urgencyLevels.map(level => (
                <option key={level} value={level}>
                  {level === 'all' ? 'All Urgency' : level.charAt(0).toUpperCase() + level.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Suggestions Table */}
      <div className="card">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="table-header">Product</th>
                <th className="table-header">Category</th>
                <th className="table-header">Current Stock</th>
                <th className="table-header">Suggested Qty</th>
                <th className="table-header">Confidence</th>
                <th className="table-header">Urgency</th>
                <th className="table-header">Factors</th>
                <th className="table-header">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredSuggestions.map((suggestion) => (
                <tr key={suggestion.id} className="hover:bg-gray-50">
                  <td className="table-cell">
                    <div>
                      <div className="font-medium text-gray-900">{suggestion.productName}</div>
                      <div className="text-sm text-gray-500">{suggestion.supplier}</div>
                      <div className="text-sm text-gray-500">{suggestion.price}</div>
                    </div>
                  </td>
                  <td className="table-cell">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                      {suggestion.category}
                    </span>
                  </td>
                  <td className="table-cell">
                    <div className="flex items-center">
                      <Package className="h-4 w-4 text-gray-400 mr-2" />
                      {suggestion.currentStock}
                    </div>
                  </td>
                  <td className="table-cell">
                    <div className="font-medium text-gray-900">{suggestion.suggestedQuantity}</div>
                  </td>
                  <td className="table-cell">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getConfidenceColor(suggestion.confidenceLevel)}`}>
                      {suggestion.confidenceLevel}%
                    </span>
                  </td>
                  <td className="table-cell">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getUrgencyColor(suggestion.urgency)}`}>
                      {suggestion.urgency.charAt(0).toUpperCase() + suggestion.urgency.slice(1)}
                    </span>
                  </td>
                  <td className="table-cell">
                    <div className="text-sm text-gray-900">
                      {suggestion.factors.map((factor, index) => (
                        <div key={index} className="flex items-center mb-1">
                          <TrendingUp className="h-3 w-3 text-primary-500 mr-1" />
                          {factor}
                        </div>
                      ))}
                    </div>
                  </td>
                  <td className="table-cell">
                    <div className="flex space-x-2">
                      <button
                        onClick={() => handleAccept(suggestion.id)}
                        className="inline-flex items-center p-1 border border-transparent rounded-full shadow-sm text-white bg-success-600 hover:bg-success-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-success-500"
                        title="Accept"
                      >
                        <Check className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => handleReject(suggestion.id)}
                        className="inline-flex items-center p-1 border border-transparent rounded-full shadow-sm text-white bg-danger-600 hover:bg-danger-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-danger-500"
                        title="Reject"
                      >
                        <X className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => handleModify(suggestion.id)}
                        className="inline-flex items-center p-1 border border-transparent rounded-full shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                        title="Modify"
                      >
                        <Edit className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {filteredSuggestions.length === 0 && (
          <div className="text-center py-12">
            <AlertTriangle className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No suggestions found</h3>
            <p className="mt-1 text-sm text-gray-500">Try adjusting your search or filter criteria.</p>
          </div>
        )}
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card text-center">
          <div className="text-2xl font-bold text-primary-600">{filteredSuggestions.length}</div>
          <div className="text-sm text-gray-600">Total Suggestions</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-warning-600">
            {filteredSuggestions.filter(s => s.urgency === 'high').length}
          </div>
          <div className="text-sm text-gray-600">High Priority</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-success-600">
            {filteredSuggestions.reduce((sum, s) => sum + s.suggestedQuantity, 0).toLocaleString()}
          </div>
          <div className="text-sm text-gray-600">Total Units</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-gray-600">
            {Math.round(filteredSuggestions.reduce((sum, s) => sum + s.confidenceLevel, 0) / filteredSuggestions.length || 0)}%
          </div>
          <div className="text-sm text-gray-600">Avg Confidence</div>
        </div>
      </div>
    </div>
  );
};

export default StockSuggestions; 