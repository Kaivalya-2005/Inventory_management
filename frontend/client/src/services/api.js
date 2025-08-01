const ADMIN_API_BASE_URL = process.env.REACT_APP_ADMIN_API_URL || 'http://localhost:5000';
const ML_API_BASE_URL = process.env.REACT_APP_ML_API_URL || 'http://localhost:8000';

// Helper function for API calls
const apiCall = async (endpoint, options = {}) => {
  let url;
  // Route based on endpoint
  if (
    endpoint.startsWith('/api/admin') ||
    endpoint.startsWith('/api/auth') ||
    endpoint.startsWith('/api/orders') // <-- Ensure orders go to Express backend
  ) {
    url = `${ADMIN_API_BASE_URL}${endpoint}`;
  } else {
    url = `${ML_API_BASE_URL}${endpoint}`;
  }

  // Get token from localStorage (or sessionStorage if you prefer)
  const token = localStorage.getItem('token');

  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` }),
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);
    if (!response.ok) {
      throw new Error(`API call failed: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('API call error:', error);
    throw error;
  }
};

// Authentication API calls
export const authAPI = {
  login: async (email, password) => {
    const response = await apiCall('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    if (response.success) {
      localStorage.setItem('token', response.token);
      localStorage.setItem('user', JSON.stringify(response.user));
    }
    return response;
  },

  register: async (userData) => {
    const response = await apiCall('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
    if (response.success) {
      localStorage.setItem('token', response.token);
      localStorage.setItem('user', JSON.stringify(response.user));
    }
    return response;
  },

  demoLogin: async () => {
    const response = await apiCall('/api/auth/demo', {
      method: 'POST',
    });
    if (response.success) {
      localStorage.setItem('token', response.token);
      localStorage.setItem('user', JSON.stringify(response.user));
    }
    return response;
  },

  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  },

  getProfile: async () => {
    return apiCall('/api/auth/profile');
  },

  updateProfile: async (profileData) => {
    return apiCall('/api/auth/profile', {
      method: 'PUT',
      body: JSON.stringify(profileData),
    });
  },

  changePassword: async (currentPassword, newPassword) => {
    return apiCall('/api/auth/change-password', {
      method: 'PUT',
      body: JSON.stringify({ currentPassword, newPassword }),
    });
  }
};

// Inventory and ML API calls
export const inventoryAPI = {
  // Get AI-powered stock suggestions
  getStockSuggestions: async () => {
    return apiCall('/ai-stock-suggestions');
  },

  // Get current stock levels
  getCurrentStock: async () => {
    return apiCall('/current-stock');
  },

  // Get historical accuracy data
  getHistoricalAccuracy: async () => {
    return apiCall('/historical-accuracy');
  },

  // Make a single day prediction
  predictInventory: async (data) => {
    return apiCall('/predict-inventory', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  // Make weekly predictions
  predictWeekly: async (data) => {
    return apiCall('/predict-weekly', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  // Health check
  healthCheck: async () => {
    return apiCall('/health');
  }
};

// Dashboard API calls
export const dashboardAPI = {
  // Get dashboard metrics (combining multiple endpoints)
  getDashboardData: async () => {
    try {
      const [stockSuggestions, currentStock, historicalAccuracy] = await Promise.all([
        inventoryAPI.getStockSuggestions(),
        inventoryAPI.getCurrentStock(),
        inventoryAPI.getHistoricalAccuracy()
      ]);

      // Calculate dashboard metrics from real data
      const totalStock = currentStock.items?.reduce((sum, item) => sum + item.stock, 0) || 0;
      const outOfStock = currentStock.items?.filter(item => item.stock === 0).length || 0;
      const suggestions = stockSuggestions.suggestions?.length || 0;
      
      // Mock temperature data (you can add a real temperature API later)
      const currentTemp = Math.floor(Math.random() * 15) + 20; // 20-35°C

      return {
        metrics: {
          totalStock: totalStock.toLocaleString(),
          predictedOrders: suggestions,
          outOfStock,
          currentTemp: `${currentTemp}°C`
        },
        stockSuggestions: stockSuggestions.suggestions || [],
        currentStock: currentStock.items || [],
        historicalAccuracy: historicalAccuracy.history || []
      };
    } catch (error) {
      console.error('Dashboard data fetch error:', error);
      // Return fallback data if API fails
      return {
        metrics: {
          totalStock: '12,450',
          predictedOrders: '1,234',
          outOfStock: '23',
          currentTemp: '28°C'
        },
        stockSuggestions: [],
        currentStock: [],
        historicalAccuracy: []
      };
    }
  }
};

// Admin API calls
export const adminAPI = {
  // User Management
  getUsers: async () => {
    return apiCall('/api/admin/users');
  },

  createUser: async (userData) => {
    return apiCall('/api/admin/users', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  },

  updateUser: async (id, userData) => {
    return apiCall(`/api/admin/users/${id}`, {
      method: 'PUT',
      body: JSON.stringify(userData),
    });
  },

  deleteUser: async (id) => {
    return apiCall(`/api/admin/users/${id}`, {
      method: 'DELETE',
    });
  },

  // System Settings
  getSettings: async () => {
    return apiCall('/api/admin/settings');
  },

  updateSettings: async (settings) => {
    return apiCall('/api/admin/settings', {
      method: 'PUT',
      body: JSON.stringify(settings),
    });
  },

  // API Integrations
  getIntegrations: async () => {
    return apiCall('/api/admin/integrations');
  },

  createIntegration: async (integrationData) => {
    return apiCall('/api/admin/integrations', {
      method: 'POST',
      body: JSON.stringify(integrationData),
    });
  },

  updateIntegration: async (id, integrationData) => {
    return apiCall(`/api/admin/integrations/${id}`, {
      method: 'PUT',
      body: JSON.stringify(integrationData),
    });
  },

  deleteIntegration: async (id) => {
    return apiCall(`/api/admin/integrations/${id}`, {
      method: 'DELETE',
    });
  },

  testIntegration: async (id) => {
    return apiCall(`/api/admin/integrations/${id}/test`, {
      method: 'POST',
    });
  }
};

export const orderAPI = {
  getOrders: async () => apiCall('/api/orders' , { base: 'admin' }),
  getOrder: async (id) => apiCall(`/api/orders/${id}` , { base: 'admin' }),
  createOrder: async (orderData) => apiCall('/api/orders', {
    method: 'POST',
    body: JSON.stringify(orderData),
  }),
  updateOrder: async (id, orderData) => apiCall(`/api/orders/${id}`, {
    method: 'PUT',
    body: JSON.stringify(orderData),
  }),
  deleteOrder: async (id) => apiCall(`/api/orders/${id}`, {
    method: 'DELETE',
  }),
  approveOrder: async (id) => apiCall(`/api/orders/${id}/approve`, {
    method: 'POST',
  }),
  rejectOrder: async (id) => apiCall(`/api/orders/${id}/reject`, {
    method: 'POST',
  }),
};

export default {
  auth: authAPI,
  inventory: inventoryAPI,
  dashboard: dashboardAPI,
  admin: adminAPI,
  orders: orderAPI,
}; 