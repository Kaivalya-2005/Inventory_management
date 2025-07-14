import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './contexts/AuthContext';
import Layout from './components/Layout';
import ScrollToTop from './components/ScrollToTop';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import StockSuggestions from './pages/StockSuggestions';
import HistoricalData from './pages/HistoricalData';
import StockDetails from './pages/StockDetails';
import ModelFactors from './pages/ModelFactors';
import ModelInsights from './pages/ModelInsights';
import OrderManagement from './pages/OrderManagement';
import AdminSettings from './pages/AdminSettings';

function App() {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Login />;
  }

  return (
    <Layout>
      <ScrollToTop />
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/stock-suggestions" element={<StockSuggestions />} />
        <Route path="/historical-data" element={<HistoricalData />} />
        <Route path="/stock-details" element={<StockDetails />} />
        <Route path="/model-factors" element={<ModelFactors />} />
        <Route path="/model-insights" element={<ModelInsights />} />
        <Route path="/order-management" element={<OrderManagement />} />
        <Route path="/admin-settings" element={<AdminSettings />} />
      </Routes>
    </Layout>
  );
}

export default App; 