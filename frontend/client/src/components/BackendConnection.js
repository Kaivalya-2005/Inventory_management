import React, { useState, useEffect } from 'react';
import { inventoryAPI } from '../services/api';

const BackendConnection = () => {
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [healthData, setHealthData] = useState(null);
  const [suggestionsData, setSuggestionsData] = useState(null);

  useEffect(() => {
    const testConnection = async () => {
      try {
        // Test health endpoint
        const health = await inventoryAPI.healthCheck();
        setHealthData(health);
        
        // Test stock suggestions endpoint
        const suggestions = await inventoryAPI.getStockSuggestions();
        setSuggestionsData(suggestions);
        
        setConnectionStatus('connected');
      } catch (error) {
        console.error('Backend connection failed:', error);
        setConnectionStatus('failed');
      }
    };

    testConnection();
  }, []);

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Backend Connection Status</h3>
      
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${
            connectionStatus === 'connected' ? 'bg-green-500' : 
            connectionStatus === 'failed' ? 'bg-red-500' : 'bg-yellow-500'
          }`}></div>
          <span className="font-medium">
            {connectionStatus === 'connected' ? 'Connected to Backend' :
             connectionStatus === 'failed' ? 'Connection Failed' : 'Checking Connection...'}
          </span>
        </div>

        {healthData && (
          <div className="bg-green-50 p-3 rounded">
            <p className="text-sm text-green-800">
              Health Check: {JSON.stringify(healthData)}
            </p>
          </div>
        )}

        {suggestionsData && (
          <div className="bg-blue-50 p-3 rounded">
            <p className="text-sm text-blue-800">
              Stock Suggestions: {suggestionsData.suggestions?.length || 0} suggestions loaded
            </p>
          </div>
        )}

        {connectionStatus === 'failed' && (
          <div className="bg-red-50 p-3 rounded">
            <p className="text-sm text-red-800">
              Make sure your FastAPI backend is running on http://localhost:8000
            </p>
            <p className="text-sm text-red-600 mt-1">
              Run: uvicorn app:app --reload (from backend directory)
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default BackendConnection; 