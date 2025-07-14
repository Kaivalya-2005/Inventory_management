import React, { useState } from 'react';
import { 
  Thermometer, 
  Calendar, 
  Upload, 
  Save, 
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Globe,
  Cloud,
  Sun,
  CloudRain,
  Snowflake
} from 'lucide-react';
import toast from 'react-hot-toast';

const ModelFactors = () => {
  const [temperature, setTemperature] = useState(28);
  const [demandForecast, setDemandForecast] = useState(1200);
  const [specialEvents, setSpecialEvents] = useState([
    { id: 1, name: 'Summer Festival', date: '2024-07-15', impact: 'high', active: true },
    { id: 2, name: 'Flash Sale', date: '2024-01-25', impact: 'medium', active: true },
    { id: 3, name: 'Holiday Weekend', date: '2024-02-10', impact: 'high', active: false }
  ]);
  const [newEvent, setNewEvent] = useState({ name: '', date: '', impact: 'medium' });
  const [weatherData, setWeatherData] = useState({
    current: 28,
    forecast: [26, 29, 31, 28, 25, 27, 30],
    condition: 'sunny',
    humidity: 65,
    windSpeed: 12
  });

  const weatherConditions = [
    { name: 'sunny', icon: Sun, color: 'text-yellow-500' },
    { name: 'cloudy', icon: Cloud, color: 'text-gray-500' },
    { name: 'rainy', icon: CloudRain, color: 'text-blue-500' },
    { name: 'snowy', icon: Snowflake, color: 'text-blue-300' }
  ];

  const getWeatherIcon = (condition) => {
    const weather = weatherConditions.find(w => w.name === condition);
    return weather ? weather.icon : Sun;
  };

  const getWeatherColor = (condition) => {
    const weather = weatherConditions.find(w => w.name === condition);
    return weather ? weather.color : 'text-gray-500';
  };

  const handleSaveFactors = () => {
    toast.success('Model factors updated successfully!');
  };

  const handleRefreshWeather = () => {
    // Mock weather API call
    setWeatherData(prev => ({
      ...prev,
      current: Math.floor(Math.random() * 15) + 20,
      forecast: Array.from({ length: 7 }, () => Math.floor(Math.random() * 15) + 20)
    }));
    toast.success('Weather data refreshed!');
  };

  const handleUploadData = (event) => {
    const file = event.target.files[0];
    if (file) {
      toast.success(`File "${file.name}" uploaded successfully!`);
    }
  };

  const handleAddEvent = () => {
    if (newEvent.name && newEvent.date) {
      const event = {
        id: Date.now(),
        ...newEvent,
        active: true
      };
      setSpecialEvents([...specialEvents, event]);
      setNewEvent({ name: '', date: '', impact: 'medium' });
      toast.success('Special event added!');
    } else {
      toast.error('Please fill in all fields');
    }
  };

  const handleToggleEvent = (id) => {
    setSpecialEvents(events => 
      events.map(event => 
        event.id === id ? { ...event, active: !event.active } : event
      )
    );
  };

  const handleDeleteEvent = (id) => {
    setSpecialEvents(events => events.filter(event => event.id !== id));
    toast.success('Event removed!');
  };

  const getImpactColor = (impact) => {
    switch (impact) {
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
        <h1 className="text-2xl font-bold text-gray-900">Model Input Factors</h1>
        <p className="text-gray-600">Configure and manage factors that influence stock predictions</p>
      </div>

      {/* Current Weather */}
      <div className="card">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Current Weather</h3>
          <button
            onClick={handleRefreshWeather}
            className="btn-secondary flex items-center"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg">
            <div className="flex justify-center mb-2">
              <Thermometer className="h-8 w-8 text-blue-600" />
            </div>
            <div className="text-2xl font-bold text-blue-900">{weatherData.current}°C</div>
            <div className="text-sm text-blue-700">Current Temperature</div>
          </div>

          <div className="text-center p-4 bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg">
            <div className="flex justify-center mb-2">
              {React.createElement(getWeatherIcon(weatherData.condition), { 
                className: `h-8 w-8 ${getWeatherColor(weatherData.condition)}` 
              })}
            </div>
            <div className="text-lg font-semibold text-gray-900 capitalize">{weatherData.condition}</div>
            <div className="text-sm text-gray-600">Weather Condition</div>
          </div>

          <div className="text-center p-4 bg-gradient-to-br from-green-50 to-green-100 rounded-lg">
            <div className="flex justify-center mb-2">
              <Cloud className="h-8 w-8 text-green-600" />
            </div>
            <div className="text-2xl font-bold text-green-900">{weatherData.humidity}%</div>
            <div className="text-sm text-green-700">Humidity</div>
          </div>

          <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg">
            <div className="flex justify-center mb-2">
              <Globe className="h-8 w-8 text-purple-600" />
            </div>
            <div className="text-2xl font-bold text-purple-900">{weatherData.windSpeed} km/h</div>
            <div className="text-sm text-purple-700">Wind Speed</div>
          </div>
        </div>

        {/* Weather Forecast */}
        <div className="mt-6">
          <h4 className="text-md font-medium text-gray-900 mb-3">7-Day Forecast</h4>
          <div className="grid grid-cols-7 gap-2">
            {weatherData.forecast.map((temp, index) => (
              <div key={index} className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-600">Day {index + 1}</div>
                <div className="text-lg font-bold text-gray-900">{temp}°C</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Manual Temperature Override */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Manual Temperature Override</h3>
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Temperature (°C)
            </label>
            <input
              type="number"
              value={temperature}
              onChange={(e) => setTemperature(parseInt(e.target.value))}
              className="input-field"
              min="0"
              max="50"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={handleSaveFactors}
              className="btn-primary flex items-center"
            >
              <Save className="h-4 w-4 mr-2" />
              Save
            </button>
          </div>
        </div>
      </div>

      {/* Demand Forecast */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Demand Forecast</h3>
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Expected Daily Demand (units)
            </label>
            <input
              type="number"
              value={demandForecast}
              onChange={(e) => setDemandForecast(parseInt(e.target.value))}
              className="input-field"
              min="0"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={handleSaveFactors}
              className="btn-primary flex items-center"
            >
              <Save className="h-4 w-4 mr-2" />
              Save
            </button>
          </div>
        </div>
      </div>

      {/* Special Events */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Special Events & Situations</h3>
        
        {/* Add New Event */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Event Name</label>
            <input
              type="text"
              value={newEvent.name}
              onChange={(e) => setNewEvent({ ...newEvent, name: e.target.value })}
              className="input-field"
              placeholder="e.g., Summer Festival"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Date</label>
            <input
              type="date"
              value={newEvent.date}
              onChange={(e) => setNewEvent({ ...newEvent, date: e.target.value })}
              className="input-field"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Impact</label>
            <select
              value={newEvent.impact}
              onChange={(e) => setNewEvent({ ...newEvent, impact: e.target.value })}
              className="input-field"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
          <div className="flex items-end">
            <button
              onClick={handleAddEvent}
              className="btn-primary w-full"
            >
              Add Event
            </button>
          </div>
        </div>

        {/* Events List */}
        <div className="space-y-3">
          {specialEvents.map((event) => (
            <div key={event.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-4">
                <div className="flex items-center">
                  {event.active ? (
                    <CheckCircle className="h-5 w-5 text-success-600" />
                  ) : (
                    <AlertCircle className="h-5 w-5 text-gray-400" />
                  )}
                </div>
                <div>
                  <div className="font-medium text-gray-900">{event.name}</div>
                  <div className="flex items-center space-x-4 text-sm text-gray-500">
                    <div className="flex items-center">
                      <Calendar className="h-4 w-4 mr-1" />
                      {event.date}
                    </div>
                    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getImpactColor(event.impact)}`}>
                      {event.impact.charAt(0).toUpperCase() + event.impact.slice(1)} Impact
                    </span>
                  </div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => handleToggleEvent(event.id)}
                  className={`px-3 py-1 rounded text-sm font-medium ${
                    event.active 
                      ? 'bg-success-100 text-success-700 hover:bg-success-200' 
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {event.active ? 'Active' : 'Inactive'}
                </button>
                <button
                  onClick={() => handleDeleteEvent(event.id)}
                  className="px-3 py-1 rounded text-sm font-medium bg-danger-100 text-danger-700 hover:bg-danger-200"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Data Upload */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Import</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Upload CSV Data
            </label>
            <div className="flex items-center space-x-4">
              <input
                type="file"
                accept=".csv"
                onChange={handleUploadData}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
              />
              <button className="btn-secondary flex items-center">
                <Upload className="h-4 w-4 mr-2" />
                Upload
              </button>
            </div>
          </div>
          
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <AlertCircle className="h-5 w-5 text-blue-400" />
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-blue-800">Data Format Requirements</h3>
                <div className="mt-2 text-sm text-blue-700">
                  <p>CSV should include columns: Date, Temperature, Demand, Event, Impact</p>
                  <p>Example: 2024-01-15, 25, 150, Summer Festival, high</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* API Integrations */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">API Integrations</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 border border-gray-200 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-gray-900">Weather API</h4>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-success-100 text-success-800">
                Connected
              </span>
            </div>
            <p className="text-sm text-gray-600 mb-3">Auto-updates temperature and weather conditions</p>
            <button className="btn-secondary text-sm">Configure</button>
          </div>
          
          <div className="p-4 border border-gray-200 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-gray-900">ERP System</h4>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-warning-100 text-warning-800">
                Disconnected
              </span>
            </div>
            <p className="text-sm text-gray-600 mb-3">Sync with enterprise resource planning system</p>
            <button className="btn-primary text-sm">Connect</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelFactors; 