import React, { useState } from 'react';
import { 
  Users, 
  Settings, 
  Bell, 
  Globe, 
  Shield, 
  Mail,
  Phone,
  Edit,
  Trash2,
  Plus,
  Save,
  TestTube
} from 'lucide-react';
import toast from 'react-hot-toast';

const AdminSettings = () => {
  const [activeTab, setActiveTab] = useState('users');
  const [notifications, setNotifications] = useState({
    email: true,
    sms: false,
    push: true,
    lowStock: true,
    highDemand: true,
    weatherAlerts: true,
    orderUpdates: false
  });

  // Mock user data
  const users = [
    {
      id: 1,
      name: 'John Manager',
      email: 'john.manager@company.com',
      role: 'manager',
      status: 'active',
      lastLogin: '2024-01-20 10:30 AM',
      permissions: ['view', 'edit', 'approve']
    },
    {
      id: 2,
      name: 'Sarah Admin',
      email: 'sarah.admin@company.com',
      role: 'admin',
      status: 'active',
      lastLogin: '2024-01-20 09:15 AM',
      permissions: ['view', 'edit', 'approve', 'admin']
    },
    {
      id: 3,
      name: 'Mike Supervisor',
      email: 'mike.supervisor@company.com',
      role: 'supervisor',
      status: 'inactive',
      lastLogin: '2024-01-19 03:45 PM',
      permissions: ['view', 'edit']
    }
  ];

  // Mock API integrations
  const apiIntegrations = [
    {
      id: 1,
      name: 'Weather API',
      provider: 'OpenWeatherMap',
      status: 'connected',
      lastSync: '2024-01-20 11:00 AM',
      apiKey: '***wxyz789',
      endpoint: 'https://api.openweathermap.org/data/2.5/weather'
    },
    {
      id: 2,
      name: 'ERP System',
      provider: 'SAP Business One',
      status: 'disconnected',
      lastSync: '2024-01-18 02:30 PM',
      apiKey: '***abc123',
      endpoint: 'https://erp.company.com/api/v1'
    },
    {
      id: 3,
      name: 'Supplier Portal',
      provider: 'SupplierHub',
      status: 'connected',
      lastSync: '2024-01-20 10:45 AM',
      apiKey: '***def456',
      endpoint: 'https://api.supplierhub.com/v2'
    }
  ];

  const tabs = [
    { id: 'users', name: 'User Management', icon: Users },
    { id: 'integrations', name: 'API Integrations', icon: Globe },
    { id: 'notifications', name: 'Notifications', icon: Bell },
    { id: 'security', name: 'Security', icon: Shield },
    { id: 'system', name: 'System Settings', icon: Settings }
  ];

  const handleNotificationChange = (key) => {
    setNotifications(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
    toast.success('Notification settings updated!');
  };

  const handleTestAPI = (id) => {
    toast.success('API connection test successful!');
  };

  const handleSaveSettings = () => {
    toast.success('Settings saved successfully!');
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'text-success-600 bg-success-50';
      case 'inactive': return 'text-gray-600 bg-gray-50';
      case 'connected': return 'text-success-600 bg-success-50';
      case 'disconnected': return 'text-danger-600 bg-danger-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getRoleColor = (role) => {
    switch (role) {
      case 'admin': return 'text-purple-600 bg-purple-50';
      case 'manager': return 'text-blue-600 bg-blue-50';
      case 'supervisor': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const renderUsersTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-900">User Management</h3>
        <button className="btn-primary flex items-center">
          <Plus className="h-4 w-4 mr-2" />
          Add User
        </button>
      </div>

      <div className="card">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="table-header">User</th>
                <th className="table-header">Role</th>
                <th className="table-header">Status</th>
                <th className="table-header">Last Login</th>
                <th className="table-header">Permissions</th>
                <th className="table-header">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {users.map((user) => (
                <tr key={user.id} className="hover:bg-gray-50">
                  <td className="table-cell">
                    <div>
                      <div className="font-medium text-gray-900">{user.name}</div>
                      <div className="text-sm text-gray-500">{user.email}</div>
                    </div>
                  </td>
                  <td className="table-cell">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getRoleColor(user.role)}`}>
                      {user.role.charAt(0).toUpperCase() + user.role.slice(1)}
                    </span>
                  </td>
                  <td className="table-cell">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(user.status)}`}>
                      {user.status.charAt(0).toUpperCase() + user.status.slice(1)}
                    </span>
                  </td>
                  <td className="table-cell">
                    <div className="text-sm text-gray-900">{user.lastLogin}</div>
                  </td>
                  <td className="table-cell">
                    <div className="flex flex-wrap gap-1">
                      {user.permissions.map((permission) => (
                        <span key={permission} className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-800">
                          {permission}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="table-cell">
                    <div className="flex space-x-2">
                      <button className="inline-flex items-center p-1 border border-transparent rounded-full shadow-sm text-white bg-primary-600 hover:bg-primary-700">
                        <Edit className="h-4 w-4" />
                      </button>
                      <button className="inline-flex items-center p-1 border border-transparent rounded-full shadow-sm text-white bg-danger-600 hover:bg-danger-700">
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderIntegrationsTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-900">API Integrations</h3>
        <button className="btn-primary flex items-center">
          <Plus className="h-4 w-4 mr-2" />
          Add Integration
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {apiIntegrations.map((integration) => (
          <div key={integration.id} className="card">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h4 className="font-medium text-gray-900">{integration.name}</h4>
                <p className="text-sm text-gray-500">{integration.provider}</p>
              </div>
              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(integration.status)}`}>
                {integration.status.charAt(0).toUpperCase() + integration.status.slice(1)}
              </span>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">API Key:</span>
                <span className="font-mono text-gray-900">{integration.apiKey}</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Last Sync:</span>
                <span className="text-gray-900">{integration.lastSync}</span>
              </div>
              <div className="text-sm">
                <span className="text-gray-600">Endpoint:</span>
                <div className="font-mono text-gray-900 text-xs mt-1 break-all">{integration.endpoint}</div>
              </div>
            </div>

            <div className="flex space-x-2 mt-4">
              <button
                onClick={() => handleTestAPI(integration.id)}
                className="btn-secondary flex items-center text-sm"
              >
                <TestTube className="h-4 w-4 mr-2" />
                Test Connection
              </button>
              <button className="btn-primary flex items-center text-sm">
                <Edit className="h-4 w-4 mr-2" />
                Configure
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderNotificationsTab = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900">Notification Settings</h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Delivery Methods */}
        <div className="card">
          <h4 className="font-medium text-gray-900 mb-4">Delivery Methods</h4>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Mail className="h-5 w-5 text-gray-400 mr-3" />
                <span className="text-sm font-medium text-gray-900">Email Notifications</span>
              </div>
              <button
                onClick={() => handleNotificationChange('email')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  notifications.email ? 'bg-primary-600' : 'bg-gray-200'
                }`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  notifications.email ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Phone className="h-5 w-5 text-gray-400 mr-3" />
                <span className="text-sm font-medium text-gray-900">SMS Notifications</span>
              </div>
              <button
                onClick={() => handleNotificationChange('sms')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  notifications.sms ? 'bg-primary-600' : 'bg-gray-200'
                }`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  notifications.sms ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Bell className="h-5 w-5 text-gray-400 mr-3" />
                <span className="text-sm font-medium text-gray-900">Push Notifications</span>
              </div>
              <button
                onClick={() => handleNotificationChange('push')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  notifications.push ? 'bg-primary-600' : 'bg-gray-200'
                }`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  notifications.push ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>
          </div>
        </div>

        {/* Alert Types */}
        <div className="card">
          <h4 className="font-medium text-gray-900 mb-4">Alert Types</h4>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-900">Low Stock Alerts</span>
              <button
                onClick={() => handleNotificationChange('lowStock')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  notifications.lowStock ? 'bg-primary-600' : 'bg-gray-200'
                }`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  notifications.lowStock ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-900">High Demand Predictions</span>
              <button
                onClick={() => handleNotificationChange('highDemand')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  notifications.highDemand ? 'bg-primary-600' : 'bg-gray-200'
                }`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  notifications.highDemand ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-900">Weather Alerts</span>
              <button
                onClick={() => handleNotificationChange('weatherAlerts')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  notifications.weatherAlerts ? 'bg-primary-600' : 'bg-gray-200'
                }`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  notifications.weatherAlerts ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-900">Order Updates</span>
              <button
                onClick={() => handleNotificationChange('orderUpdates')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  notifications.orderUpdates ? 'bg-primary-600' : 'bg-gray-200'
                }`}
              >
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  notifications.orderUpdates ? 'translate-x-6' : 'translate-x-1'
                }`} />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-end">
        <button onClick={handleSaveSettings} className="btn-primary flex items-center">
          <Save className="h-4 w-4 mr-2" />
          Save Settings
        </button>
      </div>
    </div>
  );

  const renderSecurityTab = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900">Security Settings</h3>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h4 className="font-medium text-gray-900 mb-4">Authentication</h4>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Session Timeout (minutes)
              </label>
              <input type="number" className="input-field" defaultValue="30" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Maximum Login Attempts
              </label>
              <input type="number" className="input-field" defaultValue="5" />
            </div>
            <div className="flex items-center">
              <input type="checkbox" className="h-4 w-4 text-primary-600" defaultChecked />
              <label className="ml-2 text-sm text-gray-900">Require Two-Factor Authentication</label>
            </div>
          </div>
        </div>

        <div className="card">
          <h4 className="font-medium text-gray-900 mb-4">Data Protection</h4>
          <div className="space-y-4">
            <div className="flex items-center">
              <input type="checkbox" className="h-4 w-4 text-primary-600" defaultChecked />
              <label className="ml-2 text-sm text-gray-900">Enable Data Encryption</label>
            </div>
            <div className="flex items-center">
              <input type="checkbox" className="h-4 w-4 text-primary-600" defaultChecked />
              <label className="ml-2 text-sm text-gray-900">Automatic Backup</label>
            </div>
            <div className="flex items-center">
              <input type="checkbox" className="h-4 w-4 text-primary-600" />
              <label className="ml-2 text-sm text-gray-900">Audit Logging</label>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSystemTab = () => (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-900">System Settings</h3>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h4 className="font-medium text-gray-900 mb-4">General</h4>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Company Name
              </label>
              <input type="text" className="input-field" defaultValue="Inventory Management Corp" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Time Zone
              </label>
              <select className="input-field">
                <option>UTC-5 (Eastern Time)</option>
                <option>UTC-8 (Pacific Time)</option>
                <option>UTC+0 (GMT)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Currency
              </label>
              <select className="input-field">
                <option>USD ($)</option>
                <option>EUR (€)</option>
                <option>GBP (£)</option>
              </select>
            </div>
          </div>
        </div>

        <div className="card">
          <h4 className="font-medium text-gray-900 mb-4">Performance</h4>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Data Refresh Interval (minutes)
              </label>
              <input type="number" className="input-field" defaultValue="15" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Cache Duration (hours)
              </label>
              <input type="number" className="input-field" defaultValue="24" />
            </div>
            <div className="flex items-center">
              <input type="checkbox" className="h-4 w-4 text-primary-600" defaultChecked />
              <label className="ml-2 text-sm text-gray-900">Enable Performance Monitoring</label>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'users':
        return renderUsersTab();
      case 'integrations':
        return renderIntegrationsTab();
      case 'notifications':
        return renderNotificationsTab();
      case 'security':
        return renderSecurityTab();
      case 'system':
        return renderSystemTab();
      default:
        return renderUsersTab();
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Admin Settings</h1>
        <p className="text-gray-600">Manage system configuration and user access</p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="h-4 w-4 mr-2" />
                {tab.name}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      {renderTabContent()}
    </div>
  );
};

export default AdminSettings; 