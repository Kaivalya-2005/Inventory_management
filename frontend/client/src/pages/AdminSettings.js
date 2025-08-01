import React, { useState, useEffect } from 'react';
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
  TestTube,
  X
} from 'lucide-react';
import toast from 'react-hot-toast';
import { adminAPI } from '../services/api';

const defaultUser = {
  name: '',
  email: '',
  role: 'manager',
  status: 'active',
  permissions: ['view'],
  password: '', // Add password to defaultUser
};
const defaultIntegration = {
  name: '',
  provider: '',
  apiKey: '',
  endpoint: '',
};

const AdminSettings = () => {
  const [activeTab, setActiveTab] = useState('users');
  // Users
  const [users, setUsers] = useState([]);
  const [userModalOpen, setUserModalOpen] = useState(false);
  const [editUser, setEditUser] = useState(null);
  const [userForm, setUserForm] = useState(defaultUser);
  const [usersLoading, setUsersLoading] = useState(false);
  // Integrations
  const [integrations, setIntegrations] = useState([]);
  const [integrationModalOpen, setIntegrationModalOpen] = useState(false);
  const [editIntegration, setEditIntegration] = useState(null);
  const [integrationForm, setIntegrationForm] = useState(defaultIntegration);
  const [integrationsLoading, setIntegrationsLoading] = useState(false);
  // Settings
  const [settings, setSettings] = useState(null);
  const [settingsLoading, setSettingsLoading] = useState(false);
  // Local state for settings tabs
  const [notifications, setNotifications] = useState({});
  const [security, setSecurity] = useState({});
  const [system, setSystem] = useState({});
  // Loading state for saving
  const [savingSettings, setSavingSettings] = useState(false);
  const [passwordError, setPasswordError] = useState('');

  // Fetch all data on mount
  useEffect(() => {
    fetchAll();
  }, []);

  const fetchAll = async () => {
    setUsersLoading(true);
    setIntegrationsLoading(true);
    setSettingsLoading(true);
    try {
      const [usersRes, integrationsRes, settingsRes] = await Promise.all([
        adminAPI.getUsers(),
        adminAPI.getIntegrations(),
        adminAPI.getSettings(),
      ]);
      setUsers(usersRes.users || []);
      setIntegrations(integrationsRes.integrations || []);
      setSettings(settingsRes.settings || {});
      setNotifications(settingsRes.settings?.notifications || {});
      setSecurity(settingsRes.settings?.security || {});
      setSystem(settingsRes.settings?.system || {});
    } catch (e) {
      toast.error('Failed to load admin data');
    } finally {
      setUsersLoading(false);
      setIntegrationsLoading(false);
      setSettingsLoading(false);
    }
  };

  // ===== USER MANAGEMENT =====
  const openUserModal = (user = null) => {
    setEditUser(user);
    setUserForm(user ? { ...user } : defaultUser);
    setUserModalOpen(true);
  };
  const closeUserModal = () => {
    setUserModalOpen(false);
    setEditUser(null);
    setUserForm(defaultUser);
  };
  const handleUserFormChange = (e) => {
    const { name, value } = e.target;
    setUserForm((prev) => ({ ...prev, [name]: value }));
    if (name === 'password') {
      if (value.length > 0 && value.length < 6) {
        setPasswordError('Password must be at least 6 characters');
      } else {
        setPasswordError('');
      }
    }
  };
  const handleUserPermissionChange = (perm) => {
    setUserForm((prev) => ({
      ...prev,
      permissions: prev.permissions.includes(perm)
        ? prev.permissions.filter((p) => p !== perm)
        : [...prev.permissions, perm],
    }));
  };
  const handleUserSubmit = async (e) => {
    e.preventDefault();
    try {
      if (editUser) {
        await adminAPI.updateUser(editUser.id, userForm);
        toast.success('User updated');
      } else {
        await adminAPI.createUser(userForm);
        toast.success('User created');
      }
      closeUserModal();
      fetchAll();
    } catch (e) {
      toast.error('Failed to save user');
    }
  };
  const handleDeleteUser = async (id) => {
    if (!window.confirm('Delete this user?')) return;
    try {
      await adminAPI.deleteUser(id);
      toast.success('User deleted');
      fetchAll();
    } catch (e) {
      toast.error('Failed to delete user');
    }
  };

  // ===== INTEGRATIONS =====
  const openIntegrationModal = (integration = null) => {
    setEditIntegration(integration);
    setIntegrationForm(integration ? { ...integration } : defaultIntegration);
    setIntegrationModalOpen(true);
  };
  const closeIntegrationModal = () => {
    setIntegrationModalOpen(false);
    setEditIntegration(null);
    setIntegrationForm(defaultIntegration);
  };
  const handleIntegrationFormChange = (e) => {
    const { name, value } = e.target;
    setIntegrationForm((prev) => ({ ...prev, [name]: value }));
  };
  const handleIntegrationSubmit = async (e) => {
    e.preventDefault();
    try {
      if (editIntegration) {
        await adminAPI.updateIntegration(editIntegration._id, integrationForm);
        toast.success('Integration updated');
      } else {
        await adminAPI.createIntegration(integrationForm);
        toast.success('Integration created');
      }
      closeIntegrationModal();
      fetchAll();
    } catch (e) {
      toast.error('Failed to save integration');
    }
  };
  const handleDeleteIntegration = async (id) => {
    if (!window.confirm('Delete this integration?')) return;
    try {
      await adminAPI.deleteIntegration(id);
      toast.success('Integration deleted');
      fetchAll();
    } catch (e) {
      toast.error('Failed to delete integration');
    }
  };
  const handleTestAPI = async (id) => {
    try {
      const res = await adminAPI.testIntegration(id);
      if (res.success) {
    toast.success('API connection test successful!');
      } else {
        toast.error('API connection test failed.');
      }
      fetchAll();
    } catch (e) {
      toast.error('Failed to test integration');
    }
  };

  // ===== SETTINGS (Notifications, Security, System) =====
  const handleNotificationChange = (key) => {
    setNotifications((prev) => ({ ...prev, [key]: !prev[key] }));
  };
  const handleSecurityChange = (key, value) => {
    setSecurity((prev) => ({ ...prev, [key]: value }));
  };
  const handleSystemChange = (key, value) => {
    setSystem((prev) => ({ ...prev, [key]: value }));
  };
  const handleSaveSettings = async () => {
    setSavingSettings(true);
    try {
      await adminAPI.updateSettings({ notifications, security, system });
      toast.success('Settings saved!');
      fetchAll();
    } catch (e) {
      toast.error('Failed to save settings');
    } finally {
      setSavingSettings(false);
    }
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

  const tabs = [
    { id: 'users', name: 'User Management', icon: Users },
    { id: 'integrations', name: 'API Integrations', icon: Globe },
    { id: 'notifications', name: 'Notifications', icon: Bell },
    { id: 'security', name: 'Security', icon: Shield },
    { id: 'system', name: 'System Settings', icon: Settings }
  ];

  // ===== RENDER TABS =====
  const renderUsersTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-900">User Management</h3>
        <button className="btn-primary flex items-center" onClick={() => openUserModal()}>
          <Plus className="h-4 w-4 mr-2" />
          Add User
        </button>
      </div>
      {usersLoading ? (
        <div>Loading users...</div>
      ) : (
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
                        <button className="inline-flex items-center p-1 border border-transparent rounded-full shadow-sm text-white bg-primary-600 hover:bg-primary-700" onClick={() => openUserModal(user)}>
                        <Edit className="h-4 w-4" />
                      </button>
                        <button className="inline-flex items-center p-1 border border-transparent rounded-full shadow-sm text-white bg-danger-600 hover:bg-danger-700" onClick={() => handleDeleteUser(user.id)}>
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
      )}
      {/* User Modal */}
      {userModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-30">
          <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md relative">
            <button className="absolute top-2 right-2 text-gray-400 hover:text-gray-600" onClick={closeUserModal}><X /></button>
            <h3 className="text-lg font-semibold mb-4">{editUser ? 'Edit User' : 'Add User'}</h3>
            <form onSubmit={handleUserSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input name="name" value={userForm.name} onChange={handleUserFormChange} className="input-field w-full" required />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input name="email" value={userForm.email} onChange={handleUserFormChange} className="input-field w-full" type="email" required />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
                <select name="role" value={userForm.role} onChange={handleUserFormChange} className="input-field w-full">
                  <option value="admin">Admin</option>
                  <option value="manager">Manager</option>
                  <option value="supervisor">Supervisor</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
                <select name="status" value={userForm.status} onChange={handleUserFormChange} className="input-field w-full">
                  <option value="active">Active</option>
                  <option value="inactive">Inactive</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Permissions</label>
                <div className="flex flex-wrap gap-2">
                  {['view', 'edit', 'approve', 'admin'].map((perm) => (
                    <label key={perm} className="flex items-center space-x-1">
                      <input
                        type="checkbox"
                        checked={userForm.permissions.includes(perm)}
                        onChange={() => handleUserPermissionChange(perm)}
                        className="h-4 w-4 text-primary-600"
                      />
                      <span className="text-xs">{perm}</span>
                    </label>
                  ))}
                </div>
              </div>
              {/* Password field only for new user */}
              {!editUser && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Password <span className="text-xs text-gray-500">(min 6 characters)</span></label>
                  <input
                    name="password"
                    type="password"
                    value={userForm.password}
                    onChange={handleUserFormChange}
                    className="input-field w-full"
                    minLength={6}
                    required
                  />
                  {passwordError && (
                    <div className="text-red-500 text-xs mt-1">{passwordError}</div>
                  )}
                </div>
              )}
              <div className="flex justify-end">
                <button type="submit" className="btn-primary px-4 py-2" disabled={!editUser && (userForm.password.length < 6)}>{editUser ? 'Update' : 'Create'}</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );

  const renderIntegrationsTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-900">API Integrations</h3>
        <button className="btn-primary flex items-center" onClick={() => openIntegrationModal()}>
          <Plus className="h-4 w-4 mr-2" />
          Add Integration
        </button>
      </div>
      {integrationsLoading ? (
        <div>Loading integrations...</div>
      ) : (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {integrations.map((integration) => (
          <div key={integration._id} className="card">
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
                onClick={() => handleTestAPI(integration._id)}
                className="btn-secondary flex items-center text-sm"
              >
                <TestTube className="h-4 w-4 mr-2" />
                Test Connection
              </button>
                <button className="btn-primary flex items-center text-sm" onClick={() => openIntegrationModal(integration)}>
                <Edit className="h-4 w-4 mr-2" />
                Configure
              </button>
                <button className="btn-danger flex items-center text-sm" onClick={() => handleDeleteIntegration(integration._id)}>
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
      {/* Integration Modal */}
      {integrationModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-30">
          <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md relative">
            <button className="absolute top-2 right-2 text-gray-400 hover:text-gray-600" onClick={closeIntegrationModal}><X /></button>
            <h3 className="text-lg font-semibold mb-4">{editIntegration ? 'Edit Integration' : 'Add Integration'}</h3>
            <form onSubmit={handleIntegrationSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                <input name="name" value={integrationForm.name} onChange={handleIntegrationFormChange} className="input-field w-full" required />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Provider</label>
                <input name="provider" value={integrationForm.provider} onChange={handleIntegrationFormChange} className="input-field w-full" required />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">API Key</label>
                <input name="apiKey" value={integrationForm.apiKey} onChange={handleIntegrationFormChange} className="input-field w-full" required />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Endpoint</label>
                <input name="endpoint" value={integrationForm.endpoint} onChange={handleIntegrationFormChange} className="input-field w-full" required />
              </div>
              <div className="flex justify-end">
                <button type="submit" className="btn-primary px-4 py-2">{editIntegration ? 'Update' : 'Create'}</button>
              </div>
            </form>
          </div>
      </div>
      )}
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
        <button onClick={handleSaveSettings} className="btn-primary flex items-center" disabled={savingSettings}>
          <Save className="h-4 w-4 mr-2" />
          {savingSettings ? 'Saving...' : 'Save Settings'}
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
              <input type="number" className="input-field" value={security.sessionTimeout || ''} onChange={e => handleSecurityChange('sessionTimeout', Number(e.target.value))} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Maximum Login Attempts
              </label>
              <input type="number" className="input-field" value={security.maxLoginAttempts || ''} onChange={e => handleSecurityChange('maxLoginAttempts', Number(e.target.value))} />
            </div>
            <div className="flex items-center">
              <input type="checkbox" className="h-4 w-4 text-primary-600" checked={!!security.requireTwoFactor} onChange={e => handleSecurityChange('requireTwoFactor', e.target.checked)} />
              <label className="ml-2 text-sm text-gray-900">Require Two-Factor Authentication</label>
            </div>
          </div>
        </div>
        <div className="card">
          <h4 className="font-medium text-gray-900 mb-4">Data Protection</h4>
          <div className="space-y-4">
            <div className="flex items-center">
              <input type="checkbox" className="h-4 w-4 text-primary-600" checked={!!security.dataEncryption} onChange={e => handleSecurityChange('dataEncryption', e.target.checked)} />
              <label className="ml-2 text-sm text-gray-900">Enable Data Encryption</label>
            </div>
            <div className="flex items-center">
              <input type="checkbox" className="h-4 w-4 text-primary-600" checked={!!security.automaticBackup} onChange={e => handleSecurityChange('automaticBackup', e.target.checked)} />
              <label className="ml-2 text-sm text-gray-900">Automatic Backup</label>
            </div>
            <div className="flex items-center">
              <input type="checkbox" className="h-4 w-4 text-primary-600" checked={!!security.auditLogging} onChange={e => handleSecurityChange('auditLogging', e.target.checked)} />
              <label className="ml-2 text-sm text-gray-900">Audit Logging</label>
            </div>
          </div>
        </div>
      </div>
      <div className="flex justify-end">
        <button onClick={handleSaveSettings} className="btn-primary flex items-center" disabled={savingSettings}>
          <Save className="h-4 w-4 mr-2" />
          {savingSettings ? 'Saving...' : 'Save Settings'}
        </button>
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
              <input type="text" className="input-field" value={system.companyName || ''} onChange={e => handleSystemChange('companyName', e.target.value)} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Time Zone
              </label>
              <select className="input-field" value={system.timezone || ''} onChange={e => handleSystemChange('timezone', e.target.value)}>
                <option>UTC-5 (Eastern Time)</option>
                <option>UTC-8 (Pacific Time)</option>
                <option>UTC+0 (GMT)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Currency
              </label>
              <select className="input-field" value={system.currency || ''} onChange={e => handleSystemChange('currency', e.target.value)}>
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
              <input type="number" className="input-field" value={system.dataRefreshInterval || ''} onChange={e => handleSystemChange('dataRefreshInterval', Number(e.target.value))} />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Cache Duration (hours)
              </label>
              <input type="number" className="input-field" value={system.cacheDuration || ''} onChange={e => handleSystemChange('cacheDuration', Number(e.target.value))} />
            </div>
            <div className="flex items-center">
              <input type="checkbox" className="h-4 w-4 text-primary-600" checked={!!system.performanceMonitoring} onChange={e => handleSystemChange('performanceMonitoring', e.target.checked)} />
              <label className="ml-2 text-sm text-gray-900">Enable Performance Monitoring</label>
            </div>
          </div>
        </div>
      </div>
      <div className="flex justify-end">
        <button onClick={handleSaveSettings} className="btn-primary flex items-center" disabled={savingSettings}>
          <Save className="h-4 w-4 mr-2" />
          {savingSettings ? 'Saving...' : 'Save Settings'}
        </button>
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