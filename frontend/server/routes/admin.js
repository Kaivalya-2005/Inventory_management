const express = require('express');
const router = express.Router();
const authenticateToken = require('../middleware/auth');
const User = require('../models/User');
const Integration = require('../models/Integration');
const Setting = require('../models/Setting');

// ===== USER MANAGEMENT =====

// Get all users
router.get('/users', authenticateToken, async (req, res) => {
  try {
    const users = await User.find({}).sort({ createdAt: -1 });
    res.json({ users });
  } catch (error) {
    console.error('Error fetching users:', error);
    res.status(500).json({ message: 'Failed to fetch users' });
  }
});

// Create new user
router.post('/users', authenticateToken, async (req, res) => {
  try {
    const { name, email, role, permissions, password } = req.body;
    
    if (!name || !email || !role || !password) {
      return res.status(400).json({ message: 'Name, email, role, and password are required' });
    }

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: 'User with this email already exists' });
    }

    const newUser = new User({
      name,
      email,
      password,
      role,
      status: 'active',
      permissions: permissions || ['view']
    });

    await newUser.save();
    res.status(201).json({ user: newUser.toJSON(), message: 'User created successfully' });
  } catch (error) {
    console.error('Error creating user:', error);
    res.status(500).json({ message: 'Failed to create user' });
  }
});

// Update user
router.put('/users/:id', authenticateToken, async (req, res) => {
  try {
    const userId = req.params.id;
    const { name, email, role, status, permissions } = req.body;
    
    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Update fields
    if (name) user.name = name;
    if (email) user.email = email;
    if (role) user.role = role;
    if (status) user.status = status;
    if (permissions) user.permissions = permissions;

    await user.save();
    res.json({ user: user.toJSON(), message: 'User updated successfully' });
  } catch (error) {
    console.error('Error updating user:', error);
    res.status(500).json({ message: 'Failed to update user' });
  }
});

// Delete user
router.delete('/users/:id', authenticateToken, async (req, res) => {
  try {
    const userId = req.params.id;
    const user = await User.findById(userId);
    
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    await User.findByIdAndDelete(userId);
    res.json({ message: 'User deleted successfully' });
  } catch (error) {
    console.error('Error deleting user:', error);
    res.status(500).json({ message: 'Failed to delete user' });
  }
});

// ===== SYSTEM SETTINGS =====

// Get all settings
router.get('/settings', authenticateToken, async (req, res) => {
  try {
    const settings = await Setting.getAllSettings();
    res.json({ settings });
  } catch (error) {
    console.error('Error fetching settings:', error);
    res.status(500).json({ message: 'Failed to fetch settings' });
  }
});

// Update settings
router.put('/settings', authenticateToken, async (req, res) => {
  try {
    const { notifications, security, system } = req.body;
    const updatedSettings = await Setting.updateSettings({ notifications, security, system });
    
    res.json({ settings: updatedSettings, message: 'Settings updated successfully' });
  } catch (error) {
    console.error('Error updating settings:', error);
    res.status(500).json({ message: 'Failed to update settings' });
  }
});

// ===== API INTEGRATIONS =====

// Get all integrations
router.get('/integrations', authenticateToken, async (req, res) => {
  try {
    const integrations = await Integration.find({}).sort({ createdAt: -1 });
    
    // Transform integrations to include masked API keys
    const transformedIntegrations = integrations.map(integration => ({
      ...integration.toJSON(),
      apiKey: integration.getMaskedApiKey()
    }));
    
    res.json({ integrations: transformedIntegrations });
  } catch (error) {
    console.error('Error fetching integrations:', error);
    res.status(500).json({ message: 'Failed to fetch integrations' });
  }
});

// Create new integration
router.post('/integrations', authenticateToken, async (req, res) => {
  try {
    const { name, provider, apiKey, endpoint } = req.body;
    
    if (!name || !provider || !apiKey || !endpoint) {
      return res.status(400).json({ message: 'Name, provider, API key, and endpoint are required' });
    }

    const newIntegration = new Integration({
      name,
      provider,
      apiKey,
      endpoint,
      status: 'disconnected'
    });

    await newIntegration.save();
    
    // Return with masked API key
    const responseIntegration = {
      ...newIntegration.toJSON(),
      apiKey: newIntegration.getMaskedApiKey()
    };
    
    res.status(201).json({ integration: responseIntegration, message: 'Integration created successfully' });
  } catch (error) {
    console.error('Error creating integration:', error);
    res.status(500).json({ message: 'Failed to create integration' });
  }
});

// Update integration
router.put('/integrations/:id', authenticateToken, async (req, res) => {
  try {
    const integrationId = req.params.id;
    const { name, provider, apiKey, endpoint, status } = req.body;
    
    const integration = await Integration.findById(integrationId);
    if (!integration) {
      return res.status(404).json({ message: 'Integration not found' });
    }

    // Update fields
    if (name) integration.name = name;
    if (provider) integration.provider = provider;
    if (apiKey) integration.apiKey = apiKey;
    if (endpoint) integration.endpoint = endpoint;
    if (status) integration.status = status;

    await integration.save();
    
    // Return with masked API key
    const responseIntegration = {
      ...integration.toJSON(),
      apiKey: integration.getMaskedApiKey()
    };
    
    res.json({ integration: responseIntegration, message: 'Integration updated successfully' });
  } catch (error) {
    console.error('Error updating integration:', error);
    res.status(500).json({ message: 'Failed to update integration' });
  }
});

// Delete integration
router.delete('/integrations/:id', authenticateToken, async (req, res) => {
  try {
    const integrationId = req.params.id;
    const integration = await Integration.findById(integrationId);
    
    if (!integration) {
      return res.status(404).json({ message: 'Integration not found' });
    }

    await Integration.findByIdAndDelete(integrationId);
    res.json({ message: 'Integration deleted successfully' });
  } catch (error) {
    console.error('Error deleting integration:', error);
    res.status(500).json({ message: 'Failed to delete integration' });
  }
});

// Test API connection
router.post('/integrations/:id/test', authenticateToken, async (req, res) => {
  try {
    const integrationId = req.params.id;
    const integration = await Integration.findById(integrationId);
    
    if (!integration) {
      return res.status(404).json({ message: 'Integration not found' });
    }

    // Test the connection using the integration's method
    const result = await integration.testConnection();
    
    res.json(result);
  } catch (error) {
    console.error('Error testing integration:', error);
    res.status(500).json({ message: 'Failed to test integration' });
  }
});

module.exports = router; 