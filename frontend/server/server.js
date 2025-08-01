const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

// Import database connection
const connectDB = require('./config/database');

// Import routes and middleware
const authRoutes = require('./routes/auth');
const dashboardRoutes = require('./routes/dashboard');
const stockRoutes = require('./routes/stock');
const ordersRoutes = require('./routes/orders');
const analyticsRoutes = require('./routes/analytics');
const adminRoutes = require('./routes/admin');
const weatherRoutes = require('./routes/weather');
const authenticateToken = require('./middleware/auth');

const app = express();
const PORT = process.env.PORT || 5000;

// Connect to MongoDB
connectDB();

// Middleware
app.use(cors());
app.use(express.json());

// Only serve static files in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../client/build')));
}

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/stock', stockRoutes);
app.use('/api/orders', ordersRoutes);
app.use('/api/analytics', analyticsRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api', weatherRoutes);

// Initialize default data
const initializeDatabase = async () => {
  try {
    const User = require('./models/User');
    const Setting = require('./models/Setting');
    const Integration = require('./models/Integration');

    // Create default admin user if none exists
    const adminExists = await User.findOne({ email: 'admin@walmart.com' });
    if (!adminExists) {
      const adminUser = new User({
        name: 'System Administrator',
        email: 'admin@walmart.com',
        password: 'admin123',
        role: 'admin',
        status: 'active',
        permissions: ['view', 'edit', 'approve', 'admin']
      });
      await adminUser.save();
      console.log('Default admin user created');
    }

    // Initialize default settings if none exist
    const settingsExist = await Setting.findOne({});
    if (!settingsExist) {
      const defaultSettings = new Setting({
        category: 'system',
        notifications: {
          email: true,
          sms: false,
          push: true,
          lowStock: true,
          highDemand: true,
          weatherAlerts: true,
          orderUpdates: false
        },
        security: {
          sessionTimeout: 30,
          maxLoginAttempts: 5,
          requireTwoFactor: true,
          dataEncryption: true,
          automaticBackup: true,
          auditLogging: false
        },
        system: {
          companyName: 'Inventory Management Corp',
          timezone: 'UTC-5 (Eastern Time)',
          currency: 'USD ($)',
          dataRefreshInterval: 15,
          cacheDuration: 24,
          performanceMonitoring: true
        }
      });
      await defaultSettings.save();
      console.log('Default settings initialized');
    }

    // Create sample integrations if none exist
    const integrationsExist = await Integration.findOne({});
    if (!integrationsExist) {
      const sampleIntegrations = [
        {
          name: 'Weather API',
          provider: 'OpenWeatherMap',
          apiKey: 'sample_key_123',
          endpoint: 'https://api.openweathermap.org/data/2.5/weather',
          status: 'disconnected'
        },
        {
          name: 'ERP System',
          provider: 'SAP Business One',
          apiKey: 'sample_key_456',
          endpoint: 'https://erp.company.com/api/v1',
          status: 'disconnected'
        }
      ];

      for (const integration of sampleIntegrations) {
        const newIntegration = new Integration(integration);
        await newIntegration.save();
      }
      console.log('Sample integrations created');
    }

    console.log('Database initialization completed');
  } catch (error) {
    console.error('Database initialization error:', error);
  }
};

// Initialize database after connection
setTimeout(initializeDatabase, 2000);

// Serve React app for all other routes (only in production)
if (process.env.NODE_ENV === 'production') {
  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../client/build', 'index.html'));
  });
}

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ message: 'Something went wrong!' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`MongoDB URI: ${process.env.MONGODB_URI ? 'Configured' : 'Not configured'}`);
}); 