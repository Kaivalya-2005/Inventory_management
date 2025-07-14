const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

// Import routes and middleware
const authRoutes = require('./routes/auth');
const dashboardRoutes = require('./routes/dashboard');
const stockRoutes = require('./routes/stock');
const ordersRoutes = require('./routes/orders');
const analyticsRoutes = require('./routes/analytics');
const authenticateToken = require('./middleware/auth');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Only serve static files in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../client/build')));
}

// Routes

// Authentication routes
app.use('/api/auth', authRoutes);

// Dashboard routes
app.use('/api/dashboard', dashboardRoutes);

// Stock routes
app.use('/api/stock', stockRoutes);

// Orders routes
app.use('/api/orders', ordersRoutes);

// Analytics routes
app.use('/api/analytics', analyticsRoutes);

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
}); 