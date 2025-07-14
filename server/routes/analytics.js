const express = require('express');
const router = express.Router();
const authenticateToken = require('../middleware/auth');
const mockData = require('../data/mockData');

// Get historical data
router.get('/historical-data', authenticateToken, (req, res) => {
  res.json(mockData.historicalData);
});

// Get model factors
router.get('/model-factors', authenticateToken, (req, res) => {
  res.json(mockData.modelFactors);
});

// Update temperature factor
router.put('/model-factors/temperature', authenticateToken, (req, res) => {
  const { temperature } = req.body;
  // In a real app, this would update the database
  res.json({ message: 'Temperature updated', temperature });
});

// Get model insights
router.get('/model-insights', authenticateToken, (req, res) => {
  res.json(mockData.modelInsights);
});

module.exports = router; 