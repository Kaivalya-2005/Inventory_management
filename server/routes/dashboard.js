const express = require('express');
const router = express.Router();
const authenticateToken = require('../middleware/auth');
const mockData = require('../data/mockData');

// Get dashboard metrics
router.get('/metrics', authenticateToken, (req, res) => {
  res.json(mockData.dashboard);
});

module.exports = router; 