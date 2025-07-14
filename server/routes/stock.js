const express = require('express');
const router = express.Router();
const authenticateToken = require('../middleware/auth');
const mockData = require('../data/mockData');

// Get stock suggestions
router.get('/suggestions', authenticateToken, (req, res) => {
  res.json(mockData.stockSuggestions);
});

// Accept stock suggestion
router.post('/suggestions/:id/accept', authenticateToken, (req, res) => {
  const { id } = req.params;
  // In a real app, this would update the database
  res.json({ message: 'Suggestion accepted', id });
});

// Reject stock suggestion
router.post('/suggestions/:id/reject', authenticateToken, (req, res) => {
  const { id } = req.params;
  // In a real app, this would update the database
  res.json({ message: 'Suggestion rejected', id });
});

// Get stock details
router.get('/details', authenticateToken, (req, res) => {
  res.json(mockData.stockDetails);
});

// Get specific stock detail
router.get('/details/:id', authenticateToken, (req, res) => {
  const { id } = req.params;
  const stock = mockData.stockDetails.find(s => s.id === parseInt(id));
  if (!stock) {
    return res.status(404).json({ message: 'Stock not found' });
  }
  res.json(stock);
});

module.exports = router; 