const express = require('express');
const router = express.Router();
const authenticateToken = require('../middleware/auth');
const mockData = require('../data/mockData');

// Get all orders
router.get('/', authenticateToken, (req, res) => {
  res.json(mockData.orders);
});

// Create new order
router.post('/', authenticateToken, (req, res) => {
  const newOrder = {
    id: mockData.orders.length + 1,
    ...req.body,
    orderDate: new Date().toISOString().split('T')[0],
    status: 'pending'
  };
  // In a real app, this would save to database
  res.json(newOrder);
});

// Update order status
router.put('/:id/status', authenticateToken, (req, res) => {
  const { id } = req.params;
  const { status } = req.body;
  // In a real app, this would update the database
  res.json({ message: 'Order status updated', id, status });
});

module.exports = router; 