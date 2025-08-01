const express = require('express');
const router = express.Router();
const authenticateToken = require('../middleware/auth');
const Order = require('../models/Order');

// Get all orders
router.get('/', authenticateToken, async (req, res) => {
  try {
    const orders = await Order.find().sort({ orderDate: -1 });
    res.json({ orders });
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch orders' });
  }
});

// Get a single order
router.get('/:id', authenticateToken, async (req, res) => {
  try {
    const order = await Order.findById(req.params.id);
    if (!order) return res.status(404).json({ message: 'Order not found' });
    res.json({ order });
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch order' });
  }
});

// Create a new order
router.post('/', authenticateToken, async (req, res) => {
  try {
    const orderData = req.body;
    const order = new Order(orderData);
    await order.save();
    res.status(201).json({ order });
  } catch (error) {
    res.status(400).json({ message: 'Failed to create order', error: error.message });
  }
});

// Update an order
router.put('/:id', authenticateToken, async (req, res) => {
  try {
    const order = await Order.findByIdAndUpdate(req.params.id, req.body, { new: true });
    if (!order) return res.status(404).json({ message: 'Order not found' });
    res.json({ order });
  } catch (error) {
    res.status(400).json({ message: 'Failed to update order', error: error.message });
  }
});

// Delete an order
router.delete('/:id', authenticateToken, async (req, res) => {
  try {
    const order = await Order.findByIdAndDelete(req.params.id);
    if (!order) return res.status(404).json({ message: 'Order not found' });
    res.json({ message: 'Order deleted' });
  } catch (error) {
    res.status(400).json({ message: 'Failed to delete order', error: error.message });
  }
});

// Approve an order
router.post('/:id/approve', authenticateToken, async (req, res) => {
  try {
    const order = await Order.findById(req.params.id);
    if (!order) return res.status(404).json({ message: 'Order not found' });
    order.status = 'approved';
    order.approvedBy = req.user.email;
    await order.save();
    res.json({ order });
  } catch (error) {
    res.status(400).json({ message: 'Failed to approve order', error: error.message });
  }
});

// Reject an order
router.post('/:id/reject', authenticateToken, async (req, res) => {
  try {
    const order = await Order.findById(req.params.id);
    if (!order) return res.status(404).json({ message: 'Order not found' });
    order.status = 'rejected';
    order.approvedBy = req.user.email;
    await order.save();
    res.json({ order });
  } catch (error) {
    res.status(400).json({ message: 'Failed to reject order', error: error.message });
  }
});

module.exports = router; 