// routes/weather.js
const express = require('express');
const axios = require('axios');
const router = express.Router();

const OPENWEATHER_API_KEY = process.env.OPENWEATHER_API_KEY;

// Example: Get current weather for a city
router.get('/weather', async (req, res) => {
  const { city = 'New York' } = req.query;
  try {
    const response = await axios.get(
      `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&appid=${OPENWEATHER_API_KEY}&units=metric`
    );
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch weather data', error: error.message });
  }
});

module.exports = router;