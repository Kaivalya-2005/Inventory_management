const express = require('express');
const axios = require('axios');
const router = express.Router();

const OPENWEATHER_API_KEY = process.env.OPENWEATHER_API_KEY;

// GET /api/weather?city=CityName
router.get('/weather', async (req, res) => {
  const city = req.query.city || 'New York';
  if (!OPENWEATHER_API_KEY) {
    return res.status(500).json({ message: 'OpenWeatherMap API key not set in environment variables.' });
  }
  try {
    const response = await axios.get(
      `https://api.openweathermap.org/data/2.5/weather`, {
        params: {
          q: city,
          appid: OPENWEATHER_API_KEY,
          units: 'metric'
        }
      }
    );
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ message: 'Failed to fetch weather data', error: error.message });
  }
});

module.exports = router; 