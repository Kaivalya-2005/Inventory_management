const mongoose = require('mongoose');

const integrationSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true
  },
  provider: {
    type: String,
    required: true,
    trim: true
  },
  apiKey: {
    type: String,
    required: true,
    trim: true
  },
  endpoint: {
    type: String,
    required: true,
    trim: true
  },
  status: {
    type: String,
    enum: ['connected', 'disconnected', 'testing'],
    default: 'disconnected'
  },
  lastSync: {
    type: Date,
    default: null
  },
  lastTestResult: {
    success: Boolean,
    message: String,
    testedAt: Date
  },
  config: {
    type: Map,
    of: String,
    default: {}
  }
}, {
  timestamps: true
});

// Method to mask API key for display
integrationSchema.methods.getMaskedApiKey = function() {
  if (!this.apiKey) return '';
  return `***${this.apiKey.slice(-4)}`;
};

// Method to test the integration
integrationSchema.methods.testConnection = async function() {
  try {
    const axios = require('axios');
    
    // Make a test request to the endpoint
    const response = await axios.get(this.endpoint, {
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      timeout: 10000 // 10 second timeout
    });
    
    this.status = 'connected';
    this.lastSync = new Date();
    this.lastTestResult = {
      success: true,
      message: 'Connection successful',
      testedAt: new Date()
    };
    
    await this.save();
    return { success: true, message: 'Connection test successful!' };
    
  } catch (error) {
    this.status = 'disconnected';
    this.lastTestResult = {
      success: false,
      message: error.message || 'Connection test failed',
      testedAt: new Date()
    };
    
    await this.save();
    return { 
      success: false, 
      message: `Connection test failed: ${error.message}` 
    };
  }
};

module.exports = mongoose.model('Integration', integrationSchema); 