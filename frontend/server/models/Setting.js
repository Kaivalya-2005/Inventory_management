const mongoose = require('mongoose');

const settingSchema = new mongoose.Schema({
  category: {
    type: String,
    required: true,
    enum: ['notifications', 'security', 'system'],
    default: 'system'
  },
  notifications: {
    email: { type: Boolean, default: true },
    sms: { type: Boolean, default: false },
    push: { type: Boolean, default: true },
    lowStock: { type: Boolean, default: true },
    highDemand: { type: Boolean, default: true },
    weatherAlerts: { type: Boolean, default: true },
    orderUpdates: { type: Boolean, default: false }
  },
  security: {
    sessionTimeout: { type: Number, default: 30, min: 5, max: 480 },
    maxLoginAttempts: { type: Number, default: 5, min: 1, max: 20 },
    requireTwoFactor: { type: Boolean, default: true },
    dataEncryption: { type: Boolean, default: true },
    automaticBackup: { type: Boolean, default: true },
    auditLogging: { type: Boolean, default: false }
  },
  system: {
    companyName: { type: String, default: 'Inventory Management Corp' },
    timezone: { type: String, default: 'UTC-5 (Eastern Time)' },
    currency: { type: String, default: 'USD ($)' },
    dataRefreshInterval: { type: Number, default: 15, min: 1, max: 60 },
    cacheDuration: { type: Number, default: 24, min: 1, max: 168 },
    performanceMonitoring: { type: Boolean, default: true }
  }
}, {
  timestamps: true
});

// Ensure only one settings document exists
settingSchema.index({ category: 1 }, { unique: true });

// Method to get all settings as a single object
settingSchema.statics.getAllSettings = async function() {
  const settings = await this.find({});
  const result = {
    notifications: {},
    security: {},
    system: {}
  };
  
  settings.forEach(setting => {
    if (setting.notifications) result.notifications = setting.notifications;
    if (setting.security) result.security = setting.security;
    if (setting.system) result.system = setting.system;
  });
  
  return result;
};

// Method to update settings
settingSchema.statics.updateSettings = async function(updates) {
  const { notifications, security, system } = updates;
  
  if (notifications) {
    await this.findOneAndUpdate(
      { category: 'notifications' },
      { notifications },
      { upsert: true, new: true }
    );
  }
  
  if (security) {
    await this.findOneAndUpdate(
      { category: 'security' },
      { security },
      { upsert: true, new: true }
    );
  }
  
  if (system) {
    await this.findOneAndUpdate(
      { category: 'system' },
      { system },
      { upsert: true, new: true }
    );
  }
  
  return this.getAllSettings();
};

module.exports = mongoose.model('Setting', settingSchema); 