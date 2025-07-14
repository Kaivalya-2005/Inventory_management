module.exports = {
  port: process.env.PORT || 5000,
  jwtSecret: process.env.JWT_SECRET || 'your-super-secret-jwt-key-change-this-in-production',
  environment: process.env.NODE_ENV || 'development',
  cors: {
    origin: process.env.NODE_ENV === 'production' 
      ? ['https://yourdomain.com'] 
      : ['http://localhost:3000'],
    credentials: true
  },
  database: {
    uri: process.env.MONGODB_URI || 'mongodb://localhost:27017/inventory-management'
  }
}; 