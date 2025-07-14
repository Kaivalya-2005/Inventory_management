# Inventory Management System - Backend

This is the backend server for the AI-powered inventory management system, built with Node.js and Express.

## 🏗️ Project Structure

```
server/
├── server.js              # Main server file
├── package.json           # Server dependencies
├── config/
│   └── config.js          # Configuration settings
├── routes/
│   ├── auth.js            # Authentication routes
│   ├── dashboard.js       # Dashboard routes
│   ├── stock.js           # Stock management routes
│   ├── orders.js          # Order management routes
│   └── analytics.js       # Analytics and insights routes
├── middleware/
│   └── auth.js            # JWT authentication middleware
└── data/
    └── mockData.js        # Mock data for development
```

## 🚀 Getting Started

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn

### Installation
1. Navigate to the server directory:
   ```bash
   cd server
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env` file in the server directory:
   ```env
   PORT=5000
   JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
   NODE_ENV=development
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

## 📡 API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/demo` - Demo login
- `GET /api/auth/profile` - Get user profile

### Dashboard
- `GET /api/dashboard/metrics` - Get dashboard metrics

### Stock Management
- `GET /api/stock/suggestions` - Get stock suggestions
- `POST /api/stock/suggestions/:id/accept` - Accept suggestion
- `POST /api/stock/suggestions/:id/reject` - Reject suggestion
- `GET /api/stock/details` - Get stock details
- `GET /api/stock/details/:id` - Get specific stock detail

### Order Management
- `GET /api/orders` - Get all orders
- `POST /api/orders` - Create new order
- `PUT /api/orders/:id/status` - Update order status

### Analytics
- `GET /api/analytics/historical-data` - Get historical data
- `GET /api/analytics/model-factors` - Get model factors
- `PUT /api/analytics/model-factors/temperature` - Update temperature
- `GET /api/analytics/model-insights` - Get model insights

## 🔐 Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## 📊 Mock Data

The server includes comprehensive mock data for:
- Dashboard metrics and alerts
- Stock suggestions with confidence levels
- Historical data and accuracy trends
- Model factors and insights
- Order management

## 🛠️ Development

### Running in Development Mode
```bash
npm run dev
```

### Running in Production Mode
```bash
NODE_ENV=production npm start
```

### Testing
```bash
npm test
```

## 🔧 Configuration

Configuration is managed through the `config/config.js` file and environment variables:

- `PORT` - Server port (default: 5000)
- `JWT_SECRET` - Secret key for JWT tokens
- `NODE_ENV` - Environment (development/production)
- `MONGODB_URI` - Database connection string

## 📝 Environment Variables

Create a `.env` file in the server directory:

```env
PORT=5000
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
NODE_ENV=development
MONGODB_URI=mongodb://localhost:27017/inventory-management
```

## 🔒 Security Features

- JWT-based authentication
- CORS configuration
- Input validation
- Error handling middleware
- Secure password hashing with bcrypt

## 📈 Future Enhancements

- Database integration (MongoDB/PostgreSQL)
- Real-time notifications (WebSocket)
- File upload for CSV data
- Advanced analytics and ML integration
- Rate limiting and API throttling
- Comprehensive logging and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. 