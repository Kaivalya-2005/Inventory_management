# Inventory Management System

An AI-powered inventory management system for stock prediction and management, designed for retail operations like Walmart. The system uses machine learning to predict stock requirements based on factors like temperature, demand patterns, and special events.

## 🌟 Features

### 📊 Dashboard
- Real-time overview of inventory metrics
- Interactive charts showing demand predictions
- Temperature trends and seasonal alerts
- Quick action buttons for common tasks

### 🛒 Stock Suggestions
- AI-powered recommendations for stock replenishment
- Confidence levels and influencing factors
- Filter and search by category, urgency, and warehouse
- Accept/Reject/Modify actions for each suggestion

### 📁 Historical Data
- Comprehensive analysis of past predictions vs actual orders
- Model accuracy tracking and error analysis
- Export functionality for data analysis
- Temperature correlation charts

### 🔍 Stock Details
- Detailed per-product information with SKU tracking
- Current stock levels, safety stock, and reorder points
- Supplier information and contact details
- Demand trend visualization

### ⚙️ Model Input Factors
- Real-time weather data integration
- Manual temperature override capabilities
- Special events and situations management
- CSV data import functionality
- API integration status monitoring

### 🧠 Model Insights
- Feature importance visualization
- Decision factor breakdown by product
- Model performance metrics
- Recent AI decisions with explanations
- Model transparency information

### 📦 Order Management
- Auto-generated and manual purchase orders
- Order status tracking (pending, approved, placed, delivered)
- Supplier management and delivery scheduling
- Order notes and comments system

### 👩‍💻 Admin Settings
- User management with role-based access
- API integration configuration
- Notification settings (email, SMS, push)
- Security and system configuration

## 🚀 Getting Started

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd inventory-management-system
   ```

2. **Install dependencies**
   ```bash
   # Install backend dependencies
   npm install

   # Install frontend dependencies
   cd client
   npm install
   cd ..
   ```

3. **Start the development server**
   ```bash
   # Start both frontend and backend
   npm run dev

   # Or start them separately:
   npm run server    # Backend only
   npm run client    # Frontend only
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend: http://localhost:5000

### Demo Login
For demonstration purposes, you can use any email and password to login to the system.

## 🏗️ Project Structure

```
inventory-management-system/
├── client/                 # React frontend
│   ├── public/            # Static files
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── contexts/      # React contexts (Auth)
│   │   ├── pages/         # Page components
│   │   ├── App.js         # Main app component
│   │   └── index.js       # Entry point
│   └── package.json
├── server/                # Express backend (to be implemented)
├── package.json           # Root package.json
└── README.md
```

## 🎨 UI/UX Features

### Modern Design
- Clean, professional interface using Tailwind CSS
- Responsive design for desktop and mobile
- Intuitive navigation with sidebar menu
- Consistent color scheme and typography

### Interactive Elements
- Real-time charts and graphs using Recharts
- Toast notifications for user feedback
- Loading states and error handling
- Hover effects and smooth transitions

### Data Visualization
- Bar charts for demand predictions
- Pie charts for stock level distribution
- Line charts for temperature trends
- Progress bars for stock levels

## 🔧 Technology Stack

### Frontend
- **React 18** - UI framework
- **React Router** - Navigation
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **Lucide React** - Icons
- **React Hot Toast** - Notifications
- **React Query** - Data fetching

### Backend (To be implemented)
- **Express.js** - Server framework
- **MongoDB** - Database
- **Mongoose** - ODM
- **JWT** - Authentication
- **bcryptjs** - Password hashing

## 📱 Mobile Responsiveness

The application is fully responsive and includes:
- Mobile-first design approach
- Collapsible sidebar for mobile devices
- Touch-friendly interface elements
- Optimized layouts for different screen sizes

## 🔔 Notification System

- Real-time alerts for low stock items
- Weather-based demand predictions
- Order status updates
- Configurable notification preferences

## 🌍 Multi-location Support

The system is designed to support multiple warehouses/stores with:
- Location switching capabilities
- Warehouse-specific data views
- Centralized management interface

## 🚀 Future Enhancements

- **Backend API Implementation** - Complete the Express.js backend
- **Database Integration** - MongoDB with Mongoose
- **Real-time Updates** - WebSocket integration
- **Advanced Analytics** - More sophisticated ML models
- **Mobile App** - React Native or PWA
- **API Integrations** - Real weather APIs, ERP systems
- **Reporting** - Advanced reporting and analytics
- **Multi-language Support** - Internationalization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

---

**Note**: This is a demonstration project for a Walmart Hackathon or similar setting. The AI predictions and data are simulated for showcase purposes. 