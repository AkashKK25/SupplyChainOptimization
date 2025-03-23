# Supply Chain Optimization for Small Businesses: Technical Report

## 1. Introduction

Small businesses face unique challenges in managing their inventory efficiently. Unlike large enterprises, they often lack sophisticated inventory management systems and dedicated supply chain analysts. This leads to common problems such as:

- Excess inventory tying up working capital
- Stockouts resulting in lost sales and customer dissatisfaction
- Inefficient ordering patterns increasing operational costs
- Inability to forecast demand accurately

This project aims to provide small businesses with a data-driven solution to optimize their inventory management. By applying time-series forecasting and inventory optimization models, created a system that helps small businesses minimize costs while maintaining appropriate stock levels.

## 2. Methodology

### 2.1 Data Generation

Since this is a portfolio project, generated synthetic data that mimics real-world supply chain operations. The data includes:

- Product information (10 products across 4 categories)
- 2 years of daily sales data with realistic patterns
- Inventory events (levels, reorders, deliveries, stockouts)
- Cost components (holding costs, stockout costs, reorder costs)

The generated data incorporates several realistic business behaviors:
- Seasonal demand patterns
- Product-specific trends
- Random demand fluctuations
- Variable lead times
- Different cost structures by product category

### 2.2 Exploratory Data Analysis

Conducted a comprehensive analysis of the data to understand patterns and relationships:

- Time series analysis of demand by product and category
- Inventory level fluctuations and stockout events
- Lead time variability analysis
- Cost breakdown and relationship with inventory metrics

Key insights from EDA:
- Clear seasonal patterns in demand for certain products
- Correlation between inventory turnover and total costs
- Significant costs associated with both stockouts and excess inventory
- Variability in lead times affecting optimal safety stock levels

### 2.3 Demand Forecasting

Implemented and compared four forecasting models:

1. **Moving Average**: A simple baseline model
   - Advantages: Easy to implement and understand
   - Disadvantages: Cannot capture trends or seasonality

2. **Holt-Winters Exponential Smoothing**:
   - Advantages: Captures both trend and seasonality
   - Disadvantages: Sensitive to parameter selection

3. **SARIMA (Seasonal ARIMA)**:
   - Advantages: Strong statistical foundation
   - Disadvantages: Requires stationary data, complex parameter tuning

4. **Random Forest with Feature Engineering**:
   - Advantages: Captures non-linear relationships, handles multiple features
   - Disadvantages: More complex, risk of overfitting

For each model, we:
- Split data into training and test sets
- Optimized model parameters
- Evaluated performance using RMSE, MAE, and R²
- Generated 30-day forecasts for all products

### 2.4 Inventory Optimization

Implemented several classical inventory models:

1. **Economic Order Quantity (EOQ)**:
   - Calculates the optimal order size that minimizes total costs
   - Formula: EOQ = √(2DS/H) where D = annual demand, S = setup cost, H = holding cost

2. **Reorder Point (ROP)**:
   - Determines when to place a new order based on lead time and demand
   - Formula: ROP = L×D + Z×σL where L = lead time, D = daily demand, Z = safety factor, σL = standard deviation during lead time

3. **Safety Stock Calculation**:
   - Additional inventory to prevent stockouts due to variability
   - Formula: SS = Z×σL where Z = safety factor, σL = standard deviation during lead time

4. **Service Level Analysis**:
   - Calculated optimal parameters at different service levels (80%, 90%, 95%, 98%)
   - Analyzed the cost trade-offs of higher service levels

Produced actionable recommendations:
- Optimal order quantities for each product
- Reorder points based on forecast demand
- Safety stock levels to maintain desired service levels
- Reorder schedule with specific dates and quantities
- Estimated cost savings from optimization

### 2.5 Interactive Dashboard

Created a Dash web application with four main components:

1. **Overview**: Summary statistics and key performance indicators
2. **Demand Forecasting**: Historical vs. forecast demand and seasonal patterns
3. **Inventory Optimization**: Optimal parameters and cost comparisons
4. **Reorder Planning**: Reorder schedule and inventory projections

The dashboard provides:
- Visual representation of complex data
- Interactive filters for different products and service levels
- Actionable insights for inventory management
- Clear recommendations for reordering

## 3. Results and Insights

### 3.1 Forecasting Performance

When comparing the forecasting models, found that:
- Random Forest with feature engineering provided the best overall performance
- SARIMA performed well for products with strong seasonal patterns
- Holt-Winters was a good compromise between accuracy and complexity
- Moving Average served as a reasonable baseline but underperformed for products with trends or seasonality

### 3.2 Inventory Optimization Benefits

The optimization recommendations demonstrated significant benefits:

- **Cost Savings**: Potential 15-30% reduction in total inventory costs
- **Service Level Improvement**: Reduction in stockouts while maintaining optimal inventory
- **Working Capital Efficiency**: Less capital tied up in excess inventory
- **Operational Efficiency**: Clear reorder schedule to streamline purchasing

### 3.3 Trade-offs Analysis

Analyzed the trade-offs between different service levels:
- Higher service levels (98%) required significantly more safety stock
- Lower service levels (80%) had lower holding costs but higher stockout risks
- The optimal service level depends on the specific product characteristics and business priorities

## 4. Conclusion and Future Work

### 4.1 Conclusion

This project demonstrates how small businesses can leverage data science to optimize their inventory management. By combining time series forecasting with inventory optimization models, businesses can reduce costs, prevent stockouts, and make more informed decisions.

The implemented solution provides:
- Data-driven demand forecasts
- Optimal inventory parameters
- Clear reorder recommendations
- Visualization of key metrics and insights

### 4.2 Limitations

- The model assumes relatively stable demand patterns
- Lead time variability is modeled simply
- Supplier constraints and volume discounts are not considered
- The optimization is product-by-product rather than portfolio-based

### 4.3 Future Work

Several enhancements could further improve the system:

- **Multi-echelon Optimization**: Consider warehouse and retail location inventory together
- **Supplier Integration**: Include supplier constraints and volume discounts
- **Machine Learning for Safety Stock**: Dynamically adjust safety stock based on more factors
- **Real-time Integration**: Connect with point-of-sale systems for automatic updates
- **What-if Scenario Analysis**: Allow users to simulate different scenarios

## 5. Technical Implementation

### 5.1 Technology Stack

- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning models
- **Statsmodels**: Statistical time series models
- **Plotly & Dash**: Interactive dashboard

### 5.2 Project Structure

The project follows a modular structure:
- Data generation and processing
- Exploratory data analysis
- Time series forecasting
- Inventory optimization
- Interactive dashboard

### 5.3 Code Quality and Documentation

The project maintains high code quality through:
- Consistent documentation with docstrings
- Modular design with reusable functions
- Clear variable naming and code organization
- Comprehensive comments explaining complex logic

## 6. References

1. Silver, E. A., Pyke, D. F., & Peterson, R. (1998). Inventory management and production planning and scheduling (Vol. 3). New York: Wiley.

2. Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice. OTexts.

3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.

4. Wild, T. (2017). Best practice in inventory management. Routledge.

5. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons.