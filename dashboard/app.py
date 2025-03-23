# Import necessary libraries
import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Load the data
def load_data():
    # Find the most recent data files
    data_dir = '../data/'

    # Load product data
    products_files = sorted([f for f in os.listdir(data_dir) if f.startswith('products_')])
    products_file = products_files[-1] if products_files else None
    
    # Load sales data
    sales_files = sorted([f for f in os.listdir(data_dir) if f.startswith('sales_')])
    sales_file = sales_files[-1] if sales_files else None
    
    # Load inventory data
    inventory_files = sorted([f for f in os.listdir(data_dir) 
                            if f.startswith('inventory_') 
                            and not f == 'inventory_metrics.csv'
                            and not f == 'inventory_optimization_results.csv'
                            and not f == 'inventory_stock_status.csv'
                            ])
    inventory_file = inventory_files[-1] if inventory_files else None
    
    # Load forecasts
    forecasts_file = 'demand_forecasts.csv' if 'demand_forecasts.csv' in os.listdir(data_dir) else None
    
    # Load optimization results
    optimization_file = 'inventory_optimization_results.csv' if 'inventory_optimization_results.csv' in os.listdir(data_dir) else None
    
    # Load stock status
    stock_status_file = 'inventory_stock_status.csv' if 'inventory_stock_status.csv' in os.listdir(data_dir) else None
    
    # Load reorder schedule
    reorder_schedule_file = 'reorder_schedule.csv' if 'reorder_schedule.csv' in os.listdir(data_dir) else None
    
    # Dictionary to store dataframes
    data = {}
    
    # Load each dataframe if file exists
    if products_file:
        data['products'] = pd.read_csv(os.path.join(data_dir, products_file))
    else:
        data['products'] = pd.DataFrame()
    
    if sales_file:
        data['sales'] = pd.read_csv(os.path.join(data_dir, sales_file))
        data['sales']['date'] = pd.to_datetime(data['sales']['date'])
    else:
        data['sales'] = pd.DataFrame()
    
    if inventory_file:
        try:
            data['inventory'] = pd.read_csv(os.path.join(data_dir, inventory_file))
            
            # Check if 'date' column exists before converting
            if 'date' in data['inventory'].columns:
                data['inventory']['date'] = pd.to_datetime(data['inventory']['date'])
                if 'expected_delivery' in data['inventory'].columns:
                    data['inventory']['expected_delivery'] = pd.to_datetime(data['inventory']['expected_delivery'])
            else:
                print(f"Warning: 'date' column not found in inventory file: {inventory_file}")
                print(f"Available columns: {data['inventory'].columns.tolist()}")
                
                # Let's try to find the correct inventory file
                raw_inventory_files = [f for f in os.listdir(data_dir) 
                                    if f.startswith('inventory_') 
                                    and not f == 'inventory_metrics.csv'
                                    and not f == 'inventory_optimization_results.csv'
                                    and not f == 'inventory_stock_status.csv']
                
                if raw_inventory_files:
                    raw_inventory_file = sorted(raw_inventory_files)[-1]
                    print(f"Found alternative inventory file: {raw_inventory_file}")
                    data['inventory'] = pd.read_csv(os.path.join(data_dir, raw_inventory_file))
                    
                    if 'date' in data['inventory'].columns:
                        data['inventory']['date'] = pd.to_datetime(data['inventory']['date'])
                        if 'expected_delivery' in data['inventory'].columns:
                            data['inventory']['expected_delivery'] = pd.to_datetime(data['inventory']['expected_delivery'])
                    else:
                        # If we still can't find the right file, create an empty DataFrame with the expected structure
                        print("Could not find inventory file with 'date' column. Using empty DataFrame.")
                        data['inventory'] = pd.DataFrame(columns=['date', 'product_id', 'event_type', 'quantity', 'expected_delivery'])
                else:
                    # If we still can't find the right file, create an empty DataFrame with the expected structure
                    print("Could not find any suitable inventory files. Using empty DataFrame.")
                    data['inventory'] = pd.DataFrame(columns=['date', 'product_id', 'event_type', 'quantity', 'expected_delivery'])
        except Exception as e:
            print(f"Error loading inventory data: {e}")
            data['inventory'] = pd.DataFrame(columns=['date', 'product_id', 'event_type', 'quantity', 'expected_delivery'])
    else:
        data['inventory'] = pd.DataFrame()
    
    if forecasts_file:
        data['forecasts'] = pd.read_csv(os.path.join(data_dir, forecasts_file))
        data['forecasts']['date'] = pd.to_datetime(data['forecasts']['date'])
    else:
        data['forecasts'] = pd.DataFrame()
    
    if optimization_file:
        data['optimization'] = pd.read_csv(os.path.join(data_dir, optimization_file))
    else:
        data['optimization'] = pd.DataFrame()
    
    if stock_status_file:
        data['stock_status'] = pd.read_csv(os.path.join(data_dir, stock_status_file))
    else:
        data['stock_status'] = pd.DataFrame()
    
    if reorder_schedule_file:
        data['reorder_schedule'] = pd.read_csv(os.path.join(data_dir, reorder_schedule_file))
        data['reorder_schedule']['reorder_date'] = pd.to_datetime(data['reorder_schedule']['reorder_date'])
        data['reorder_schedule']['expected_delivery'] = pd.to_datetime(data['reorder_schedule']['expected_delivery'])
    else:
        data['reorder_schedule'] = pd.DataFrame()
    
    return data

# Load the data
data = load_data()

# Define the app layout
app.layout = html.Div([
    html.H1("Supply Chain Optimization Dashboard", style={'textAlign': 'center'}),
    
    # Navigation tabs
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Overview', value='tab-1'),
        dcc.Tab(label='Demand Forecasting', value='tab-2'),
        dcc.Tab(label='Inventory Optimization', value='tab-3'),
        dcc.Tab(label='Reorder Planning', value='tab-4'),
    ]),
    
    # Content div
    html.Div(id='tabs-content')
])

# Callback to handle tab switching
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        # Overview Tab
        return html.Div([
            html.H2("Business Overview"),
            
            # Summary statistics
            html.Div([
                html.Div([
                    html.H4("Key Metrics"),
                    html.Div([
                        html.Div([
                            html.H5("Total Products"),
                            html.H3(f"{len(data['products'])}")
                        ], className='stat-box'),
                        html.Div([
                            html.H5("Total Sales"),
                            html.H3(f"${data['sales']['total_sales'].sum():,.2f}")
                        ], className='stat-box'),
                        html.Div([
                            html.H5("Avg Daily Demand"),
                            html.H3(f"{data['sales'].groupby('date')['demand'].sum().mean():.1f} units")
                        ], className='stat-box'),
                        html.Div([
                            html.H5("Current Inventory Value"),
                            html.H3(f"${sum(data['stock_status']['current_inventory'] * data['products']['price']):,.2f}")
                        ], className='stat-box'),
                    ], style={'display': 'flex', 'justifyContent': 'space-around'}),
                ], style={'marginBottom': '30px'}),
            ]),
            
            # Sales trend chart
            html.Div([
                html.H4("Sales Trend"),
                dcc.Graph(id='sales-trend')
            ]),
            
            # Product category breakdown
            html.Div([
                html.Div([
                    html.H4("Sales by Category"),
                    dcc.Graph(id='sales-by-category')
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H4("Inventory Status"),
                    dcc.Graph(id='inventory-status')
                ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
            ]),
            
            # Cost analysis
            html.Div([
                html.H4("Cost Analysis"),
                dcc.Graph(id='cost-analysis')
            ]),
            
        ])
    
    elif tab == 'tab-2':
        # Demand Forecasting Tab
        return html.Div([
            html.H2("Demand Forecasting"),
            
            # Product selector
            html.Div([
                html.H4("Select Product"),
                dcc.Dropdown(
                    id='product-selector-forecast',
                    options=[{'label': f"{row['product_name']} ({row['product_id']})", 
                             'value': row['product_id']} 
                             for _, row in data['products'].iterrows()],
                    value=data['products']['product_id'].iloc[0]
                )
            ]),
            
            # Historical vs forecast chart
            html.Div([
                html.H4("Historical Demand vs Forecast"),
                dcc.Graph(id='historical-forecast')
            ]),
            
            # Forecast statistics
            html.Div([
                html.H4("Forecast Statistics"),
                dash_table.DataTable(
                    id='forecast-stats-table',
                    style_cell={'textAlign': 'center'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                )
            ]),
            
            # Seasonal patterns
            html.Div([
                html.H4("Seasonal Patterns"),
                dcc.Graph(id='seasonal-patterns')
            ]),
        ])
    
    elif tab == 'tab-3':
        # Inventory Optimization Tab
        return html.Div([
            html.H2("Inventory Optimization"),
            
            # Service level selector
            html.Div([
                html.H4("Select Service Level"),
                dcc.RadioItems(
                    id='service-level-selector',
                    options=[
                        {'label': '80% Service Level', 'value': 0.8},
                        {'label': '90% Service Level', 'value': 0.9},
                        {'label': '95% Service Level', 'value': 0.95},
                        {'label': '98% Service Level', 'value': 0.98},
                    ],
                    value=0.95,
                    inline=True
                )
            ]),
            
            # Optimal parameters table
            html.Div([
                html.H4("Optimal Inventory Parameters"),
                dash_table.DataTable(
                    id='optimization-table',
                    sort_action='native',
                    filter_action='native',
                    style_cell={'textAlign': 'center'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                )
            ]),
            
            # Current vs optimal inventory
            html.Div([
                html.H4("Current vs Optimal Inventory"),
                dcc.Graph(id='current-vs-optimal')
            ]),
            
            # Cost comparison
            html.Div([
                html.H4("Cost Comparison"),
                dcc.Graph(id='cost-comparison')
            ]),
            
            # Potential savings
            html.Div([
                html.H4("Potential Cost Savings"),
                dcc.Graph(id='potential-savings')
            ]),
        ])
    
    elif tab == 'tab-4':
        # Reorder Planning Tab
        return html.Div([
            html.H2("Reorder Planning"),
            
            # Reorder schedule table
            html.Div([
                html.H4("Reorder Schedule"),
                dash_table.DataTable(
                    id='reorder-schedule-table',
                    sort_action='native',
                    filter_action='native',
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{days_until_reorder} = 0',
                            },
                            'backgroundColor': 'rgba(255, 100, 100, 0.3)',
                            'fontWeight': 'bold'
                        },
                    ],
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                )
            ]),
            
            # Reorder timeline
            html.Div([
                html.H4("Reorder Timeline"),
                dcc.Graph(id='reorder-timeline')
            ]),
            
            # Inventory projection
            html.Div([
                html.H4("Inventory Projection"),
                html.Div([
                    html.H4("Select Product"),
                    dcc.Dropdown(
                        id='product-selector-projection',
                        options=[{'label': f"{row['product_name']} ({row['product_id']})", 
                                 'value': row['product_id']} 
                                 for _, row in data['products'].iterrows()],
                        value=data['products']['product_id'].iloc[0]
                    )
                ]),
                dcc.Graph(id='inventory-projection')
            ]),
        ])

# Callback for Overview Tab
@app.callback(
    [Output('sales-trend', 'figure'),
     Output('sales-by-category', 'figure'),
     Output('inventory-status', 'figure'),
     Output('cost-analysis', 'figure')],
    Input('tabs', 'value')
)
def update_overview_charts(tab):
    # Only calculate if we're on the overview tab
    if tab != 'tab-1':
        return [go.Figure()] * 4
    
    # Sales trend chart
    # Group by date and sum demand
    if not data['sales'].empty:
        daily_sales = data['sales'].groupby('date')['total_sales'].sum().reset_index()
        
        # Calculate 7-day moving average
        daily_sales['7d_avg'] = daily_sales['total_sales'].rolling(window=7, min_periods=1).mean()
        
        sales_trend_fig = make_subplots()
        
        sales_trend_fig.add_trace(
            go.Scatter(x=daily_sales['date'], y=daily_sales['total_sales'],
                     mode='lines', name='Daily Sales', line=dict(color='lightblue'))
        )
        
        sales_trend_fig.add_trace(
            go.Scatter(x=daily_sales['date'], y=daily_sales['7d_avg'],
                     mode='lines', name='7-Day Avg', line=dict(color='darkblue', width=2))
        )
        
        sales_trend_fig.update_layout(
            title='Daily Sales Trend',
            xaxis_title='Date',
            yaxis_title='Total Sales ($)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
    else:
        sales_trend_fig = go.Figure()
    
    # Sales by category chart
    if not data['sales'].empty and not data['products'].empty:
        # Create a merged dataset with product categories
        merged_data = pd.merge(
            data['sales'], 
            data['products'][['product_id', 'category']],
            on='product_id'
        )
        
        # Group by category and sum total_sales
        category_sales = merged_data.groupby('category')['total_sales'].sum().reset_index()
        
        sales_category_fig = px.pie(
            category_sales, 
            values='total_sales', 
            names='category',
            title='Sales by Product Category',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        sales_category_fig.update_traces(textposition='inside', textinfo='percent+label')
    else:
        sales_category_fig = go.Figure()
    
    # Inventory status chart
    if not data['stock_status'].empty:
        status_counts = data['stock_status']['inventory_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        color_map = {
            'Understocked': 'red',
            'Optimal': 'green',
            'Overstocked': 'orange'
        }
        
        inventory_status_fig = px.bar(
            status_counts, 
            x='Status', 
            y='Count',
            title='Inventory Status Distribution',
            color='Status',
            color_discrete_map=color_map
        )
    else:
        inventory_status_fig = go.Figure()
    
    # Cost analysis chart
    if not data['stock_status'].empty:
        cost_data = data['stock_status'][['product_id', 'current_total_cost', 'optimized_annual_cost', 'potential_savings']]
        cost_data = cost_data.sort_values('potential_savings', ascending=False)
        
        cost_analysis_fig = go.Figure()
        
        cost_analysis_fig.add_trace(
            go.Bar(x=cost_data['product_id'], y=cost_data['current_total_cost'],
                 name='Current Annual Cost', marker_color='crimson')
        )
        
        cost_analysis_fig.add_trace(
            go.Bar(x=cost_data['product_id'], y=cost_data['optimized_annual_cost'],
                 name='Optimized Annual Cost', marker_color='royalblue')
        )
        
        cost_analysis_fig.update_layout(
            title='Current vs Optimized Annual Costs',
            xaxis_title='Product ID',
            yaxis_title='Annual Cost ($)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            barmode='group'
        )
    else:
        cost_analysis_fig = go.Figure()
    
    return sales_trend_fig, sales_category_fig, inventory_status_fig, cost_analysis_fig

# Callback for Demand Forecasting Tab
@app.callback(
    [Output('historical-forecast', 'figure'),
     Output('forecast-stats-table', 'data'),
     Output('forecast-stats-table', 'columns'),
     Output('seasonal-patterns', 'figure')],
    Input('product-selector-forecast', 'value')
)
def update_forecast_charts(product_id):
    if not product_id or data['sales'].empty or data['forecasts'].empty:
        return go.Figure(), [], [], go.Figure()
    
    # Filter data for the selected product
    product_sales = data['sales'][data['sales']['product_id'] == product_id].copy()
    product_forecast = data['forecasts'][data['forecasts']['product_id'] == product_id].copy()
    
    # Get product name
    product_name = data['products'][data['products']['product_id'] == product_id]['product_name'].iloc[0]
    
    # Historical vs Forecast chart
    historical_forecast_fig = go.Figure()
    
    # Add historical data
    historical_forecast_fig.add_trace(
        go.Scatter(x=product_sales['date'], y=product_sales['demand'],
                 mode='lines', name='Historical Demand', line=dict(color='blue'))
    )
    
    # Add forecast data
    historical_forecast_fig.add_trace(
        go.Scatter(x=product_forecast['date'], y=product_forecast['forecast_demand'],
                 mode='lines', name='Forecast', line=dict(color='red', dash='dash'))
    )
    
    historical_forecast_fig.update_layout(
        title=f'Historical Demand vs Forecast for {product_name}',
        xaxis_title='Date',
        yaxis_title='Demand',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Forecast statistics
    # Calculate forecast statistics
    forecast_mean = product_forecast['forecast_demand'].mean()
    forecast_std = product_forecast['forecast_demand'].std()
    forecast_min = product_forecast['forecast_demand'].min()
    forecast_max = product_forecast['forecast_demand'].max()
    forecast_total = product_forecast['forecast_demand'].sum()
    
    # Historical statistics for comparison
    historical_mean = product_sales['demand'].mean()
    historical_std = product_sales['demand'].std()
    
    # Calculate percent change
    percent_change = ((forecast_mean - historical_mean) / historical_mean) * 100 if historical_mean > 0 else 0
    
    stats_data = [{
        'Metric': 'Average Daily Demand',
        'Historical': f'{historical_mean:.2f}',
        'Forecast': f'{forecast_mean:.2f}',
        'Change': f'{percent_change:.2f}%'
    },
    {
        'Metric': 'Standard Deviation',
        'Historical': f'{historical_std:.2f}',
        'Forecast': f'{forecast_std:.2f}',
        'Change': 'N/A'
    },
    {
        'Metric': 'Minimum Demand',
        'Historical': f'{product_sales["demand"].min():.2f}',
        'Forecast': f'{forecast_min:.2f}',
        'Change': 'N/A'
    },
    {
        'Metric': 'Maximum Demand',
        'Historical': f'{product_sales["demand"].max():.2f}',
        'Forecast': f'{forecast_max:.2f}',
        'Change': 'N/A'
    },
    {
        'Metric': 'Total 30-Day Demand',
        'Historical': f'{product_sales["demand"].tail(30).sum():.2f}',
        'Forecast': f'{forecast_total:.2f}',
        'Change': 'N/A'
    }]
    
    stats_columns = [
        {'name': 'Metric', 'id': 'Metric'},
        {'name': 'Historical', 'id': 'Historical'},
        {'name': 'Forecast', 'id': 'Forecast'},
        {'name': 'Change', 'id': 'Change'}
    ]
    
    # Seasonal patterns chart
    # Add day of week to sales data
    product_sales['day_of_week'] = product_sales['date'].dt.dayofweek
    product_sales['day_name'] = product_sales['date'].dt.day_name()
    product_sales['month'] = product_sales['date'].dt.month
    product_sales['month_name'] = product_sales['date'].dt.month_name()
    
    # Calculate average demand by day of week
    day_of_week_avg = product_sales.groupby('day_of_week').agg({
        'demand': 'mean',
        'day_name': 'first'
    }).reset_index()
    day_of_week_avg = day_of_week_avg.sort_values('day_of_week')
    
    # Calculate average demand by month
    month_avg = product_sales.groupby('month').agg({
        'demand': 'mean',
        'month_name': 'first'
    }).reset_index()
    month_avg = month_avg.sort_values('month')
    
    # Create subplots
    seasonal_fig = make_subplots(rows=1, cols=2, 
                                subplot_titles=('Average Demand by Day of Week', 
                                               'Average Demand by Month'))
    
    # Day of week subplot
    seasonal_fig.add_trace(
        go.Bar(x=day_of_week_avg['day_name'], y=day_of_week_avg['demand'],
              marker_color='lightskyblue'),
        row=1, col=1
    )
    
    # Month subplot
    seasonal_fig.add_trace(
        go.Bar(x=month_avg['month_name'], y=month_avg['demand'],
              marker_color='lightgreen'),
        row=1, col=2
    )
    
    seasonal_fig.update_layout(
        title=f'Seasonal Patterns for {product_name}',
        showlegend=False,
        height=500
    )
    
    return historical_forecast_fig, stats_data, stats_columns, seasonal_fig

# Callback for Inventory Optimization Tab
@app.callback(
    [Output('optimization-table', 'data'),
     Output('optimization-table', 'columns'),
     Output('current-vs-optimal', 'figure'),
     Output('cost-comparison', 'figure'),
     Output('potential-savings', 'figure')],
    Input('service-level-selector', 'value')
)
def update_optimization_charts(service_level):
    if data['optimization'].empty or data['stock_status'].empty:
        return [], [], go.Figure(), go.Figure(), go.Figure()
    
    # Filter optimization data for selected service level
    opt_data = data['optimization'][data['optimization']['service_level'] == service_level]
    
    # Prepare data for the table
    table_data = []
    
    for _, row in opt_data.iterrows():
        table_data.append({
            'Product ID': row['product_id'],
            'Product Name': row['product_name'],
            'Category': row['category'],
            'Current Inventory': int(row['current_inventory']),
            'Reorder Point': int(row['reorder_point']),
            'Optimal Order Quantity': int(row['optimal_order_quantity']),
            'Safety Stock': int(row['safety_stock']),
            'Order Urgency': row['order_urgency']
        })
    
    table_columns = [
        {'name': 'Product ID', 'id': 'Product ID'},
        {'name': 'Product Name', 'id': 'Product Name'},
        {'name': 'Category', 'id': 'Category'},
        {'name': 'Current Inventory', 'id': 'Current Inventory'},
        {'name': 'Reorder Point', 'id': 'Reorder Point'},
        {'name': 'Optimal Order Quantity', 'id': 'Optimal Order Quantity'},
        {'name': 'Safety Stock', 'id': 'Safety Stock'},
        {'name': 'Order Urgency', 'id': 'Order Urgency'}
    ]
    
    # Current vs Optimal Inventory chart
    current_vs_optimal_fig = go.Figure()
    
    current_vs_optimal_fig.add_trace(
        go.Bar(x=opt_data['product_id'], y=opt_data['current_inventory'],
             name='Current Inventory', marker_color='blue')
    )
    
    current_vs_optimal_fig.add_trace(
        go.Bar(x=opt_data['product_id'], y=opt_data['reorder_point'],
             name='Reorder Point', marker_color='red')
    )
    
    current_vs_optimal_fig.add_trace(
        go.Bar(x=opt_data['product_id'], y=opt_data['safety_stock'],
             name='Safety Stock', marker_color='green')
    )
    
    current_vs_optimal_fig.update_layout(
        title=f'Current Inventory vs Optimal Levels ({service_level*100:.0f}% Service Level)',
        xaxis_title='Product ID',
        yaxis_title='Units',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        barmode='group'
    )
    
    # Cost comparison chart
    # Use the stock status data which has cost calculations
    stock_data = data['stock_status'][data['stock_status']['service_level'] == service_level]
    
    cost_comparison_fig = go.Figure()
    
    cost_comparison_fig.add_trace(
        go.Bar(
            x=stock_data['product_id'],
            y=stock_data['current_annual_holding_cost'],
            name='Current Holding Cost',
            marker_color='darkblue'
        )
    )
    
    cost_comparison_fig.add_trace(
        go.Bar(
            x=stock_data['product_id'],
            y=stock_data['current_annual_stockout_cost'],
            name='Current Stockout Cost',
            marker_color='darkred'
        )
    )
    
    cost_comparison_fig.add_trace(
        go.Bar(
            x=stock_data['product_id'],
            y=stock_data['optimized_annual_cost'],
            name='Optimized Total Cost',
            marker_color='darkgreen'
        )
    )
    
    cost_comparison_fig.update_layout(
        title=f'Cost Breakdown Comparison ({service_level*100:.0f}% Service Level)',
        xaxis_title='Product ID',
        yaxis_title='Annual Cost ($)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        barmode='group'
    )
    
    # Potential savings chart
    savings_fig = go.Figure()
    
    # Sort by savings
    sorted_stock_data = stock_data.sort_values('potential_savings', ascending=False)
    
    savings_fig.add_trace(
        go.Bar(
            x=sorted_stock_data['product_id'],
            y=sorted_stock_data['potential_savings'],
            marker_color='green',
            text=sorted_stock_data['potential_savings'].apply(lambda x: f'${x:,.2f}'),
            textposition='auto'
        )
    )
    
    savings_fig.update_layout(
        title=f'Potential Annual Cost Savings ({service_level*100:.0f}% Service Level)',
        xaxis_title='Product ID',
        yaxis_title='Cost Savings ($)'
    )
    
    return table_data, table_columns, current_vs_optimal_fig, cost_comparison_fig, savings_fig

# Callback for Reorder Planning Tab
@app.callback(
    [Output('reorder-schedule-table', 'data'),
     Output('reorder-schedule-table', 'columns'),
     Output('reorder-timeline', 'figure'),
     Output('inventory-projection', 'figure')],
    [Input('tabs', 'value'),
     Input('product-selector-projection', 'value')]
)
def update_reorder_planning(tab, selected_product):
    if tab != 'tab-4' or data['reorder_schedule'].empty:
        return [], [], go.Figure(), go.Figure()
    
    # Prepare data for the reorder schedule table
    table_data = []
    
    for _, row in data['reorder_schedule'].sort_values('days_until_reorder').iterrows():
        table_data.append({
            'Product ID': row['product_id'],
            'Product Name': row['product_name'],
            'Current Inventory': int(row['current_inventory']),
            'Reorder Point': int(row['reorder_point']),
            'Days Until Reorder': int(row['days_until_reorder']),
            'Reorder Date': row['reorder_date'].strftime('%Y-%m-%d'),
            'Order Quantity': int(row['order_quantity']),
            'Expected Delivery': row['expected_delivery'].strftime('%Y-%m-%d'),
            'Days of Coverage': f"{row['days_of_coverage_after_delivery']:.1f}"
        })
    
    table_columns = [
        {'name': 'Product ID', 'id': 'Product ID'},
        {'name': 'Product Name', 'id': 'Product Name'},
        {'name': 'Current Inventory', 'id': 'Current Inventory'},
        {'name': 'Reorder Point', 'id': 'Reorder Point'},
        {'name': 'Days Until Reorder', 'id': 'Days Until Reorder'},
        {'name': 'Reorder Date', 'id': 'Reorder Date'},
        {'name': 'Order Quantity', 'id': 'Order Quantity'},
        {'name': 'Expected Delivery', 'id': 'Expected Delivery'},
        {'name': 'Days of Coverage', 'id': 'Days of Coverage'}
    ]
    
    # Reorder timeline chart
    # Create a Gantt chart for reordering
    reorder_schedule = data['reorder_schedule'].copy()
    
    # Create task lists for the Gantt chart
    tasks = []
    
    for _, row in reorder_schedule.iterrows():
        # Add a task for the lead time period
        tasks.append({
            'Task': row['product_id'],
            'Start': row['reorder_date'],
            'Finish': row['expected_delivery'],
            'Resource': 'Lead Time'
        })
        
        # Calculate coverage end date
        coverage_end = row['expected_delivery'] + pd.Timedelta(days=int(row['days_of_coverage_after_delivery']))
        
        # Add a task for the coverage period
        tasks.append({
            'Task': row['product_id'],
            'Start': row['expected_delivery'],
            'Finish': coverage_end,
            'Resource': 'Inventory Coverage'
        })
    
    # Convert to DataFrame for the Gantt chart
    tasks_df = pd.DataFrame(tasks)
    
    # Create color map
    color_map = {
        'Lead Time': 'rgb(220, 0, 0)',
        'Inventory Coverage': 'rgb(0, 220, 0)'
    }
    
    # Create the Gantt chart
    timeline_fig = px.timeline(
        tasks_df, 
        x_start='Start', 
        x_end='Finish', 
        y='Task',
        color='Resource',
        color_discrete_map=color_map,
        title='Reorder and Inventory Coverage Timeline',
        labels={'Task': 'Product ID'}
    )
    
    timeline_fig.update_layout(
        xaxis_title='Date',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Inventory projection chart
    if not selected_product:
        inventory_projection_fig = go.Figure()
    else:
        # Get data for the selected product
        product_forecast = data['forecasts'][data['forecasts']['product_id'] == selected_product].copy()
        product_reorder = data['reorder_schedule'][data['reorder_schedule']['product_id'] == selected_product].iloc[0]
        
        # Get product details
        product_name = data['products'][data['products']['product_id'] == selected_product]['product_name'].iloc[0]
        
        # Get current inventory and reorder parameters
        current_inventory = product_reorder['current_inventory']
        reorder_point = product_reorder['reorder_point']
        reorder_date = product_reorder['reorder_date']
        order_quantity = product_reorder['order_quantity']
        expected_delivery = product_reorder['expected_delivery']
        
        # Create a date range for the projection
        start_date = product_forecast['date'].min()
        end_date = product_forecast['date'].max()
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Create a DataFrame for the projection
        projection_df = pd.DataFrame({'date': date_range})
        projection_df['product_id'] = selected_product
        
        # Merge with forecast data
        projection_df = pd.merge(projection_df, product_forecast[['date', 'forecast_demand']], on=['date'], how='left')
        
        # Calculate inventory projection
        projection_df['inventory'] = current_inventory
        
        # Initialize delivery date index
        delivery_idx = None
        
        for i in range(1, len(projection_df)):
            prev_inventory = projection_df.loc[i-1, 'inventory']
            demand = projection_df.loc[i-1, 'forecast_demand']
            
            # Update inventory (subtract demand)
            new_inventory = max(0, prev_inventory - demand)
            
            # Check if this is the delivery date
            if projection_df.loc[i, 'date'].date() == expected_delivery.date():
                delivery_idx = i
                new_inventory += order_quantity
            
            projection_df.loc[i, 'inventory'] = new_inventory
        
        # Create the figure
        inventory_projection_fig = go.Figure()
        
        # Add inventory projection line
        inventory_projection_fig.add_trace(
            go.Scatter(
                x=projection_df['date'],
                y=projection_df['inventory'],
                mode='lines',
                name='Projected Inventory',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add reorder point line
        inventory_projection_fig.add_trace(
            go.Scatter(
                x=projection_df['date'],
                y=[reorder_point] * len(projection_df),
                mode='lines',
                name='Reorder Point',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Add demand bars
        inventory_projection_fig.add_trace(
            go.Bar(
                x=projection_df['date'],
                y=projection_df['forecast_demand'],
                name='Daily Demand',
                marker_color='lightblue',
                opacity=0.5,
                yaxis='y2'
            )
        )
        
        # Add delivery marker if applicable
        if delivery_idx is not None:
            inventory_projection_fig.add_trace(
                go.Scatter(
                    x=[projection_df.loc[delivery_idx, 'date']],
                    y=[projection_df.loc[delivery_idx, 'inventory']],
                    mode='markers',
                    name='Order Delivery',
                    marker=dict(color='green', size=12, symbol='triangle-up')
                )
            )
        
        # Update layout
        inventory_projection_fig.update_layout(
            title=f'30-Day Inventory Projection for {product_name}',
            xaxis_title='Date',
            yaxis_title='Inventory Level',
            yaxis2=dict(
                title='Daily Demand',
                overlaying='y',
                side='right'
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
    
    return table_data, table_columns, timeline_fig, inventory_projection_fig

# Add CSS for styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Supply Chain Optimization Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }
            
            .stat-box {
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                margin: 10px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                flex: 1;
                text-align: center;
            }
            
            h1, h2, h4 {
                color: #2c3e50;
            }
            
            h3 {
                color: #3498db;
                margin: 5px 0;
            }
            
            h5 {
                color: #7f8c8d;
                margin: 5px 0;
            }
            
            .dash-table-container {
                margin: 20px 0;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run the app (Local)
# if __name__ == '__main__':
#     app.run_server(debug=True, port=8050)

# server = app.server  # Expose the server (Render)

# if __name__ == '__main__':
#     app.run_server(debug=False)

# AWS Runner
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)