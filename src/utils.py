"""
Utility functions for the Supply Chain Optimization project.

This module contains reusable functions for data processing,
calculation of inventory metrics, and optimization models.
"""

import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_eoq(annual_demand, setup_cost, holding_cost_rate, unit_price):
    """
    Calculate the Economic Order Quantity.
    
    Parameters:
    -----------
    annual_demand : float
        The annual demand for the product in units
    setup_cost : float
        The fixed cost of placing an order
    holding_cost_rate : float
        The annual holding cost as a percentage of unit price
    unit_price : float
        The cost per unit
        
    Returns:
    --------
    float
        The optimal order quantity
    """
    holding_cost = holding_cost_rate * unit_price  # Annual holding cost per unit
    eoq = math.sqrt(2 * annual_demand * setup_cost / holding_cost)
    return eoq

def calculate_reorder_point(lead_time, daily_demand, safety_factor, std_demand):
    """
    Calculate the Reorder Point.
    
    Parameters:
    -----------
    lead_time : float
        The average lead time in days
    daily_demand : float
        The average daily demand
    safety_factor : float
        The safety factor (z-score) based on desired service level
    std_demand : float
        The standard deviation of daily demand
        
    Returns:
    --------
    float
        The reorder point
    """
    # Adjust standard deviation for lead time
    std_lead_time = std_demand * math.sqrt(lead_time)
    rop = lead_time * daily_demand + safety_factor * std_lead_time
    return rop

def calculate_safety_stock(safety_factor, std_demand, lead_time):
    """
    Calculate Safety Stock.
    
    Parameters:
    -----------
    safety_factor : float
        The safety factor (z-score) based on desired service level
    std_demand : float
        The standard deviation of daily demand
    lead_time : float
        The average lead time in days
        
    Returns:
    --------
    float
        The safety stock level
    """
    std_lead_time = std_demand * math.sqrt(lead_time)
    safety_stock = safety_factor * std_lead_time
    return safety_stock

def calculate_total_cost(annual_demand, unit_price, order_quantity, setup_cost, holding_cost_rate):
    """
    Calculate Total Annual Inventory Cost.
    
    Parameters:
    -----------
    annual_demand : float
        The annual demand for the product in units
    unit_price : float
        The cost per unit
    order_quantity : float
        The order quantity
    setup_cost : float
        The fixed cost of placing an order
    holding_cost_rate : float
        The annual holding cost as a percentage of unit price
        
    Returns:
    --------
    float
        The total annual inventory cost
    """
    holding_cost = holding_cost_rate * unit_price  # Annual holding cost per unit
    purchase_cost = annual_demand * unit_price
    ordering_cost = (annual_demand / order_quantity) * setup_cost
    holding_cost_total = (order_quantity / 2) * holding_cost
    return purchase_cost + ordering_cost + holding_cost_total

def service_level_to_z(service_level):
    """
    Convert service level to Z-score for normal distribution.
    
    Parameters:
    -----------
    service_level : float
        The desired service level (0.0 to 1.0)
        
    Returns:
    --------
    float
        The corresponding Z-score
    """
    # Common service level to Z-score mapping
    sl_to_z = {
        0.50: 0.00,
        0.75: 0.67,
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.65,
        0.98: 2.05,
        0.99: 2.33
    }
    
    # Return the closest value if exact match not found
    if service_level in sl_to_z:
        return sl_to_z[service_level]
    else:
        # Find the closest key
        closest_key = min(sl_to_z.keys(), key=lambda k: abs(k - service_level))
        return sl_to_z[closest_key]

def plot_inventory_profile(order_quantity, demand_rate, lead_time, initial_inventory=None):
    """
    Plot the inventory profile over time.
    
    Parameters:
    -----------
    order_quantity : float
        The order quantity
    demand_rate : float
        The daily demand rate
    lead_time : float
        The lead time in days
    initial_inventory : float, optional
        The starting inventory level, defaults to order_quantity
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    if initial_inventory is None:
        initial_inventory = order_quantity
    
    # Calculate cycle time (time to use up one order)
    cycle_time = order_quantity / demand_rate
    
    # Calculate reorder point
    reorder_point = lead_time * demand_rate
    
    # Create time points
    time_points = np.linspace(0, cycle_time * 3, 1000)
    
    # Initialize inventory levels
    inventory = np.zeros_like(time_points)
    
    # Set initial inventory
    inventory[0] = initial_inventory
    
    # Current time and inventory
    current_time = 0
    current_inventory = initial_inventory
    
    # Track orders
    orders = []
    deliveries = []
    
    # Simulate inventory changes
    for i, t in enumerate(time_points[1:], 1):
        # Time increment
        dt = time_points[i] - time_points[i-1]
        
        # Decrease inventory due to demand
        current_inventory -= demand_rate * dt
        
        # Ensure non-negative inventory
        current_inventory = max(0, current_inventory)
        
        # Check if we need to place an order
        if current_inventory <= reorder_point and not any(o <= t <= o + lead_time for o in orders):
            orders.append(t)
            deliveries.append(t + lead_time)
        
        # Check if an order arrives
        for delivery_time in deliveries:
            if time_points[i-1] < delivery_time <= t:
                current_inventory += order_quantity
        
        # Update inventory array
        inventory[i] = current_inventory
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot inventory level
    ax.plot(time_points, inventory, label='Inventory Level')
    
    # Plot reorder point
    ax.axhline(y=reorder_point, color='r', linestyle='--', label='Reorder Point')
    
    # Mark order placements
    for order_time in orders:
        ax.axvline(x=order_time, color='g', linestyle=':', alpha=0.7)
    
    # Mark deliveries
    for delivery_time in deliveries:
        ax.axvline(x=delivery_time, color='b', linestyle=':', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Inventory Level')
    ax.set_title('Inventory Profile with EOQ Model')
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_eoq_costs(annual_demand, setup_cost, holding_cost_rate, unit_price):
    """
    Plot the total cost curve for different order quantities.
    
    Parameters:
    -----------
    annual_demand : float
        The annual demand for the product in units
    setup_cost : float
        The fixed cost of placing an order
    holding_cost_rate : float
        The annual holding cost as a percentage of unit price
    unit_price : float
        The cost per unit
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Calculate EOQ
    eoq = calculate_eoq(annual_demand, setup_cost, holding_cost_rate, unit_price)
    
    # Generate a range of order quantities
    order_quantities = np.linspace(eoq * 0.2, eoq * 3, 100)
    
    # Calculate costs for each order quantity
    holding_costs = []
    ordering_costs = []
    total_costs = []
    
    for q in order_quantities:
        # Calculate holding cost
        h_cost = (q / 2) * holding_cost_rate * unit_price
        holding_costs.append(h_cost)
        
        # Calculate ordering cost
        o_cost = (annual_demand / q) * setup_cost
        ordering_costs.append(o_cost)
        
        # Calculate total cost (excluding purchase cost)
        total_costs.append(h_cost + o_cost)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot costs
    ax.plot(order_quantities, holding_costs, label='Holding Cost', color='blue')
    ax.plot(order_quantities, ordering_costs, label='Ordering Cost', color='green')
    ax.plot(order_quantities, total_costs, label='Total Cost', color='red', linewidth=2)
    
    # Mark EOQ
    ax.axvline(x=eoq, color='black', linestyle='--', label=f'EOQ = {eoq:.2f}')
    
    # Set labels and title
    ax.set_xlabel('Order Quantity')
    ax.set_ylabel('Annual Cost')
    ax.set_title('EOQ Cost Analysis')
    ax.legend()
    ax.grid(True)
    
    return fig