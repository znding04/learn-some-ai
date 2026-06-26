---
title: "Inventory Management and Demand Forecasting with ML"
difficulty: beginner
topic: ai-for-operations-research
order: 4
estimatedTime: "15 minutes"
summary: "Covers inventory optimization models (newsvendor, EOQ, base-stock policies) and modern ML-based demand forecasting using gradient boosting, LSTMs, and Transformer architectures."
---
# Inventory Management and Demand Forecasting with ML

## Overview

Inventory management is the heart of supply chain operations. Every retailer, wholesaler, and manufacturer must answer: **how much should I order, and when?** Too much inventory ties up capital and risks obsolescence; too little causes stockouts and lost sales. This tension is captured by the classic newsvendor model, and extended by EOQ (Economic Order Quantity), Wagner-Whitin dynamic programming, and the (s,S) base-stock policy.

Demand forecasting is the upstream enabler of good inventory decisions. Classical time-series methods — moving averages, exponential smoothing (Holt-Winters), ARIMA — have served operations managers well for decades. But demand in real supply chains is influenced by trends, seasonality, promotions, weather, holidays, and macroeconomic factors — patterns ML can capture better, especially with large historical datasets.

Modern approaches to demand forecasting include:

- **Gradient Boosting (XGBoost, LightGBM)**: Treat demand forecasting as a supervised regression problem with engineered calendar and lag features. Strong baseline, widely used in industry.
- **LSTM / GRU networks**: Sequence models that capture temporal dependencies in demand time series.
- **Transformer-based models (Temporal Fusion Transformer, N-BEATS)**: State-of-the-art for multi-horizon forecasting, capable of handling multiple related series simultaneously.
- **Hierarchical reconciliation**: Ensuring forecasts are coherent across product hierarchies (SKU → product family → category) and geographic hierarchies (store → region → chain).

Forecasted demand $\hat{y}_t$ feeds into inventory optimization via the newsvendor critical fractile:

$$q^* = F^{-1}\left(\frac{p - c}{p}\right)$$

where $F$ is the CDF of demand, $p$ is selling price, $c$ is cost (or underage cost $c_u = p - c$, overage cost $c_o = c - v$). With uncertain demand, expected profit is maximized by ordering the $(p-c)/p$ fractile of the demand distribution.

```mermaid
flowchart TD
    A[Historical Sales Data] --> B[Feature Engineering]
    B --> C[Demand Forecasting Model]
    C --> D[Point Forecast + Uncertainty Quantification]
    D --> E[Inventory Optimization\nNewsvendor / (s,S) / EOQ]
    E --> F[Order Quantity Decision]
    F --> G[Procurement / Production]
    G --> A
```

## Key Concepts

- **EOQ (Economic Order Quantity)**: The optimal fixed order quantity that minimizes total holding + ordering costs, assuming constant demand rate $D$: $Q^* = \sqrt{2KD/h}$.
- **Newsvendor / Single Period**: Order quantity for perishable goods. Optimal $Q^*$ is the critical fractile quantile of the demand distribution.
- **(s,S) Policy**: Base-stock policy: order up to $S$ whenever inventory $I < s$. Optimal for random lead times and i.i.d. demand under certain conditions.
- **Demand forecasting features**: Calendar effects (day-of-week, month, holiday), lag features (demand at t-1, t-7, t-30), promotional flags, price, weather, external signals.
- **XGBoost for demand**: Encode all features, train gradient-boosted trees to predict next-period demand. Handles nonlinearity and interactions automatically.
- **LSTM for time series**: Recurrent architecture that maintains hidden state capturing demand dynamics. Good for short-term (1-4 week) forecasting with strong seasonal patterns.
- **Uncertainty quantification**: Predicting the full distribution (via quantile regression or deep ensembles) matters for inventory — the mean forecast is often not the optimal order quantity under asymmetric costs.

## Code Examples

```python
# Demand forecasting with XGBoost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for demand forecasting."""
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_holiday'] = df['is_holiday'].astype(int)
    df['promotion'] = df['promotion'].astype(int)
    # Lag features
    for lag in [1, 7, 14, 28]:
        df[f'demand_lag_{lag}'] = df['demand'].shift(lag)
    df = df.dropna()
    return df

# Example feature set
features = ['day_of_week', 'month', 'is_holiday', 'promotion',
            'demand_lag_1', 'demand_lag_7', 'demand_lag_14', 'demand_lag_28']
target = 'demand'

# XGBoost model
try:
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        objective='reg:squarederror',
    )
    model.fit(X_train, y_train)
    forecast = model.predict(X_test)
except ImportError:
    print("XGBoost not available — showing scikit-learn alternative")
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5)
    model.fit(X_train, y_train)
    forecast = model.predict(X_test)

# Newsvendor with forecast distribution
from scipy.stats import norm

def newsvendor_order(forecast_mean: float, forecast_std: float,
                     price: float, cost: float, salvage: float = 0) -> float:
    """
    Optimal order quantity using normal demand approximation.
    Critical ratio = (price - cost) / (price - salvage)
    """
    critical_ratio = (price - cost) / (price - salvage)
    optimal_quantile = norm.ppf(critical_ratio)
    q_star = forecast_mean + forecast_std * optimal_quantile
    return max(0, q_star)

q_opt = newsvendor_order(forecast_mean=100, forecast_std=25, price=50, cost=20)
print(f"Optimal order quantity: {q_opt:.1f} units")
```

## Exercises/Projects

- **Exercise 1**: Download a retail demand dataset (e.g., Walmart dataset on Kaggle) and build a lag-feature based XGBoost demand forecaster. Compare MAPE vs. a simple moving average baseline.
- **Exercise 2**: Implement an (s,S) base-stock policy simulator. For a given replenishment lead time and demand distribution, estimate the average inventory level and fill rate.
- **Project**: Build a hierarchical demand forecasting pipeline: forecast at SKU level, then reconcile across product categories using top-down and bottom-up approaches. Evaluate forecast accuracy at each level and compare against independent SKU forecasts.

## Further Reading

- [Demand Forecasting for Operations](https://www.amazon.com/ Forecasting-Prediction-Analytics/dp/1119147022) — Chopra & Van Mieghem (focused on OR applications)
- [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09163) — Lim et al., 2019 (state-of-the-art multi-horizon forecasting)