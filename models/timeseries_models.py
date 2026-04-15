"""Time series forecasting models for Singapore data."""
import numpy as np
import pandas as pd
from models.database import query_db, execute_db
import json


def prepare_time_series_data():
    """Prepare monthly HDB price time series."""
    rows = query_db("""
        SELECT t.year, t.month, AVG(f.resale_price) as avg_price,
               COUNT(*) as transaction_count
        FROM fact_hdb_transactions f
        JOIN dim_time t ON f.time_id = t.time_id
        GROUP BY t.year, t.month
        ORDER BY t.year, t.month
    """)
    if not rows:
        return None
    df = pd.DataFrame([dict(r) for r in rows])
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.sort_values('date')
    return df


def simple_moving_average_forecast(window=3):
    """Simple Moving Average forecast for HDB prices."""
    df = prepare_time_series_data()
    if df is None or len(df) < window + 1:
        return {'error': 'Not enough time series data. Need more monthly data points.'}

    df['sma'] = df['avg_price'].rolling(window=window).mean()
    df['sma_forecast'] = df['sma'].shift(1)

    # Calculate error on available data
    valid = df.dropna(subset=['sma_forecast'])
    if len(valid) > 0:
        mae = float(np.mean(np.abs(valid['avg_price'] - valid['sma_forecast'])))
        rmse = float(np.sqrt(np.mean((valid['avg_price'] - valid['sma_forecast'])**2)))
    else:
        mae, rmse = 0, 0

    # Forecast next period
    last_window = df['avg_price'].tail(window).values
    next_forecast = float(np.mean(last_window))

    result = {
        'model_name': f'Simple Moving Average (window={window})',
        'model_type': 'time_series',
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'next_period_forecast': round(next_forecast, 2),
        'historical': [
            {
                'date': row['date'].strftime('%Y-%m'),
                'actual': round(float(row['avg_price']), 2),
                'forecast': round(float(row['sma']), 2) if pd.notna(row['sma']) else None
            }
            for _, row in df.iterrows()
        ],
        'window': window
    }

    execute_db(
        "INSERT INTO model_metrics (model_name, model_type, metric_name, metric_value, parameters, dataset_size) VALUES (?,?,?,?,?,?)",
        (f'SMA_{window}', 'time_series', 'mae', mae,
         json.dumps({'window': window}), len(df))
    )

    return result


def exponential_smoothing_forecast(alpha=0.3):
    """Simple Exponential Smoothing for HDB prices."""
    df = prepare_time_series_data()
    if df is None or len(df) < 3:
        return {'error': 'Not enough time series data.'}

    prices = df['avg_price'].values
    forecasts = [prices[0]]

    for i in range(1, len(prices)):
        forecasts.append(alpha * prices[i - 1] + (1 - alpha) * forecasts[-1])

    # Next period forecast
    next_forecast = alpha * prices[-1] + (1 - alpha) * forecasts[-1]

    errors = np.array(prices[1:]) - np.array(forecasts[1:])
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    result = {
        'model_name': f'Exponential Smoothing (α={alpha})',
        'model_type': 'time_series',
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'alpha': alpha,
        'next_period_forecast': round(float(next_forecast), 2),
        'historical': [
            {
                'date': df.iloc[i]['date'].strftime('%Y-%m'),
                'actual': round(float(prices[i]), 2),
                'forecast': round(float(forecasts[i]), 2)
            }
            for i in range(len(prices))
        ]
    }

    execute_db(
        "INSERT INTO model_metrics (model_name, model_type, metric_name, metric_value, parameters, dataset_size) VALUES (?,?,?,?,?,?)",
        (f'ExpSmoothing_{alpha}', 'time_series', 'mae', mae,
         json.dumps({'alpha': alpha}), len(df))
    )

    return result


def linear_trend_forecast():
    """Linear trend decomposition and forecast."""
    df = prepare_time_series_data()
    if df is None or len(df) < 3:
        return {'error': 'Not enough time series data.'}

    prices = df['avg_price'].values
    x = np.arange(len(prices))

    # Fit linear trend
    coeffs = np.polyfit(x, prices, 1)
    trend = np.polyval(coeffs, x)

    # Forecast next period
    next_x = len(prices)
    next_forecast = float(np.polyval(coeffs, next_x))

    residuals = prices - trend
    mae = float(np.mean(np.abs(residuals)))

    result = {
        'model_name': 'Linear Trend',
        'model_type': 'time_series',
        'slope': round(float(coeffs[0]), 2),
        'intercept': round(float(coeffs[1]), 2),
        'mae': round(mae, 2),
        'next_period_forecast': round(next_forecast, 2),
        'historical': [
            {
                'date': df.iloc[i]['date'].strftime('%Y-%m'),
                'actual': round(float(prices[i]), 2),
                'trend': round(float(trend[i]), 2)
            }
            for i in range(len(prices))
        ]
    }

    return result


def get_transport_time_series():
    """Get transport usage time series for visualization."""
    rows = query_db("""
        SELECT s.station_name, s.line_name, t.year, t.month,
               u.total_trips, u.peak_hour_pct
        FROM fact_transport_usage u
        JOIN dim_transport_station s ON u.station_id = s.station_id
        JOIN dim_time t ON u.time_id = t.time_id
        ORDER BY s.station_name, t.year, t.month
    """)
    return [dict(r) for r in rows] if rows else []
