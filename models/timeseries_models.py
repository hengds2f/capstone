"""Time series forecasting models — works with any time-indexed dataset."""
import numpy as np
import pandas as pd
from models.database import query_db, execute_db, get_all_tables
import json


def prepare_time_series_data(table_name=None, date_col=None, value_col=None):
    """Prepare time series from any available table with date and numeric columns.

    Auto-detects date and value columns if not specified.
    """
    all_tables = get_all_tables()

    if table_name:
        candidates = [table_name]
    else:
        candidates = [t for t in all_tables if not t.startswith(('model_', 'api_', 'pipeline_', 'stream_', 'sqlite_'))]

    for t in candidates:
        rows = query_db(f'SELECT * FROM "{t}" LIMIT 500')
        if not rows:
            continue
        df = pd.DataFrame([dict(r) for r in rows])

        # Find date column
        if date_col and date_col in df.columns:
            dcol = date_col
        else:
            # Auto-detect: look for date-like columns
            dcol = None
            for col in df.columns:
                if any(kw in col.lower() for kw in ['date', 'time', 'month', 'year', 'period']):
                    dcol = col
                    break
            if dcol is None:
                continue

        # Find value column
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        if value_col and value_col in df.columns:
            vcol = value_col
        elif numeric_cols:
            vcol = numeric_cols[-1]  # Use last numeric column
        else:
            continue

        # Try to parse dates
        try:
            df['_parsed_date'] = pd.to_datetime(df[dcol])
            df = df.sort_values('_parsed_date')
            # Aggregate by date if needed (daily/monthly)
            agg = df.groupby('_parsed_date')[vcol].mean().reset_index()
            agg.columns = ['date', 'value']
            agg = agg.dropna()
            if len(agg) >= 3:
                return agg, t, dcol, vcol
        except Exception:
            # Try year+month approach
            if 'year' in df.columns and 'month' in df.columns:
                try:
                    df['_parsed_date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
                    agg = df.groupby('_parsed_date')[vcol].mean().reset_index()
                    agg.columns = ['date', 'value']
                    agg = agg.sort_values('date').dropna()
                    if len(agg) >= 3:
                        return agg, t, 'year+month', vcol
                except Exception:
                    pass

    return None, None, None, None


def simple_moving_average_forecast(window=3):
    """Simple Moving Average forecast on any time series."""
    df, src_table, dcol, vcol = prepare_time_series_data()
    if df is None or len(df) < window + 1:
        return {'error': 'Not enough time series data. Upload a dataset with date and numeric columns.'}

    df['sma'] = df['value'].rolling(window=window).mean()
    df['sma_forecast'] = df['sma'].shift(1)

    valid = df.dropna(subset=['sma_forecast'])
    if len(valid) > 0:
        mae = float(np.mean(np.abs(valid['value'] - valid['sma_forecast'])))
        rmse = float(np.sqrt(np.mean((valid['value'] - valid['sma_forecast'])**2)))
    else:
        mae, rmse = 0, 0

    last_window = df['value'].tail(window).values
    next_forecast = float(np.mean(last_window))

    result = {
        'model_name': f'Simple Moving Average (window={window})',
        'model_type': 'time_series',
        'source_table': src_table,
        'date_column': dcol,
        'value_column': vcol,
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'next_period_forecast': round(next_forecast, 2),
        'historical': [
            {
                'date': row['date'].strftime('%Y-%m-%d'),
                'actual': round(float(row['value']), 2),
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
    """Simple Exponential Smoothing on any time series."""
    df, src_table, dcol, vcol = prepare_time_series_data()
    if df is None or len(df) < 3:
        return {'error': 'Not enough time series data.'}

    prices = df['value'].values
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
        'source_table': src_table,
        'date_column': dcol,
        'value_column': vcol,
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'alpha': alpha,
        'next_period_forecast': round(float(next_forecast), 2),
        'historical': [
            {
                'date': df.iloc[i]['date'].strftime('%Y-%m-%d'),
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
    df, src_table, dcol, vcol = prepare_time_series_data()
    if df is None or len(df) < 3:
        return {'error': 'Not enough time series data.'}

    values = df['value'].values
    x = np.arange(len(values))

    coeffs = np.polyfit(x, values, 1)
    trend = np.polyval(coeffs, x)

    next_x = len(values)
    next_forecast = float(np.polyval(coeffs, next_x))

    residuals = values - trend
    mae = float(np.mean(np.abs(residuals)))

    result = {
        'model_name': 'Linear Trend',
        'model_type': 'time_series',
        'source_table': src_table,
        'date_column': dcol,
        'value_column': vcol,
        'slope': round(float(coeffs[0]), 2),
        'intercept': round(float(coeffs[1]), 2),
        'mae': round(mae, 2),
        'next_period_forecast': round(next_forecast, 2),
        'historical': [
            {
                'date': df.iloc[i]['date'].strftime('%Y-%m-%d'),
                'actual': round(float(values[i]), 2),
                'trend': round(float(trend[i]), 2)
            }
            for i in range(len(values))
        ]
    }

    return result


def get_transport_time_series():
    """Get time series data for visualization from any available table."""
    all_tables = get_all_tables()
    for t in all_tables:
        rows = query_db(f'SELECT * FROM "{t}" LIMIT 200')
        if rows:
            return [dict(r) for r in rows]
    return []
