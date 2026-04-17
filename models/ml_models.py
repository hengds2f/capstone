"""Machine learning models — works with any dataset."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             silhouette_score, classification_report, accuracy_score)
import json
from datetime import datetime
from models.database import query_db, execute_db, get_connection, get_all_tables


def _get_training_data(table_name=None, target_col=None):
    """Auto-detect and prepare training data from any available table.

    If table_name/target_col are not specified, automatically finds the first
    suitable table with numeric columns.
    """
    if table_name and target_col:
        rows = query_db(f'SELECT * FROM "{table_name}" LIMIT 1000')
        if not rows:
            return None
        df = pd.DataFrame([dict(r) for r in rows])
        if target_col not in df.columns:
            return None
    else:
        # Auto-detect: find first table with enough numeric columns
        all_tables = get_all_tables()
        data_tables = [t for t in all_tables if not t.startswith(('model_', 'api_', 'pipeline_', 'stream_', 'sqlite_'))]
        df = None
        for t in data_tables:
            rows = query_db(f'SELECT * FROM "{t}" LIMIT 1000')
            if not rows:
                continue
            candidate = pd.DataFrame([dict(r) for r in rows])
            num_cols = list(candidate.select_dtypes(include=[np.number]).columns)
            if len(num_cols) >= 2:
                df = candidate
                table_name = t
                target_col = num_cols[-1]  # Use last numeric column as target
                break
        if df is None:
            return None

    # Prepare features: encode categoricals, use numerics directly
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=['object']).columns)

    feature_cols = [c for c in numeric_cols if c != target_col]
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        feature_cols.append(f'{col}_encoded')
        encoders[col] = le

    if not feature_cols:
        return None

    df_clean = df.dropna(subset=feature_cols + [target_col])
    if len(df_clean) < 5:
        return None

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    return X, y, feature_cols, encoders, df_clean, table_name, target_col


def train_linear_regression(table_name=None, target_col=None):
    """Train linear regression on any available numeric data."""
    result = _get_training_data(table_name, target_col)
    if result is None:
        return {'error': 'No training data available. Upload a dataset with numeric columns first.'}

    X, y, feature_cols, encoders, df, src_table, tgt_col = result
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'model_name': 'Linear Regression',
        'model_type': 'supervised',
        'source_table': src_table,
        'target_column': tgt_col,
        'r2_score': round(r2_score(y_test, y_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        'mae': round(mean_absolute_error(y_test, y_pred), 2),
        'coefficients': {col: round(float(coef), 4) for col, coef in zip(feature_cols, model.coef_)},
        'intercept': round(float(model.intercept_), 2),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'feature_names': feature_cols,
        'sample_predictions': [
            {'actual': float(y_test[i]), 'predicted': round(float(y_pred[i]), 2)}
            for i in range(min(5, len(y_test)))
        ]
    }

    execute_db(
        "INSERT INTO model_metrics (model_name, model_type, metric_name, metric_value, parameters, dataset_size) VALUES (?,?,?,?,?,?)",
        ('Linear Regression', 'supervised', 'r2_score', metrics['r2_score'],
         json.dumps(metrics['coefficients']), len(X))
    )
    execute_db(
        "INSERT INTO model_metrics (model_name, model_type, metric_name, metric_value, parameters, dataset_size) VALUES (?,?,?,?,?,?)",
        ('Linear Regression', 'supervised', 'rmse', metrics['rmse'], None, len(X))
    )

    return metrics


def train_random_forest(table_name=None, target_col=None):
    """Train Random Forest on any available numeric data."""
    result = _get_training_data(table_name, target_col)
    if result is None:
        return {'error': 'No training data available. Upload a dataset with numeric columns first.'}

    X, y, feature_cols, encoders, df, src_table, tgt_col = result
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importances = model.feature_importances_

    metrics = {
        'model_name': 'Random Forest Regressor',
        'model_type': 'supervised',
        'source_table': src_table,
        'target_column': tgt_col,
        'r2_score': round(r2_score(y_test, y_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        'mae': round(mean_absolute_error(y_test, y_pred), 2),
        'feature_importance': {col: round(float(imp), 4) for col, imp in zip(feature_cols, importances)},
        'n_estimators': 50,
        'max_depth': 10,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'sample_predictions': [
            {'actual': float(y_test[i]), 'predicted': round(float(y_pred[i]), 2)}
            for i in range(min(5, len(y_test)))
        ]
    }

    execute_db(
        "INSERT INTO model_metrics (model_name, model_type, metric_name, metric_value, parameters, dataset_size) VALUES (?,?,?,?,?,?)",
        ('Random Forest', 'supervised', 'r2_score', metrics['r2_score'],
         json.dumps(metrics['feature_importance']), len(X))
    )

    return metrics


def train_kmeans_clustering(table_name=None):
    """KMeans clustering on any table with numeric columns."""
    all_tables = get_all_tables()
    data_tables = [t for t in all_tables if not t.startswith(('model_', 'api_', 'pipeline_', 'stream_', 'sqlite_'))]

    if table_name:
        data_tables = [table_name]

    df = None
    src_table = None
    for t in data_tables:
        rows = query_db(f'SELECT * FROM "{t}" LIMIT 500')
        if not rows:
            continue
        candidate = pd.DataFrame([dict(r) for r in rows])
        num_cols = list(candidate.select_dtypes(include=[np.number]).columns)
        if len(num_cols) >= 2:
            df = candidate
            src_table = t
            break

    if df is None:
        return {'error': 'Not enough numeric data for clustering. Upload a dataset first.'}

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(include=['object']).columns)
    features = numeric_cols[:6]  # Use up to 6 numeric features

    X = df[features].dropna().values
    if len(X) < 4:
        return {'error': 'Not enough rows for clustering.'}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(4, len(X) - 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels) if n_clusters > 1 and len(X) > n_clusters else 0

    df_clean = df[features].dropna().copy()
    df_clean['cluster'] = labels

    # Build cluster summary
    label_col = cat_cols[0] if cat_cols else None
    cluster_summary = {}
    for c in range(n_clusters):
        cluster_df = df_clean[df_clean['cluster'] == c]
        summary = {
            'size': int(len(cluster_df)),
            'means': {f: round(float(cluster_df[f].mean()), 2) for f in features}
        }
        if label_col and label_col in df.columns:
            summary['labels'] = list(df.loc[cluster_df.index, label_col].head(5).values)
        cluster_summary[f'Cluster {c}'] = summary

    metrics = {
        'model_name': 'KMeans Clustering',
        'model_type': 'unsupervised',
        'source_table': src_table,
        'n_clusters': n_clusters,
        'silhouette_score': round(float(sil_score), 4),
        'cluster_summary': cluster_summary,
        'feature_names': features,
    }

    execute_db(
        "INSERT INTO model_metrics (model_name, model_type, metric_name, metric_value, parameters, dataset_size) VALUES (?,?,?,?,?,?)",
        ('KMeans', 'unsupervised', 'silhouette_score', metrics['silhouette_score'],
         json.dumps({'n_clusters': n_clusters}), len(X))
    )

    return metrics


def predict_hdb_price(floor_area, storey, town_code, flat_type_code, building_age):
    """Make a prediction with the trained model (legacy compatibility).
    For new datasets, use the API endpoint with table_name and target_col."""
    result = _get_training_data()
    if result is None:
        return {'error': 'No model data available'}

    X, y, feature_cols, _, _, src_table, tgt_col = result
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, y)

    # Pad or truncate input to match feature count
    input_vals = [floor_area, storey, town_code, flat_type_code, building_age]
    if len(input_vals) < len(feature_cols):
        input_vals.extend([0] * (len(feature_cols) - len(input_vals)))
    input_vals = input_vals[:len(feature_cols)]

    input_features = np.array([input_vals])
    prediction = model.predict(input_features)[0]

    return {
        'predicted_value': round(float(prediction), 2),
        'source_table': src_table,
        'target_column': tgt_col,
        'input': dict(zip(feature_cols, input_vals))
    }


def get_model_history():
    """Get all model training history."""
    rows = query_db("SELECT * FROM model_metrics ORDER BY trained_at DESC")
    return [dict(r) for r in rows]
