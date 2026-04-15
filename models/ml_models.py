"""Machine learning models for Singapore data."""
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
from models.database import query_db, execute_db, get_connection


def get_hdb_training_data():
    """Prepare HDB resale data for ML training."""
    rows = query_db("""
        SELECT l.town, l.floor_area_sqm, l.storey_range,
               p.flat_type, p.lease_commence_year,
               t.year, t.month,
               f.resale_price
        FROM fact_hdb_transactions f
        JOIN dim_location l ON f.location_id = l.location_id
        JOIN dim_time t ON f.time_id = t.time_id
        JOIN dim_property_type p ON f.property_type_id = p.property_type_id
    """)
    if not rows:
        return None

    df = pd.DataFrame([dict(r) for r in rows])

    # Feature engineering
    le_town = LabelEncoder()
    df['town_encoded'] = le_town.fit_transform(df['town'])

    le_flat = LabelEncoder()
    df['flat_type_encoded'] = le_flat.fit_transform(df['flat_type'])

    # Extract mid-storey from range
    df['mid_storey'] = df['storey_range'].apply(
        lambda x: np.mean([int(s) for s in x.split(' TO ')]) if ' TO ' in str(x) else 5
    )

    # Building age
    df['building_age'] = df['year'] - df['lease_commence_year']

    feature_cols = ['floor_area_sqm', 'mid_storey', 'town_encoded',
                    'flat_type_encoded', 'building_age']
    X = df[feature_cols].values
    y = df['resale_price'].values

    return X, y, feature_cols, le_town, le_flat, df


def train_linear_regression():
    """Train linear regression on HDB resale data."""
    result = get_hdb_training_data()
    if result is None:
        return {'error': 'No training data available. Run the pipeline first.'}

    X, y, feature_cols, _, _, df = result
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'model_name': 'Linear Regression',
        'model_type': 'supervised',
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

    # Log to database
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


def train_random_forest():
    """Train Random Forest on HDB resale data."""
    result = get_hdb_training_data()
    if result is None:
        return {'error': 'No training data available. Run the pipeline first.'}

    X, y, feature_cols, _, _, df = result
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    importances = model.feature_importances_

    metrics = {
        'model_name': 'Random Forest Regressor',
        'model_type': 'supervised',
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


def train_kmeans_clustering():
    """KMeans clustering for neighborhood segmentation."""
    rows = query_db("""
        SELECT l.town,
               AVG(f.resale_price) as avg_price,
               AVG(f.price_per_sqm) as avg_price_sqm,
               AVG(l.floor_area_sqm) as avg_area,
               COUNT(*) as transaction_count
        FROM fact_hdb_transactions f
        JOIN dim_location l ON f.location_id = l.location_id
        GROUP BY l.town
        HAVING COUNT(*) >= 2
    """)
    if not rows:
        return {'error': 'Not enough data for clustering. Run the pipeline first.'}

    df = pd.DataFrame([dict(r) for r in rows])
    features = ['avg_price', 'avg_price_sqm', 'avg_area', 'transaction_count']
    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(4, len(X))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels) if n_clusters > 1 and len(X) > n_clusters else 0

    df['cluster'] = labels
    cluster_summary = {}
    for c in range(n_clusters):
        cluster_df = df[df['cluster'] == c]
        cluster_summary[f'Cluster {c}'] = {
            'towns': list(cluster_df['town'].values),
            'avg_price': round(float(cluster_df['avg_price'].mean()), 2),
            'avg_area': round(float(cluster_df['avg_area'].mean()), 2),
            'count': int(len(cluster_df))
        }

    metrics = {
        'model_name': 'KMeans Clustering',
        'model_type': 'unsupervised',
        'n_clusters': n_clusters,
        'silhouette_score': round(float(sil_score), 4),
        'cluster_summary': cluster_summary,
        'feature_names': features,
        'town_assignments': {row['town']: int(row['cluster']) for _, row in df.iterrows()}
    }

    execute_db(
        "INSERT INTO model_metrics (model_name, model_type, metric_name, metric_value, parameters, dataset_size) VALUES (?,?,?,?,?,?)",
        ('KMeans', 'unsupervised', 'silhouette_score', metrics['silhouette_score'],
         json.dumps({'n_clusters': n_clusters}), len(X))
    )

    return metrics


def predict_hdb_price(floor_area, storey, town_code, flat_type_code, building_age):
    """Make a prediction with the trained model."""
    result = get_hdb_training_data()
    if result is None:
        return {'error': 'No model data available'}

    X, y, feature_cols, _, _, _ = result
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, y)

    input_features = np.array([[floor_area, storey, town_code, flat_type_code, building_age]])
    prediction = model.predict(input_features)[0]

    return {
        'predicted_price': round(float(prediction), 2),
        'input': {
            'floor_area_sqm': floor_area,
            'mid_storey': storey,
            'town_code': town_code,
            'flat_type_code': flat_type_code,
            'building_age': building_age
        }
    }


def get_model_history():
    """Get all model training history."""
    rows = query_db("SELECT * FROM model_metrics ORDER BY trained_at DESC")
    return [dict(r) for r in rows]
