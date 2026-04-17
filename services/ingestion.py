"""Data ingestion service: CSV loading, API mock, schema validation."""
import pandas as pd
import os
import json
import requests
from datetime import datetime
from urllib.parse import urlparse
from models.database import get_connection, execute_db


def ingest_csv_to_table(csv_path, table_name, if_exists='append'):
    """Load a CSV file into a SQLite table."""
    df = pd.read_csv(csv_path)
    conn = get_connection()
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        execute_db(
            "INSERT INTO pipeline_runs (pipeline_name, step_name, status, started_at, completed_at, rows_processed) VALUES (?,?,?,?,?,?)",
            ('csv_ingestion', f'load_{table_name}', 'completed', datetime.now().isoformat(), datetime.now().isoformat(), len(df))
        )
        return len(df), list(df.columns)
    finally:
        conn.close()


def ingest_from_url(url, table_name, save_dir, if_exists='append'):
    """Download a CSV/JSON/Excel file from an external URL and load into a SQLite table."""
    parsed = urlparse(url)
    if not parsed.scheme in ('http', 'https'):
        raise ValueError("Only http and https URLs are supported")

    resp = requests.get(url, timeout=30, stream=True)
    resp.raise_for_status()

    content_type = resp.headers.get('Content-Type', '').lower()
    path_lower = parsed.path.lower()

    # Determine file format from extension or content type
    if path_lower.endswith('.json') or 'json' in content_type:
        fmt = 'json'
        ext = '.json'
    elif path_lower.endswith('.xlsx') or path_lower.endswith('.xls') or 'spreadsheet' in content_type:
        fmt = 'excel'
        ext = '.xlsx'
    else:
        fmt = 'csv'
        ext = '.csv'

    # Save file locally
    safe_name = table_name.replace(' ', '_').replace('/', '_')
    filename = f"{safe_name}{ext}"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    # Read into DataFrame
    if fmt == 'json':
        df = pd.read_json(filepath)
    elif fmt == 'excel':
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError("Downloaded file contains no data")

    conn = get_connection()
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        execute_db(
            "INSERT INTO pipeline_runs (pipeline_name, step_name, status, started_at, completed_at, rows_processed) VALUES (?,?,?,?,?,?)",
            ('url_ingestion', f'load_{table_name}', 'completed', datetime.now().isoformat(), datetime.now().isoformat(), len(df))
        )
        return len(df), list(df.columns)
    finally:
        conn.close()


def ingest_hdb_data(csv_path):
    """Ingest HDB resale data from CSV into raw and dimensional tables."""
    df = pd.read_csv(csv_path)

    conn = get_connection()
    try:
        # Load raw data
        df.to_sql('raw_hdb_resale', conn, if_exists='replace', index=False)

        # Populate dim_time
        months = df['month'].unique()
        for m in months:
            parts = m.split('-')
            year, month = int(parts[0]), int(parts[1])
            month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO dim_time (full_date, year, quarter, month, month_name) VALUES (?,?,?,?,?)",
                    (f"{year}-{month:02d}-01", year, (month - 1) // 3 + 1, month, month_names[month])
                )
            except Exception:
                pass

        # Populate dim_property_type
        prop_cols = df[['flat_type', 'flat_model', 'lease_commence_date', 'remaining_lease']].drop_duplicates()
        for _, row in prop_cols.iterrows():
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO dim_property_type (flat_type, flat_model, lease_commence_year, remaining_lease) VALUES (?,?,?,?)",
                    (row['flat_type'], row['flat_model'], int(row['lease_commence_date']), row['remaining_lease'])
                )
            except Exception:
                pass

        # Populate dim_location
        loc_cols = df[['town', 'flat_type', 'block', 'street_name', 'storey_range', 'floor_area_sqm']].drop_duplicates()
        for _, row in loc_cols.iterrows():
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO dim_location (town, flat_type, block, street_name, storey_range, floor_area_sqm, planning_area, region) VALUES (?,?,?,?,?,?,?,?)",
                    (row['town'], row['flat_type'], row['block'], row['street_name'], row['storey_range'], row['floor_area_sqm'], row['town'], 'Region')
                )
            except Exception:
                pass

        # Populate fact_hdb_transactions
        for _, row in df.iterrows():
            time_row = conn.execute("SELECT time_id FROM dim_time WHERE year=? AND month=?",
                                    (int(row['month'].split('-')[0]), int(row['month'].split('-')[1]))).fetchone()
            loc_row = conn.execute(
                "SELECT location_id FROM dim_location WHERE town=? AND block=? AND street_name=? LIMIT 1",
                (row['town'], row['block'], row['street_name'])
            ).fetchone()
            prop_row = conn.execute(
                "SELECT property_type_id FROM dim_property_type WHERE flat_type=? AND flat_model=? LIMIT 1",
                (row['flat_type'], row['flat_model'])
            ).fetchone()

            if time_row and loc_row and prop_row:
                price_per_sqm = round(row['resale_price'] / row['floor_area_sqm'], 2) if row['floor_area_sqm'] > 0 else 0
                conn.execute(
                    "INSERT INTO fact_hdb_transactions (location_id, time_id, property_type_id, resale_price, price_per_sqm) VALUES (?,?,?,?,?)",
                    (loc_row[0], time_row[0], prop_row[0], row['resale_price'], price_per_sqm)
                )

        conn.commit()
        return len(df)
    finally:
        conn.close()


def ingest_transport_data(csv_path):
    """Ingest transport data."""
    df = pd.read_csv(csv_path)
    conn = get_connection()
    try:
        # Populate dim_transport_station
        stations = df[['station_name', 'station_code', 'line_name', 'line_color',
                        'latitude', 'longitude', 'opening_year', 'is_interchange']].drop_duplicates(subset=['station_code'])
        for _, row in stations.iterrows():
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO dim_transport_station (station_name, station_code, line_name, line_color, latitude, longitude, opening_year, is_interchange) VALUES (?,?,?,?,?,?,?,?)",
                    tuple(row)
                )
            except Exception:
                pass

        # Populate fact_transport_usage
        for _, row in df.iterrows():
            station = conn.execute("SELECT station_id FROM dim_transport_station WHERE station_code=?",
                                   (row['station_code'],)).fetchone()
            time_row = conn.execute("SELECT time_id FROM dim_time WHERE year=? AND month=?",
                                    (int(row['year']), int(row['month']))).fetchone()
            if not time_row:
                conn.execute(
                    "INSERT OR IGNORE INTO dim_time (full_date, year, quarter, month, month_name) VALUES (?,?,?,?,?)",
                    (f"{int(row['year'])}-{int(row['month']):02d}-01", int(row['year']),
                     (int(row['month']) - 1) // 3 + 1, int(row['month']), '')
                )
                conn.commit()
                time_row = conn.execute("SELECT time_id FROM dim_time WHERE year=? AND month=?",
                                        (int(row['year']), int(row['month']))).fetchone()

            if station and time_row:
                total = int(row['tap_in_count']) + int(row['tap_out_count'])
                conn.execute(
                    "INSERT INTO fact_transport_usage (station_id, time_id, tap_in_count, tap_out_count, total_trips, peak_hour_pct) VALUES (?,?,?,?,?,?)",
                    (station[0], time_row[0], int(row['tap_in_count']), int(row['tap_out_count']), total, row['peak_hour_pct'])
                )

        conn.commit()
        return len(df)
    finally:
        conn.close()


def ingest_population_data(csv_path):
    """Ingest population data."""
    df = pd.read_csv(csv_path)
    conn = get_connection()
    try:
        for _, row in df.iterrows():
            conn.execute(
                "INSERT INTO fact_population (planning_area, year, age_group, gender, population_count, density_per_sqkm) VALUES (?,?,?,?,?,?)",
                (row['planning_area'], int(row['year']), row['age_group'], row['gender'],
                 int(row['population_count']), float(row['density_per_sqkm']))
            )
        conn.commit()
        return len(df)
    finally:
        conn.close()


def ingest_school_data(csv_path):
    """Ingest school data."""
    df = pd.read_csv(csv_path)
    conn = get_connection()
    try:
        for _, row in df.iterrows():
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO dim_school (school_name, school_type, zone, cluster, address, postal_code, latitude, longitude) VALUES (?,?,?,?,?,?,?,?)",
                    (row['school_name'], row['school_type'], row['zone'], row['cluster'],
                     row['address'], str(row['postal_code']), row['latitude'], row['longitude'])
                )
            except Exception:
                pass

            school = conn.execute("SELECT school_id FROM dim_school WHERE school_name=?",
                                  (row['school_name'],)).fetchone()
            if school:
                conn.execute(
                    "INSERT INTO fact_school_enrollment (school_id, year, level, enrollment_count, num_classes, avg_class_size) VALUES (?,?,?,?,?,?)",
                    (school[0], int(row['year']), row['level'], int(row['enrollment_count']),
                     int(row['num_classes']), float(row['avg_class_size']))
                )
        conn.commit()
        return len(df)
    finally:
        conn.close()


def ingest_energy_data(csv_path):
    """Ingest energy data."""
    df = pd.read_csv(csv_path)
    conn = get_connection()
    try:
        for _, row in df.iterrows():
            conn.execute(
                "INSERT INTO fact_energy (year, month, sector, energy_type, consumption_gwh, cost_million_sgd, carbon_emission_tonnes) VALUES (?,?,?,?,?,?,?)",
                (int(row['year']), int(row['month']), row['sector'], row['energy_type'],
                 float(row['consumption_gwh']), float(row['cost_million_sgd']), float(row['carbon_emission_tonnes']))
            )
        conn.commit()
        return len(df)
    finally:
        conn.close()


def ingest_feedback_data(csv_path):
    """Ingest sample feedback."""
    df = pd.read_csv(csv_path)
    conn = get_connection()
    try:
        for _, row in df.iterrows():
            conn.execute(
                "INSERT INTO user_feedback (module_name, rating, comment) VALUES (?,?,?)",
                (row['module_name'], int(row['rating']), row['comment'])
            )
        conn.commit()
        return len(df)
    finally:
        conn.close()


def mock_datagov_api(dataset_name):
    """Simulate a public data API response for educational purposes."""
    mock_responses = {
        'hdb-resale': {
            'api_url': 'https://data.gov.sg/api/action/datastore_search?resource_id=f1765b54-a209-4718-8d38-a39237f502b3',
            'note': 'In production, call requests.get(api_url) and parse response JSON. Works with any REST API.',
            'sample_response': {
                'success': True,
                'result': {
                    'records': [
                        {'month': '2024-01', 'town': 'ANG MO KIO', 'flat_type': '4 ROOM', 'resale_price': '450000'}
                    ],
                    'total': 1
                }
            }
        },
        'generic': {
            'api_url': 'https://api.example.com/v1/data?format=json&limit=100',
            'note': 'Replace with any public REST API. Use your own API keys when required.',
            'sample_response': {
                'data': [
                    {'id': 1, 'name': 'Sample Record', 'value': 42.5}
                ],
                'total': 1
            }
        }
    }
    return mock_responses.get(dataset_name, mock_responses.get('generic'))


def validate_dataframe(df, expected_columns, table_name):
    """Validate DataFrame schema before loading."""
    errors = []
    missing = set(expected_columns) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
    if df.empty:
        errors.append("DataFrame is empty")
    null_counts = df.isnull().sum()
    high_null = null_counts[null_counts > len(df) * 0.5]
    if not high_null.empty:
        errors.append(f"High null rate in columns: {list(high_null.index)}")
    return {'valid': len(errors) == 0, 'errors': errors, 'rows': len(df), 'columns': list(df.columns)}
