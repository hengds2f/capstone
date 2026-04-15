"""Data validation utilities."""
import pandas as pd
from models.database import query_db, get_connection


def check_table_integrity(table_name):
    """Run integrity checks on a table."""
    results = {}
    try:
        rows = query_db(f"SELECT COUNT(*) as cnt FROM {table_name}")
        results['row_count'] = rows[0]['cnt']

        conn = get_connection()
        pragma = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        results['columns'] = [dict(r) for r in pragma]
        results['column_count'] = len(pragma)

        # Check for nulls in NOT NULL columns
        null_issues = []
        for col in pragma:
            if col['notnull']:
                null_count = conn.execute(
                    f"SELECT COUNT(*) as cnt FROM {table_name} WHERE {col['name']} IS NULL"
                ).fetchone()['cnt']
                if null_count > 0:
                    null_issues.append({'column': col['name'], 'null_count': null_count})
        results['null_violations'] = null_issues

        conn.close()
        results['status'] = 'pass' if not null_issues else 'warn'
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
    return results


def validate_hdb_prices():
    """Validate HDB resale price data quality."""
    checks = []
    rows = query_db("SELECT COUNT(*) as cnt FROM raw_hdb_resale WHERE resale_price <= 0")
    checks.append({
        'check': 'Non-positive prices',
        'result': 'pass' if rows[0]['cnt'] == 0 else 'fail',
        'detail': f"{rows[0]['cnt']} rows with price <= 0"
    })

    rows = query_db("SELECT COUNT(*) as cnt FROM raw_hdb_resale WHERE floor_area_sqm <= 0 OR floor_area_sqm > 300")
    checks.append({
        'check': 'Invalid floor area',
        'result': 'pass' if rows[0]['cnt'] == 0 else 'fail',
        'detail': f"{rows[0]['cnt']} rows with invalid area"
    })

    rows = query_db("SELECT COUNT(*) as cnt FROM raw_hdb_resale WHERE town IS NULL OR town = ''")
    checks.append({
        'check': 'Missing town',
        'result': 'pass' if rows[0]['cnt'] == 0 else 'fail',
        'detail': f"{rows[0]['cnt']} rows missing town"
    })

    rows = query_db("SELECT COUNT(DISTINCT town) as cnt FROM raw_hdb_resale")
    checks.append({
        'check': 'Town diversity',
        'result': 'pass' if rows[0]['cnt'] >= 5 else 'warn',
        'detail': f"{rows[0]['cnt']} distinct towns"
    })

    return checks


def validate_schema_match(df, expected_dtypes):
    """Validate DataFrame dtypes against expected schema."""
    mismatches = []
    for col, expected in expected_dtypes.items():
        if col not in df.columns:
            mismatches.append({'column': col, 'expected': expected, 'actual': 'MISSING'})
        else:
            actual = str(df[col].dtype)
            if expected not in actual:
                mismatches.append({'column': col, 'expected': expected, 'actual': actual})
    return {'valid': len(mismatches) == 0, 'mismatches': mismatches}


def run_all_validations():
    """Run all validation checks and return summary."""
    results = {}
    tables = ['raw_hdb_resale', 'dim_location', 'dim_time', 'dim_property_type',
              'dim_transport_station', 'fact_hdb_transactions', 'fact_transport_usage',
              'fact_population', 'dim_school', 'fact_school_enrollment', 'fact_energy']
    for table in tables:
        try:
            results[table] = check_table_integrity(table)
        except Exception as e:
            results[table] = {'status': 'error', 'error': str(e)}

    try:
        results['hdb_price_quality'] = validate_hdb_prices()
    except Exception as e:
        results['hdb_price_quality'] = {'status': 'error', 'error': str(e)}

    return results
