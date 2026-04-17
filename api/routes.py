"""REST API endpoints for Data Science Lab."""
import json
import os
import traceback
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from models.database import query_db, execute_db, get_all_tables, get_row_count
from models.ml_models import (
    train_linear_regression, train_random_forest, train_kmeans_clustering,
    predict_hdb_price, get_model_history
)
from models.timeseries_models import (
    simple_moving_average_forecast, exponential_smoothing_forecast, linear_trend_forecast
)
from models.nlp_models import train_text_classifier, train_sentiment_classifier, classify_text
from services.pipeline import build_full_pipeline, get_pipeline_history
from services.streaming import generate_event_batch, ingest_events, process_window, get_stream_stats
from services.validation import run_all_validations
from services.ingestion import ingest_csv_to_table, ingest_from_url
from services.meltano_elt import (
    build_hdb_elt_pipeline, build_transport_elt_pipeline,
    build_energy_elt_pipeline, build_custom_elt_pipeline,
    run_full_meltano_elt
)
from services.dagster_pipeline import (
    run_dagster_job, run_all_dagster_jobs, run_dagster_assets,
    get_dagster_asset_info, get_dagster_schedules, get_dagster_sensors
)
from services.scraping import (
    scrape_html_tables, scrape_and_load_table, scrape_page_links,
    scrape_api_json, scrape_api_and_load
)

api_bp = Blueprint('api', __name__, url_prefix='/api')


def log_api_call(endpoint, method, status_code, response_time_ms):
    try:
        execute_db(
            "INSERT INTO api_logs (endpoint, method, status_code, response_time_ms) VALUES (?,?,?,?)",
            (endpoint, method, status_code, response_time_ms)
        )
    except Exception:
        pass


@api_bp.route('/summary', methods=['GET'])
def api_summary():
    """Get overall data summary."""
    start = datetime.now()
    try:
        tables = get_all_tables()
        summary = {}
        for t in tables:
            try:
                summary[t] = get_row_count(t)
            except Exception:
                summary[t] = 0

        result = {
            'status': 'ok',
            'tables': summary,
            'total_tables': len(tables),
            'total_rows': sum(summary.values()),
            'timestamp': datetime.now().isoformat()
        }
        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/summary', 'GET', 200, elapsed)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/housing', methods=['GET'])
def api_housing():
    """Get data from any table (legacy housing endpoint, now generic)."""
    start = datetime.now()
    try:
        table = request.args.get('table', 'raw_hdb_resale')
        limit = request.args.get('limit', 100, type=int)

        from models.database import get_all_tables
        all_tables = get_all_tables()
        if table not in all_tables:
            return jsonify({'status': 'error', 'message': f'Table {table} not found'}), 404

        rows = query_db(f'SELECT * FROM "{table}" LIMIT ?', [limit])
        data = [dict(r) for r in rows]

        result = {
            'status': 'ok',
            'data': data,
            'count': len(data),
            'table': table,
            'available_tables': all_tables
        }
        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/housing', 'GET', 200, elapsed)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/transport', methods=['GET'])
def api_transport():
    """Get data from any table (legacy transport endpoint, now generic)."""
    start = datetime.now()
    try:
        table = request.args.get('table', 'fact_transport_usage')
        limit = request.args.get('limit', 100, type=int)

        from models.database import get_all_tables
        all_tables = get_all_tables()
        if table not in all_tables:
            return jsonify({'status': 'error', 'message': f'Table {table} not found'}), 404

        rows = query_db(f'SELECT * FROM "{table}" LIMIT ?', [limit])
        data = [dict(r) for r in rows]

        result = {
            'status': 'ok',
            'data': data,
            'count': len(data)
        }
        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/transport', 'GET', 200, elapsed)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/eda', methods=['GET'])
def api_eda():
    """Get EDA summary statistics for all loaded tables."""
    start = datetime.now()
    try:
        import pandas as pd
        import numpy as np
        from models.database import get_all_tables, get_row_count

        all_tables = get_all_tables()
        data_tables = [t for t in all_tables if not t.startswith(('model_', 'api_', 'pipeline_', 'stream_', 'sqlite_'))]

        table_stats = {}
        for t in data_tables[:10]:
            rows = query_db(f'SELECT * FROM "{t}" LIMIT 200')
            if not rows:
                continue
            df = pd.DataFrame([dict(r) for r in rows])
            numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
            stats = {
                'row_count': get_row_count(t),
                'columns': list(df.columns),
                'numeric_columns': numeric_cols,
            }
            if numeric_cols:
                desc = df[numeric_cols].describe().round(2).to_dict()
                stats['describe'] = desc
            table_stats[t] = stats

        result = {
            'status': 'ok',
            'tables': table_stats,
            'total_tables': len(data_tables)
        }
        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/eda', 'GET', 200, elapsed)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/model/train', methods=['POST'])
def api_model_train():
    """Train a machine learning model."""
    start = datetime.now()
    try:
        data = request.get_json() or {}
        model_type = data.get('model_type', 'linear_regression')

        if model_type == 'linear_regression':
            result = train_linear_regression()
        elif model_type == 'random_forest':
            result = train_random_forest()
        elif model_type == 'kmeans':
            result = train_kmeans_clustering()
        elif model_type == 'text_classifier':
            result = train_text_classifier()
        elif model_type == 'sentiment':
            result = train_sentiment_classifier()
        elif model_type == 'sma_forecast':
            window = data.get('window', 3)
            result = simple_moving_average_forecast(window)
        elif model_type == 'exp_smoothing':
            alpha = data.get('alpha', 0.3)
            result = exponential_smoothing_forecast(alpha)
        elif model_type == 'linear_trend':
            result = linear_trend_forecast()
        else:
            return jsonify({'status': 'error', 'message': f'Unknown model type: {model_type}'}), 400

        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/model/train', 'POST', 200, elapsed)
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@api_bp.route('/model/predict', methods=['POST'])
def api_model_predict():
    """Make a prediction."""
    start = datetime.now()
    try:
        data = request.get_json() or {}
        model_type = data.get('model_type', 'hdb_price')

        if model_type == 'hdb_price':
            result = predict_hdb_price(
                floor_area=data.get('floor_area', 93),
                storey=data.get('storey', 7),
                town_code=data.get('town_code', 0),
                flat_type_code=data.get('flat_type_code', 1),
                building_age=data.get('building_age', 20)
            )
        elif model_type == 'text_classify':
            text = data.get('text', '')
            if not text:
                return jsonify({'status': 'error', 'message': 'Text is required'}), 400
            result = classify_text(text)
        else:
            return jsonify({'status': 'error', 'message': f'Unknown model type: {model_type}'}), 400

        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/model/predict', 'POST', 200, elapsed)
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/pipeline/run', methods=['POST'])
def api_pipeline_run():
    """Run the full data pipeline."""
    start = datetime.now()
    try:
        pipeline = build_full_pipeline()
        result = pipeline.run()
        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/pipeline/run', 'POST', 200, elapsed)
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/pipeline/history', methods=['GET'])
def api_pipeline_history():
    """Get pipeline run history."""
    try:
        rows = get_pipeline_history()
        return jsonify({'status': 'ok', 'data': [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/feedback', methods=['GET', 'POST'])
def api_feedback():
    """Get or submit feedback."""
    if request.method == 'POST':
        data = request.get_json() or {}
        module = data.get('module_name', '')
        rating = data.get('rating', 3)
        comment = data.get('comment', '')

        if not module:
            return jsonify({'status': 'error', 'message': 'module_name is required'}), 400
        if not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({'status': 'error', 'message': 'rating must be 1-5'}), 400

        execute_db(
            "INSERT INTO user_feedback (module_name, rating, comment) VALUES (?,?,?)",
            (module, rating, comment)
        )
        return jsonify({'status': 'ok', 'message': 'Feedback submitted'})
    else:
        rows = query_db("SELECT * FROM user_feedback ORDER BY submitted_at DESC")
        return jsonify({'status': 'ok', 'data': [dict(r) for r in rows]})


@api_bp.route('/stream/generate', methods=['POST'])
def api_stream_generate():
    """Generate and ingest simulated stream events."""
    try:
        data = request.get_json() or {}
        batch_size = data.get('batch_size', 10)
        events = generate_event_batch(batch_size)
        count = ingest_events(events)
        return jsonify({'status': 'ok', 'events_generated': count})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/stream/process', methods=['POST'])
def api_stream_process():
    """Process stream events within a window."""
    try:
        data = request.get_json() or {}
        window = data.get('window_minutes', 60)
        result = process_window(window)
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/stream/stats', methods=['GET'])
def api_stream_stats():
    """Get stream processing stats."""
    try:
        stats = get_stream_stats()
        return jsonify({'status': 'ok', 'stats': stats})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/validate', methods=['GET'])
def api_validate():
    """Run data validation checks."""
    try:
        results = run_all_validations()
        # Convert any non-serializable types
        serializable = json.loads(json.dumps(results, default=str))
        return jsonify({'status': 'ok', 'validations': serializable})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/sql/execute', methods=['POST'])
def api_sql_execute():
    """Execute a read-only SQL query (for learning purposes)."""
    try:
        data = request.get_json() or {}
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'status': 'error', 'message': 'Query is required'}), 400

        # Safety: only allow SELECT, PRAGMA, EXPLAIN
        upper = query.upper().lstrip()
        allowed_prefixes = ('SELECT', 'PRAGMA', 'EXPLAIN', 'WITH')
        if not any(upper.startswith(p) for p in allowed_prefixes):
            return jsonify({
                'status': 'error',
                'message': 'Only SELECT, PRAGMA, EXPLAIN, and WITH (CTE) queries are allowed in this learning environment.'
            }), 400

        rows = query_db(query)
        data_out = [dict(r) for r in rows[:500]]  # Limit to 500 rows

        return jsonify({
            'status': 'ok',
            'data': data_out,
            'row_count': len(data_out),
            'truncated': len(rows) > 500
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@api_bp.route('/model/history', methods=['GET'])
def api_model_history():
    """Get model training history."""
    try:
        history = get_model_history()
        return jsonify({'status': 'ok', 'data': history})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/ingest/upload', methods=['POST'])
def api_ingest_upload():
    """Upload a CSV file and load it into a database table."""
    start = datetime.now()
    try:
        file = request.files.get('file')
        table_name = request.form.get('table_name', 'uploaded_data')

        if not file or not file.filename:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400

        upload_dir = current_app.config['UPLOAD_FOLDER']
        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)

        count, cols = ingest_csv_to_table(filepath, table_name)

        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/ingest/upload', 'POST', 200, elapsed)
        return jsonify({
            'status': 'ok',
            'rows_loaded': count,
            'columns': cols,
            'table_name': table_name
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/ingest/url', methods=['POST'])
def api_ingest_url():
    """Ingest an external dataset from a URL (CSV, JSON, or Excel)."""
    start = datetime.now()
    try:
        data = request.get_json() or {}
        url = data.get('url', '').strip()
        table_name = data.get('table_name', 'external_data')

        if not url:
            return jsonify({'status': 'error', 'message': 'URL is required'}), 400

        upload_dir = current_app.config['UPLOAD_FOLDER']
        count, cols = ingest_from_url(url, table_name, upload_dir)

        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/ingest/url', 'POST', 200, elapsed)
        return jsonify({
            'status': 'ok',
            'rows_loaded': count,
            'columns': cols,
            'table_name': table_name,
            'source_url': url
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/meltano/run', methods=['POST'])
def api_meltano_run():
    """Run a Meltano ELT pipeline (hdb, transport, energy, or full)."""
    start = datetime.now()
    try:
        data = request.get_json() or {}
        pipeline_name = data.get('pipeline', 'full')

        builders = {
            'hdb': build_hdb_elt_pipeline,
            'transport': build_transport_elt_pipeline,
            'energy': build_energy_elt_pipeline,
        }

        if pipeline_name == 'full':
            result = run_full_meltano_elt()
        elif pipeline_name in builders:
            pipeline = builders[pipeline_name]()
            result = pipeline.run()
        else:
            return jsonify({'status': 'error', 'message': f'Unknown pipeline: {pipeline_name}'}), 400

        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/meltano/run', 'POST', 200, elapsed)
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@api_bp.route('/meltano/custom', methods=['POST'])
def api_meltano_custom():
    """Run a custom Meltano ELT pipeline with user-defined source and transforms."""
    start = datetime.now()
    try:
        data = request.get_json() or {}
        source_file = data.get('source_file', '')
        table_name = data.get('table_name', 'raw_custom_elt')
        transforms = data.get('transforms', [])

        if not source_file:
            return jsonify({'status': 'error', 'message': 'source_file is required'}), 400

        # Resolve source path safely within data directory
        from config import Config
        source_path = os.path.join(Config.DATA_DIR, os.path.basename(source_file))
        if not os.path.isfile(source_path):
            return jsonify({'status': 'error', 'message': f'File not found: {source_file}'}), 404

        pipeline = build_custom_elt_pipeline(source_path, table_name, 'csv', transforms)
        result = pipeline.run()

        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/meltano/custom', 'POST', 200, elapsed)
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@api_bp.route('/dagster/job/run', methods=['POST'])
def api_dagster_job_run():
    """Run a Dagster job (hdb, transport, energy, or all)."""
    start = datetime.now()
    try:
        data = request.get_json() or {}
        job_name = data.get('job', 'all')

        if job_name == 'all':
            result = run_all_dagster_jobs()
        else:
            result = run_dagster_job(job_name)

        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/dagster/job/run', 'POST', 200, elapsed)
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@api_bp.route('/dagster/assets/materialize', methods=['POST'])
def api_dagster_assets_materialize():
    """Materialize Dagster software-defined assets (all or a specific one)."""
    start = datetime.now()
    try:
        data = request.get_json() or {}
        asset_key = data.get('asset_key')  # None = materialize all

        result = run_dagster_assets(asset_key)

        elapsed = (datetime.now() - start).total_seconds() * 1000
        log_api_call('/api/dagster/assets/materialize', 'POST', 200, elapsed)
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@api_bp.route('/dagster/assets', methods=['GET'])
def api_dagster_assets_info():
    """Get metadata about all defined Dagster assets."""
    try:
        info = get_dagster_asset_info()
        return jsonify({'status': 'ok', 'assets': info})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/dagster/schedules', methods=['GET'])
def api_dagster_schedules():
    """Get all Dagster schedule definitions."""
    try:
        schedules = get_dagster_schedules()
        return jsonify({'status': 'ok', 'schedules': schedules})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/dagster/sensors', methods=['GET'])
def api_dagster_sensors():
    """Get all Dagster sensor definitions."""
    try:
        sensors = get_dagster_sensors()
        return jsonify({'status': 'ok', 'sensors': sensors})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ===================== SCRAPING ENDPOINTS =====================

@api_bp.route('/scrape/tables', methods=['POST'])
def api_scrape_tables():
    """Preview HTML tables from a URL."""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        table_index = data.get('table_index')
        if not url:
            return jsonify({'status': 'error', 'message': 'URL is required'}), 400
        if table_index is not None:
            table_index = int(table_index)
        result = scrape_html_tables(url, table_index=table_index)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/scrape/tables/load', methods=['POST'])
def api_scrape_tables_load():
    """Scrape an HTML table and load into the database."""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        table_index = data.get('table_index', 0)
        table_name = data.get('table_name', 'scraped_data')
        if not url:
            return jsonify({'status': 'error', 'message': 'URL is required'}), 400
        result = scrape_and_load_table(url, int(table_index), table_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/scrape/links', methods=['POST'])
def api_scrape_links():
    """Discover links on a page."""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        pattern = data.get('pattern')
        if not url:
            return jsonify({'status': 'error', 'message': 'URL is required'}), 400
        result = scrape_page_links(url, pattern=pattern)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/scrape/api', methods=['POST'])
def api_scrape_api():
    """Preview data from a JSON API."""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        params = data.get('params')
        if not url:
            return jsonify({'status': 'error', 'message': 'URL is required'}), 400
        result = scrape_api_json(url, params=params)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/scrape/api/load', methods=['POST'])
def api_scrape_api_load():
    """Fetch JSON API data and load into the database."""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        table_name = data.get('table_name', 'api_data')
        params = data.get('params')
        if not url:
            return jsonify({'status': 'error', 'message': 'URL is required'}), 400
        result = scrape_api_and_load(url, table_name, params=params)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
