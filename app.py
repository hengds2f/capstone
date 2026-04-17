"""Data Science Lab - Flask Application Factory."""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from config import Config
from models.database import init_db, query_db, execute_db, get_all_tables, get_row_count, get_connection
from services.ingestion import (
    ingest_hdb_data, ingest_transport_data, ingest_population_data,
    ingest_school_data, ingest_energy_data, ingest_feedback_data,
    mock_datagov_api, validate_dataframe, ingest_csv_to_table, ingest_from_url
)
from services.scraping import scrape_html_tables, scrape_and_load_table, scrape_api_json, scrape_api_and_load
from services.pipeline import build_full_pipeline, get_pipeline_history
from services.validation import run_all_validations
from services.streaming import generate_event_batch, ingest_events, process_window, get_stream_stats
from models.ml_models import (
    train_linear_regression, train_random_forest, train_kmeans_clustering,
    predict_hdb_price, get_model_history
)
from models.timeseries_models import (
    simple_moving_average_forecast, exponential_smoothing_forecast,
    linear_trend_forecast, get_transport_time_series
)
from models.nlp_models import train_text_classifier, train_sentiment_classifier, classify_text
from models.cv_models import (
    generate_sample_image, preprocess_image, extract_color_histogram,
    simple_image_classifier, neural_network_pseudocode, image_to_base64
)
from api.routes import api_bp
import plotly
import plotly.express as px
import plotly.graph_objects as go
import io


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    os.makedirs(os.path.join(app.root_path, 'database'), exist_ok=True)
    os.makedirs(os.path.join(app.root_path, 'uploads'), exist_ok=True)

    # Initialize database
    with app.app_context():
        init_db()

    # Register API blueprint
    app.register_blueprint(api_bp)

    # ===================== PAGE ROUTES =====================

    @app.route('/')
    def index():
        """Home dashboard."""
        stats = {}
        try:
            tables = get_all_tables()
            for t in tables:
                try:
                    stats[t] = get_row_count(t)
                except Exception:
                    stats[t] = 0
        except Exception:
            pass

        modules = [
            {'name': 'Data Ingestion', 'url': '/ingestion', 'icon': '📥', 'desc': 'Load and validate datasets from any source'},
            {'name': 'SQL Learning', 'url': '/sql', 'icon': '🗄️', 'desc': 'DDL, DML, joins, CTEs, window functions'},
            {'name': 'Python Wrangling', 'url': '/wrangling', 'icon': '🐍', 'desc': 'Pandas & NumPy data manipulation'},
            {'name': 'EDA', 'url': '/eda', 'icon': '🔍', 'desc': 'Exploratory data analysis & correlations'},
            {'name': 'Visualization', 'url': '/visualization', 'icon': '📊', 'desc': 'Interactive Plotly charts gallery'},
            {'name': 'Big Data', 'url': '/bigdata', 'icon': '🏗️', 'desc': 'Big data and engineering concepts'},
            {'name': 'Architecture', 'url': '/architecture', 'icon': '🏛️', 'desc': 'Data architecture & modeling'},
            {'name': 'Data Flow', 'url': '/dataflow', 'icon': '🔄', 'desc': 'Encoding, serialization & data flow'},
            {'name': 'Web Scraping', 'url': '/webscraping', 'icon': '🕸️', 'desc': 'Data extraction techniques'},
            {'name': 'Data Warehouse', 'url': '/warehouse', 'icon': '🏬', 'desc': 'Warehouse design & star schema'},
            {'name': 'Pipeline', 'url': '/pipeline', 'icon': '⚙️', 'desc': 'ETL pipelines & orchestration'},
            {'name': 'Out-of-Core', 'url': '/outofcore', 'icon': '💾', 'desc': 'Memory-efficient processing'},
            {'name': 'Distributed', 'url': '/distributed', 'icon': '🌐', 'desc': 'Batch processing at scale'},
            {'name': 'Streaming', 'url': '/streaming', 'icon': '📡', 'desc': 'Real-time event processing'},
            {'name': 'Probability', 'url': '/probability', 'icon': '🎲', 'desc': 'Statistics for machine learning'},
            {'name': 'Supervised ML', 'url': '/supervised', 'icon': '🎯', 'desc': 'Regression & classification'},
            {'name': 'Unsupervised ML', 'url': '/unsupervised', 'icon': '🧩', 'desc': 'Clustering & segmentation'},
            {'name': 'Time Series', 'url': '/timeseries', 'icon': '📈', 'desc': 'Forecasting & trend analysis'},
            {'name': 'Neural Networks', 'url': '/neuralnet', 'icon': '🧠', 'desc': 'Deep learning fundamentals'},
            {'name': 'Computer Vision', 'url': '/cv', 'icon': '👁️', 'desc': 'Image processing & classification'},
            {'name': 'NLP', 'url': '/nlp', 'icon': '💬', 'desc': 'Text classification & sentiment'},
            {'name': 'Deployment', 'url': '/deployment', 'icon': '🚀', 'desc': 'Flask API & full-stack deployment'},
            {'name': 'Admin', 'url': '/admin', 'icon': '🔧', 'desc': 'Testing & validation dashboard'},
        ]

        return render_template('index.html', stats=stats, modules=modules)

    @app.route('/ingestion', methods=['GET', 'POST'])
    def ingestion():
        """Data ingestion page."""
        message = None
        if request.method == 'POST':
            action = request.form.get('action')
            if action == 'load_all':
                try:
                    pipeline = build_full_pipeline()
                    result = pipeline.run()
                    message = f"Pipeline completed. Results: {json.dumps({k: v['status'] for k, v in result.items()})}"
                except Exception as e:
                    message = f"Error: {str(e)}"
            elif action == 'upload_csv':
                file = request.files.get('csv_file')
                table = request.form.get('table_name', 'uploaded_data')
                if file and file.filename:
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(filepath)
                    try:
                        count, cols = ingest_csv_to_table(filepath, table)
                        message = f"Loaded {count} rows into '{table}'. Columns: {cols}"
                    except Exception as e:
                        message = f"Error: {str(e)}"
            elif action == 'ingest_url':
                url = request.form.get('dataset_url', '').strip()
                table = request.form.get('url_table_name', 'external_data')
                if url:
                    try:
                        count, cols = ingest_from_url(url, table, app.config['UPLOAD_FOLDER'])
                        message = f"Downloaded and loaded {count} rows into '{table}'. Columns: {cols}"
                    except Exception as e:
                        message = f"Error ingesting from URL: {str(e)}"
                else:
                    message = "Please provide a valid URL."
            elif action == 'scrape_table':
                url = request.form.get('scrape_url', '').strip()
                table_index = int(request.form.get('scrape_table_index', 0))
                table_name = request.form.get('scrape_table_name', 'scraped_data').strip()
                if url:
                    try:
                        result = scrape_and_load_table(url, table_index, table_name)
                        message = f"Scraped and loaded {result['rows_loaded']} rows into '{table_name}' from table #{table_index}. Columns: {result['columns']}"
                    except Exception as e:
                        message = f"Error scraping table: {str(e)}"
                else:
                    message = "Please provide a URL to scrape."
            elif action == 'scrape_api':
                url = request.form.get('api_scrape_url', '').strip()
                table_name = request.form.get('api_table_name', 'api_data').strip()
                if url:
                    try:
                        result = scrape_api_and_load(url, table_name)
                        message = f"Fetched API data and loaded {result['rows_loaded']} rows into '{table_name}'. Columns: {result['columns']}"
                    except Exception as e:
                        message = f"Error scraping API: {str(e)}"
                else:
                    message = "Please provide an API URL."

        mock_api = mock_datagov_api('hdb-resale')
        tables = get_all_tables()
        table_counts = {t: get_row_count(t) for t in tables}
        return render_template('ingestion.html', message=message, mock_api=mock_api,
                               tables=table_counts)

    @app.route('/sql')
    def sql_learning():
        """SQL learning module."""
        return render_template('sql_learning.html')

    @app.route('/wrangling')
    def wrangling():
        """Python data wrangling module."""
        # Prepare sample data for demonstration
        demo_data = None
        try:
            all_tables = get_all_tables()
            data_tables = [t for t in all_tables if not t.startswith(('model_', 'api_', 'pipeline_', 'stream_', 'sqlite_'))]
            for t in data_tables:
                rows = query_db(f'SELECT * FROM "{t}" LIMIT 20')
                if rows:
                    df = pd.DataFrame([dict(r) for r in rows])
                    demo_data = {
                        'head': df.head().to_html(classes='table table-striped', index=False),
                        'dtypes': df.dtypes.to_dict(),
                        'shape': df.shape,
                        'describe': df.describe().round(2).to_html(classes='table table-striped'),
                        'null_counts': df.isnull().sum().to_dict()
                    }
                    # Convert dtype objects to strings
                    demo_data['dtypes'] = {k: str(v) for k, v in demo_data['dtypes'].items()}
                    break
        except Exception:
            pass
        return render_template('python_wrangling.html', demo_data=demo_data)

    @app.route('/eda')
    def eda():
        """EDA module — works with any loaded dataset."""
        charts = {}
        stats = {}
        try:
            # Find all user tables
            all_tables = get_all_tables()
            data_tables = [t for t in all_tables if not t.startswith(('model_', 'api_', 'pipeline_', 'stream_', 'dim_', 'fact_', 'v_', 'sqlite_'))]
            if not data_tables:
                data_tables = [t for t in all_tables if not t.startswith(('sqlite_',))]

            # Use first available data table for EDA
            target_table = None
            df_main = None
            for t in data_tables:
                rows = query_db(f'SELECT * FROM "{t}" LIMIT 500')
                if rows:
                    target_table = t
                    df_main = pd.DataFrame([dict(r) for r in rows])
                    break

            if df_main is not None and not df_main.empty:
                numeric_cols = list(df_main.select_dtypes(include=[np.number]).columns)
                cat_cols = list(df_main.select_dtypes(include=['object']).columns)

                # Chart 1: Bar chart of first categorical vs first numeric
                if cat_cols and numeric_cols:
                    agg = df_main.groupby(cat_cols[0])[numeric_cols[0]].mean().reset_index()
                    agg.columns = [cat_cols[0], f'avg_{numeric_cols[0]}']
                    fig = px.bar(agg, x=cat_cols[0], y=f'avg_{numeric_cols[0]}',
                                 title=f'Average {numeric_cols[0]} by {cat_cols[0]}',
                                 labels={f'avg_{numeric_cols[0]}': f'Average {numeric_cols[0]}', cat_cols[0]: cat_cols[0]})
                    fig.update_layout(xaxis_tickangle=-45, height=500)
                    charts['bar_chart'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                # Chart 2: Distribution of first numeric column
                if numeric_cols:
                    color_col = cat_cols[0] if cat_cols else None
                    fig2 = px.histogram(df_main, x=numeric_cols[0], color=color_col,
                                        title=f'Distribution of {numeric_cols[0]}',
                                        labels={numeric_cols[0]: numeric_cols[0]})
                    fig2.update_layout(height=400)
                    charts['distribution'] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

                # Chart 3: Scatter of first two numeric columns
                if len(numeric_cols) >= 2:
                    color_col = cat_cols[0] if cat_cols else None
                    fig3 = px.scatter(df_main, x=numeric_cols[0], y=numeric_cols[1], color=color_col,
                                      title=f'{numeric_cols[0]} vs {numeric_cols[1]}',
                                      labels={numeric_cols[0]: numeric_cols[0], numeric_cols[1]: numeric_cols[1]})
                    fig3.update_layout(height=400)
                    charts['scatter'] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

                # Chart 4: Correlation heatmap
                if len(numeric_cols) >= 2:
                    corr = df_main[numeric_cols].corr().round(3)
                    fig4 = px.imshow(corr, text_auto=True, title='Correlation Heatmap',
                                     color_continuous_scale='RdBu_r')
                    fig4.update_layout(height=400)
                    charts['correlation'] = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

                # Summary statistics
                stats = {
                    'table': target_table,
                    'total_rows': len(df_main),
                    'columns': len(df_main.columns),
                    'numeric_cols': len(numeric_cols),
                    'categorical_cols': len(cat_cols),
                }
                if numeric_cols:
                    desc = df_main[numeric_cols].describe().round(2)
                    stats['describe'] = desc.to_dict()

        except Exception as e:
            stats['error'] = str(e)

        return render_template('eda.html', charts=charts, stats=stats)

    @app.route('/visualization')
    def visualization():
        """Visualization gallery — auto-generates charts from available data."""
        charts = {}
        try:
            all_tables = get_all_tables()
            data_tables = [t for t in all_tables if not t.startswith(('model_', 'api_', 'pipeline_', 'stream_', 'sqlite_'))]

            chart_idx = 0
            for t in data_tables[:5]:  # Up to 5 tables
                rows = query_db(f'SELECT * FROM "{t}" LIMIT 200')
                if not rows:
                    continue
                df = pd.DataFrame([dict(r) for r in rows])
                numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
                cat_cols = list(df.select_dtypes(include=['object']).columns)

                if cat_cols and numeric_cols:
                    agg = df.groupby(cat_cols[0])[numeric_cols[0]].mean().reset_index()
                    agg = agg.sort_values(numeric_cols[0], ascending=False).head(20)
                    fig = px.bar(agg, x=cat_cols[0], y=numeric_cols[0],
                                 title=f'{t}: {numeric_cols[0]} by {cat_cols[0]}')
                    fig.update_layout(xaxis_tickangle=-45, height=500)
                    charts[f'chart_{chart_idx}'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    chart_idx += 1

                if len(numeric_cols) >= 2:
                    color_col = cat_cols[0] if cat_cols else None
                    fig2 = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=color_col,
                                      title=f'{t}: {numeric_cols[0]} vs {numeric_cols[1]}')
                    fig2.update_layout(height=400)
                    charts[f'chart_{chart_idx}'] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
                    chart_idx += 1

                if numeric_cols:
                    fig3 = px.histogram(df, x=numeric_cols[0],
                                        title=f'{t}: Distribution of {numeric_cols[0]}')
                    fig3.update_layout(height=400)
                    charts[f'chart_{chart_idx}'] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
                    chart_idx += 1

        except Exception as e:
            charts['error'] = str(e)

        return render_template('visualization.html', charts=charts)

    @app.route('/bigdata')
    def bigdata():
        return render_template('bigdata.html')

    @app.route('/architecture')
    def architecture():
        return render_template('architecture.html')

    @app.route('/dataflow')
    def dataflow():
        return render_template('dataflow.html')

    @app.route('/webscraping')
    def webscraping():
        return render_template('webscraping.html')

    @app.route('/warehouse')
    def warehouse():
        return render_template('warehouse.html')

    @app.route('/pipeline')
    def pipeline():
        history = []
        try:
            history = [dict(r) for r in get_pipeline_history()]
        except Exception:
            pass
        return render_template('pipeline.html', history=history)

    @app.route('/outofcore')
    def outofcore():
        return render_template('outofcore.html')

    @app.route('/distributed')
    def distributed():
        return render_template('distributed.html')

    @app.route('/streaming')
    def streaming():
        stats = {}
        try:
            stats = get_stream_stats()
        except Exception:
            pass
        return render_template('streaming.html', stats=stats)

    @app.route('/probability')
    def probability():
        return render_template('probability.html')

    @app.route('/supervised')
    def supervised():
        model_history = []
        try:
            model_history = get_model_history()
        except Exception:
            pass
        return render_template('supervised.html', model_history=model_history)

    @app.route('/unsupervised')
    def unsupervised():
        return render_template('unsupervised.html')

    @app.route('/timeseries')
    def timeseries():
        return render_template('timeseries.html')

    @app.route('/neuralnet')
    def neuralnet():
        nn_info = neural_network_pseudocode()
        return render_template('neural_network.html', nn_info=nn_info)

    @app.route('/cv')
    def computer_vision():
        # Generate sample images
        images = {}
        preprocessing_result = None
        classification_result = None
        try:
            for img_type in ['building', 'park', 'street']:
                img = generate_sample_image(img_type)
                images[img_type] = image_to_base64(img)

            # Demo preprocessing
            sample_img = generate_sample_image('building')
            preprocessing_result = preprocess_image(sample_img)
            histogram = extract_color_histogram(sample_img)
            classification_result = simple_image_classifier(sample_img)
            classification_result['histogram'] = histogram
        except Exception as e:
            images['error'] = str(e)

        return render_template('computer_vision.html', images=images,
                               preprocessing=preprocessing_result,
                               classification=classification_result)

    @app.route('/nlp')
    def nlp():
        return render_template('nlp.html')

    @app.route('/deployment')
    def deployment():
        return render_template('deployment.html')

    @app.route('/admin')
    def admin():
        validation_results = {}
        api_logs = []
        try:
            validation_results = run_all_validations()
            logs = query_db("SELECT * FROM api_logs ORDER BY log_id DESC LIMIT 20")
            api_logs = [dict(r) for r in logs]
        except Exception:
            pass
        return render_template('admin.html', validations=validation_results, api_logs=api_logs)

    @app.route('/download/<dataset>')
    def download_data(dataset):
        """Download any dataset table as CSV."""
        try:
            # Allow downloading any table in the database
            all_tables = get_all_tables()
            # Map common short names to table names, or use as-is
            table_map = {
                'hdb': 'raw_hdb_resale',
                'transport': 'fact_transport_usage',
                'population': 'fact_population',
                'energy': 'fact_energy'
            }
            table = table_map.get(dataset, dataset)
            if table not in all_tables:
                return "Dataset not found", 404

            rows = query_db(f"SELECT * FROM {table}")
            df = pd.DataFrame([dict(r) for r in rows])
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)

            return app.response_class(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment;filename={dataset}_data.csv'}
            )
        except Exception as e:
            return str(e), 500

    return app
