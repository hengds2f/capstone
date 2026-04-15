"""Singapore Data Science Lab - Flask Application Factory."""
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
    mock_datagov_api, validate_dataframe, ingest_csv_to_table
)
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
            {'name': 'Data Ingestion', 'url': '/ingestion', 'icon': '📥', 'desc': 'Load and validate Singapore datasets'},
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
            rows = query_db("SELECT * FROM raw_hdb_resale LIMIT 20")
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
        except Exception:
            pass
        return render_template('python_wrangling.html', demo_data=demo_data)

    @app.route('/eda')
    def eda():
        """EDA module."""
        charts = {}
        stats = {}
        try:
            # Price by town
            rows = query_db("""
                SELECT town, AVG(resale_price) as avg_price, COUNT(*) as count
                FROM raw_hdb_resale GROUP BY town ORDER BY avg_price DESC
            """)
            if rows:
                df = pd.DataFrame([dict(r) for r in rows])
                fig = px.bar(df, x='town', y='avg_price', color='count',
                             title='Average HDB Resale Price by Town',
                             labels={'avg_price': 'Average Price (SGD)', 'town': 'Town'})
                fig.update_layout(xaxis_tickangle=-45, height=500)
                charts['price_by_town'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            # Price distribution
            rows2 = query_db("SELECT resale_price, flat_type FROM raw_hdb_resale")
            if rows2:
                df2 = pd.DataFrame([dict(r) for r in rows2])
                fig2 = px.histogram(df2, x='resale_price', color='flat_type',
                                    title='HDB Resale Price Distribution by Flat Type',
                                    labels={'resale_price': 'Resale Price (SGD)'})
                fig2.update_layout(height=400)
                charts['price_dist'] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

            # Floor area vs price scatter
            rows3 = query_db("SELECT floor_area_sqm, resale_price, town, flat_type FROM raw_hdb_resale")
            if rows3:
                df3 = pd.DataFrame([dict(r) for r in rows3])
                fig3 = px.scatter(df3, x='floor_area_sqm', y='resale_price', color='flat_type',
                                  hover_data=['town'],
                                  title='Floor Area vs Resale Price',
                                  labels={'floor_area_sqm': 'Floor Area (sqm)', 'resale_price': 'Resale Price (SGD)'})
                fig3.update_layout(height=400)
                charts['area_vs_price'] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

            # Correlation data
            if rows3:
                numeric_cols = df3.select_dtypes(include=[np.number])
                if not numeric_cols.empty:
                    corr = numeric_cols.corr().round(3)
                    fig4 = px.imshow(corr, text_auto=True, title='Correlation Heatmap',
                                     color_continuous_scale='RdBu_r')
                    fig4.update_layout(height=400)
                    charts['correlation'] = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

            # Summary statistics
            stats_rows = query_db("""
                SELECT COUNT(*) as total, ROUND(AVG(resale_price),0) as avg_price,
                       MIN(resale_price) as min_price, MAX(resale_price) as max_price,
                       COUNT(DISTINCT town) as towns, COUNT(DISTINCT flat_type) as flat_types
                FROM raw_hdb_resale
            """)
            if stats_rows:
                stats = dict(stats_rows[0])

        except Exception as e:
            stats['error'] = str(e)

        return render_template('eda.html', charts=charts, stats=stats)

    @app.route('/visualization')
    def visualization():
        """Visualization gallery."""
        charts = {}
        try:
            # Transport usage chart
            rows = query_db("""
                SELECT s.station_name, s.line_name, s.line_color, u.total_trips, u.peak_hour_pct
                FROM fact_transport_usage u
                JOIN dim_transport_station s ON u.station_id = s.station_id
                ORDER BY u.total_trips DESC LIMIT 20
            """)
            if rows:
                df = pd.DataFrame([dict(r) for r in rows])
                fig = px.bar(df, x='station_name', y='total_trips', color='line_name',
                             title='Top 20 MRT Stations by Total Trips',
                             labels={'total_trips': 'Total Trips', 'station_name': 'Station'})
                fig.update_layout(xaxis_tickangle=-45, height=500)
                charts['transport'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            # Population pyramid
            pop_rows = query_db("""
                SELECT age_group, gender, SUM(population_count) as total
                FROM fact_population GROUP BY age_group, gender
            """)
            if pop_rows:
                df_pop = pd.DataFrame([dict(r) for r in pop_rows])
                fig2 = px.bar(df_pop, x='total', y='age_group', color='gender',
                              orientation='h', barmode='group',
                              title='Population Distribution by Age and Gender',
                              labels={'total': 'Population', 'age_group': 'Age Group'})
                fig2.update_layout(height=400)
                charts['population'] = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

            # Energy consumption
            energy_rows = query_db("""
                SELECT month, sector, SUM(consumption_gwh) as total_gwh
                FROM fact_energy WHERE year = 2023
                GROUP BY month, sector ORDER BY month
            """)
            if energy_rows:
                df_energy = pd.DataFrame([dict(r) for r in energy_rows])
                fig3 = px.line(df_energy, x='month', y='total_gwh', color='sector',
                               title='Monthly Energy Consumption by Sector (2023)',
                               labels={'total_gwh': 'Consumption (GWh)', 'month': 'Month'})
                fig3.update_layout(height=400)
                charts['energy'] = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

            # School enrollment
            school_rows = query_db("""
                SELECT s.school_name, s.school_type, e.enrollment_count
                FROM fact_school_enrollment e
                JOIN dim_school s ON e.school_id = s.school_id
                ORDER BY e.enrollment_count DESC LIMIT 15
            """)
            if school_rows:
                df_school = pd.DataFrame([dict(r) for r in school_rows])
                fig4 = px.bar(df_school, x='school_name', y='enrollment_count', color='school_type',
                              title='Top 15 Schools by Enrollment',
                              labels={'enrollment_count': 'Enrollment', 'school_name': 'School'})
                fig4.update_layout(xaxis_tickangle=-45, height=500)
                charts['schools'] = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

            # HDB price map (bubble)
            hdb_rows = query_db("""
                SELECT l.town, AVG(f.resale_price) as avg_price, COUNT(*) as count,
                       AVG(l.floor_area_sqm) as avg_area
                FROM fact_hdb_transactions f
                JOIN dim_location l ON f.location_id = l.location_id
                GROUP BY l.town
            """)
            if hdb_rows:
                df_hdb = pd.DataFrame([dict(r) for r in hdb_rows])
                fig5 = px.scatter(df_hdb, x='avg_area', y='avg_price', size='count',
                                  color='town', hover_name='town',
                                  title='HDB Price vs Area by Town (bubble size = transaction count)',
                                  labels={'avg_price': 'Avg Price (SGD)', 'avg_area': 'Avg Floor Area (sqm)'})
                fig5.update_layout(height=500)
                charts['hdb_bubble'] = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

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
            for img_type in ['building', 'park', 'mrt']:
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
        """Download dataset as CSV."""
        try:
            table_map = {
                'hdb': 'raw_hdb_resale',
                'transport': 'fact_transport_usage',
                'population': 'fact_population',
                'energy': 'fact_energy'
            }
            table = table_map.get(dataset)
            if not table:
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
