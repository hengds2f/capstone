# Singapore Data Science Lab 🇸🇬

A comprehensive, interactive web application for learning data science using real-world Singapore public data. Built with Flask, SQLite, and scikit-learn.

## Features

- **24 Learning Modules** covering the full data science lifecycle
- **Interactive SQL Console** with live query execution
- **ML Model Training** — Linear Regression, Random Forest, K-Means clustering
- **Time Series Forecasting** for transport demand
- **NLP** — Text classification and sentiment analysis
- **Computer Vision** — Image generation, preprocessing, and classification
- **Data Pipeline** — Orchestrated ETL with dependency management
- **Stream Processing** — Simulated real-time event processing
- **REST API** — 15+ endpoints for programmatic access
- **Dark Theme UI** with Singapore-inspired color palette

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python run.py

# 4. Open in browser
# http://localhost:5000
```

## First Steps

1. Visit the **Dashboard** (home page)
2. Go to **Pipeline & Orchestration** → click "Execute Pipeline" to load all data
3. Explore **SQL Learning** to query the data interactively
4. Train models on the **Supervised Learning** page
5. Try the **NLP** text classifier with Singapore-related text

## Project Structure

```
Capstone/
├── app.py                  # Flask application factory & routes
├── config.py               # Configuration
├── run.py                  # Entry point
├── requirements.txt        # Dependencies
├── database/
│   └── schema.sql          # Star schema DDL (15+ tables)
├── data/                   # CSV datasets
│   ├── hdb_resale.csv      # 70 HDB resale transactions
│   ├── transport_usage.csv # 40 MRT stations
│   ├── population.csv      # 100 demographic records
│   ├── schools.csv         # 20 Singapore schools
│   ├── energy.csv          # 68 energy consumption records
│   └── sample_feedback.csv # 20 feedback entries
├── models/
│   ├── database.py         # SQLite helpers
│   ├── ml_models.py        # Regression, Random Forest, KMeans
│   ├── timeseries_models.py# SMA, exponential smoothing
│   ├── nlp_models.py       # TF-IDF + NaiveBayes/LogisticRegression
│   └── cv_models.py        # Image generation & preprocessing
├── services/
│   ├── ingestion.py        # CSV → SQLite ingestion
│   ├── pipeline.py         # Pipeline orchestrator with DAG
│   ├── streaming.py        # Event generation & windowed processing
│   └── validation.py       # Data quality checks
├── api/
│   └── routes.py           # REST API Blueprint (15 endpoints)
├── templates/              # 24 Jinja2 HTML templates
├── static/
│   ├── css/style.css       # Dark theme with MRT-line colors
│   └── js/main.js          # Client-side interactivity
└── tests/
    ├── test_database.py
    ├── test_api.py
    ├── test_pipeline.py
    └── test_models.py
```

## Learning Modules

| # | Module | Topics Covered |
|---|--------|---------------|
| 1 | Data Ingestion | CSV loading, API extraction, schema validation |
| 2 | SQL Learning | DDL, DML, JOINs, CTEs, window functions |
| 3 | Python Wrangling | NumPy, Pandas, method chaining, transformations |
| 4 | EDA | Summary statistics, distributions, correlations |
| 5 | Visualization | Plotly charts — bar, scatter, heatmap, bubble |
| 6 | Big Data Concepts | Batch vs streaming, ETL vs ELT, partitioning |
| 7 | Data Architecture | Star schema, normalization, data modeling layers |
| 8 | Data Flow | Serialization formats, encoding, data patterns |
| 9 | Web Scraping | API extraction, BeautifulSoup, ethical guidelines |
| 10 | Data Warehouse | OLAP, SCDs, aggregate tables |
| 11 | Pipeline | Orchestration, retries, idempotency |
| 12 | Out-of-Core | Chunked processing, generators, memory mapping |
| 13 | Distributed | Dask, PySpark, MapReduce concepts |
| 14 | Streaming | Windowing, anomaly detection, event processing |
| 15 | Probability | Distributions, hypothesis testing, correlation |
| 16 | Supervised Learning | Linear regression, random forest, evaluation |
| 17 | Unsupervised Learning | K-Means clustering, PCA, elbow method |
| 18 | Time Series | SMA, exponential smoothing, ARIMA, stationarity |
| 19 | Neural Networks | Feedforward, CNN, PyTorch architecture |
| 20 | Computer Vision | Image preprocessing, color histograms, augmentation |
| 21 | NLP | TF-IDF, text classification, sentiment analysis |
| 22 | Deployment | REST APIs, Flask patterns, Docker, production |
| 23 | Admin | Data validation, API logs, testing |
| 24 | Dashboard | Overview, learning paths, quick actions |

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/summary` | Database table summary |
| GET | `/api/housing?town=X` | HDB resale data |
| GET | `/api/transport` | MRT station usage |
| GET | `/api/eda` | EDA statistics |
| POST | `/api/model/train` | Train ML model |
| POST | `/api/model/predict` | Predict HDB price |
| GET | `/api/model/history` | Training history |
| POST | `/api/pipeline/run` | Execute data pipeline |
| GET | `/api/pipeline/history` | Pipeline run logs |
| POST | `/api/sql/execute` | Run read-only SQL |
| POST | `/api/stream/generate` | Generate events |
| POST | `/api/stream/process` | Process event window |
| GET | `/api/stream/stats` | Stream statistics |
| POST | `/api/feedback` | Submit feedback |
| GET | `/api/validate` | Run data validations |

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **Backend**: Flask 2.3+, Python 3.9+
- **Database**: SQLite (star schema)
- **ML**: scikit-learn, statsmodels
- **Visualization**: Plotly
- **Image Processing**: Pillow
- **Frontend**: HTML/CSS/JS, Plotly.js

## Data Sources

All data is simulated based on realistic Singapore statistics:
- HDB resale prices (based on data.gov.sg patterns)
- MRT station ridership (based on LTA DataMall patterns)
- Population demographics by planning area
- School enrollment data
- Energy consumption by sector
