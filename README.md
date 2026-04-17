---
title: Data Science Lab
emoji: 📊
colorFrom: red
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# Data Science Lab 📊

A comprehensive, interactive web application for learning data science with any dataset. Built with Flask, SQLite, and scikit-learn. Upload your own CSV data and explore all modules.

## Features

- **24 Learning Modules** covering the full data science lifecycle
- **Interactive SQL Console** with live query execution
- **ML Model Training** — Linear Regression, Random Forest, K-Means clustering
- **Time Series Forecasting** for any time-based data
- **NLP** — Text classification and sentiment analysis
- **Computer Vision** — Image generation, preprocessing, and classification
- **Data Pipeline** — Orchestrated ETL with dependency management
- **Stream Processing** — Simulated real-time event processing
- **REST API** — 15+ endpoints for programmatic access
- **Dark Theme UI** with a modern color palette
- **Dataset-Agnostic** — upload any CSV and all modules adapt automatically

## Quick Start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run.py
# Open http://localhost:5000
```

## First Steps

1. Visit the **Dashboard** (home page)
2. Go to **Data Ingestion** → upload a CSV or load sample datasets
3. Go to **Pipeline & Orchestration** → click "Execute Pipeline" to process data
4. Explore **SQL Learning** to query the data interactively
5. Train models on the **Supervised Learning** page

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/summary` | Database table summary |
| GET | `/api/housing?table=X` | Query any data table |
| GET | `/api/transport?table=X` | Query any data table |
| GET | `/api/eda` | EDA statistics for all tables |
| POST | `/api/model/train` | Train ML model |
| POST | `/api/model/predict` | Predict target value |
| GET | `/api/model/history` | Training history |
| POST | `/api/pipeline/run` | Execute data pipeline |
| POST | `/api/sql/execute` | Run read-only SQL |
| POST | `/api/stream/generate` | Generate events |

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **Backend**: Flask 2.3+, Python 3.9+
- **Database**: SQLite
- **ML**: scikit-learn, statsmodels
- **Visualization**: Plotly
- **Image Processing**: Pillow
- **Frontend**: HTML/CSS/JS, Plotly.js

## Data Sources

Sample datasets are included for demonstration. Upload your own CSV files to work with any data domain.
