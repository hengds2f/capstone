-- Singapore Data Science Lab - Database Schema
-- Designed for educational purposes: normalized star-schema hybrid

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-------------------------------------------------------
-- DIMENSION TABLES
-------------------------------------------------------

CREATE TABLE IF NOT EXISTS dim_location (
    location_id INTEGER PRIMARY KEY AUTOINCREMENT,
    town VARCHAR(100) NOT NULL,
    flat_type VARCHAR(50),
    block VARCHAR(20),
    street_name VARCHAR(200),
    storey_range VARCHAR(20),
    floor_area_sqm REAL,
    planning_area VARCHAR(100),
    region VARCHAR(50),
    latitude REAL,
    longitude REAL,
    UNIQUE(town, block, street_name, flat_type, storey_range)
);

CREATE TABLE IF NOT EXISTS dim_time (
    time_id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_date DATE,
    year INTEGER NOT NULL,
    quarter INTEGER,
    month INTEGER,
    month_name VARCHAR(20),
    day_of_week VARCHAR(15),
    is_weekend INTEGER DEFAULT 0,
    UNIQUE(full_date)
);

CREATE TABLE IF NOT EXISTS dim_property_type (
    property_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    flat_type VARCHAR(50) NOT NULL,
    flat_model VARCHAR(50),
    lease_commence_year INTEGER,
    remaining_lease VARCHAR(50),
    UNIQUE(flat_type, flat_model, lease_commence_year)
);

CREATE TABLE IF NOT EXISTS dim_transport_station (
    station_id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_name VARCHAR(100) NOT NULL,
    station_code VARCHAR(20),
    line_name VARCHAR(50),
    line_color VARCHAR(30),
    latitude REAL,
    longitude REAL,
    opening_year INTEGER,
    is_interchange INTEGER DEFAULT 0,
    UNIQUE(station_code)
);

CREATE TABLE IF NOT EXISTS dim_school (
    school_id INTEGER PRIMARY KEY AUTOINCREMENT,
    school_name VARCHAR(200) NOT NULL,
    school_type VARCHAR(50),
    zone VARCHAR(50),
    cluster VARCHAR(50),
    address VARCHAR(300),
    postal_code VARCHAR(10),
    latitude REAL,
    longitude REAL,
    UNIQUE(school_name)
);

CREATE TABLE IF NOT EXISTS dim_land_use (
    land_use_id INTEGER PRIMARY KEY AUTOINCREMENT,
    planning_area VARCHAR(100) NOT NULL,
    land_use_type VARCHAR(100),
    gpr REAL,
    area_hectares REAL,
    zoning VARCHAR(50),
    UNIQUE(planning_area, land_use_type)
);

-------------------------------------------------------
-- FACT TABLES
-------------------------------------------------------

CREATE TABLE IF NOT EXISTS raw_hdb_resale (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    month VARCHAR(10),
    town VARCHAR(100),
    flat_type VARCHAR(50),
    block VARCHAR(20),
    street_name VARCHAR(200),
    storey_range VARCHAR(20),
    floor_area_sqm REAL,
    flat_model VARCHAR(50),
    lease_commence_date INTEGER,
    remaining_lease VARCHAR(50),
    resale_price REAL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_hdb_transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id INTEGER,
    time_id INTEGER,
    property_type_id INTEGER,
    resale_price REAL NOT NULL,
    price_per_sqm REAL,
    FOREIGN KEY (location_id) REFERENCES dim_location(location_id),
    FOREIGN KEY (time_id) REFERENCES dim_time(time_id),
    FOREIGN KEY (property_type_id) REFERENCES dim_property_type(property_type_id)
);

CREATE TABLE IF NOT EXISTS fact_transport_usage (
    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id INTEGER,
    time_id INTEGER,
    tap_in_count INTEGER DEFAULT 0,
    tap_out_count INTEGER DEFAULT 0,
    total_trips INTEGER DEFAULT 0,
    peak_hour_pct REAL,
    FOREIGN KEY (station_id) REFERENCES dim_transport_station(station_id),
    FOREIGN KEY (time_id) REFERENCES dim_time(time_id)
);

CREATE TABLE IF NOT EXISTS fact_population (
    pop_id INTEGER PRIMARY KEY AUTOINCREMENT,
    planning_area VARCHAR(100),
    year INTEGER,
    age_group VARCHAR(30),
    gender VARCHAR(10),
    population_count INTEGER,
    density_per_sqkm REAL
);

CREATE TABLE IF NOT EXISTS fact_school_enrollment (
    enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    school_id INTEGER,
    year INTEGER,
    level VARCHAR(30),
    enrollment_count INTEGER,
    num_classes INTEGER,
    avg_class_size REAL,
    FOREIGN KEY (school_id) REFERENCES dim_school(school_id)
);

CREATE TABLE IF NOT EXISTS fact_energy (
    energy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER,
    month INTEGER,
    sector VARCHAR(50),
    energy_type VARCHAR(50),
    consumption_gwh REAL,
    cost_million_sgd REAL,
    carbon_emission_tonnes REAL
);

-------------------------------------------------------
-- OPERATIONAL / ML TABLES
-------------------------------------------------------

CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50),
    metric_name VARCHAR(50),
    metric_value REAL,
    parameters TEXT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dataset_size INTEGER,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_name VARCHAR(100) NOT NULL,
    step_name VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    rows_processed INTEGER DEFAULT 0,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS api_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint VARCHAR(200),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms REAL,
    ip_address VARCHAR(45),
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    module_name VARCHAR(100),
    rating INTEGER CHECK(rating BETWEEN 1 AND 5),
    comment TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stream_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type VARCHAR(50),
    source VARCHAR(50),
    payload TEXT,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed INTEGER DEFAULT 0
);

-------------------------------------------------------
-- INDEXES
-------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_raw_hdb_town ON raw_hdb_resale(town);
CREATE INDEX IF NOT EXISTS idx_raw_hdb_month ON raw_hdb_resale(month);
CREATE INDEX IF NOT EXISTS idx_fact_hdb_location ON fact_hdb_transactions(location_id);
CREATE INDEX IF NOT EXISTS idx_fact_hdb_time ON fact_hdb_transactions(time_id);
CREATE INDEX IF NOT EXISTS idx_transport_station ON fact_transport_usage(station_id);
CREATE INDEX IF NOT EXISTS idx_population_area ON fact_population(planning_area, year);
CREATE INDEX IF NOT EXISTS idx_pipeline_status ON pipeline_runs(status);
CREATE INDEX IF NOT EXISTS idx_stream_processed ON stream_events(processed, event_time);

-------------------------------------------------------
-- VIEWS
-------------------------------------------------------

CREATE VIEW IF NOT EXISTS v_hdb_summary AS
SELECT
    l.town,
    l.planning_area,
    t.year,
    t.month,
    p.flat_type,
    p.flat_model,
    f.resale_price,
    f.price_per_sqm,
    l.floor_area_sqm
FROM fact_hdb_transactions f
JOIN dim_location l ON f.location_id = l.location_id
JOIN dim_time t ON f.time_id = t.time_id
JOIN dim_property_type p ON f.property_type_id = p.property_type_id;

CREATE VIEW IF NOT EXISTS v_transport_summary AS
SELECT
    s.station_name,
    s.line_name,
    t.year,
    t.month,
    u.tap_in_count,
    u.tap_out_count,
    u.total_trips,
    u.peak_hour_pct
FROM fact_transport_usage u
JOIN dim_transport_station s ON u.station_id = s.station_id
JOIN dim_time t ON u.time_id = t.time_id;
