"""Meltano-style ELT service: Extract, Load, Transform pipeline for Singapore datasets."""
import pandas as pd
import os
import json
import sqlite3
from datetime import datetime
from models.database import get_connection, execute_db, query_db
from config import Config


class MeltanoExtractor:
    """Singer-compatible extractor that reads from various sources."""

    def __init__(self, name, source_type='csv'):
        self.name = name
        self.source_type = source_type
        self.catalog = []
        self.records_extracted = 0

    def discover(self, source_path):
        """Discover schema from source (like `meltano invoke tap-csv --discover`)."""
        if self.source_type == 'csv':
            df = pd.read_csv(source_path, nrows=5)
        elif self.source_type == 'json':
            df = pd.read_json(source_path, lines=True, nrows=5)
        elif self.source_type == 'excel':
            df = pd.read_excel(source_path, nrows=5)
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")

        type_map = {'int64': 'integer', 'float64': 'number', 'object': 'string',
                     'bool': 'boolean', 'datetime64[ns]': 'string'}

        schema = {
            'stream': os.path.splitext(os.path.basename(source_path))[0],
            'tap_stream_id': os.path.splitext(os.path.basename(source_path))[0],
            'schema': {
                'type': 'object',
                'properties': {
                    col: {'type': type_map.get(str(dtype), 'string')}
                    for col, dtype in df.dtypes.items()
                }
            },
            'metadata': [{'breadcrumb': [], 'metadata': {'selected': True}}]
        }
        self.catalog.append(schema)
        return schema

    def extract(self, source_path):
        """Extract records from source (produces Singer RECORD messages)."""
        if self.source_type == 'csv':
            df = pd.read_csv(source_path)
        elif self.source_type == 'json':
            df = pd.read_json(source_path, lines=True)
        elif self.source_type == 'excel':
            df = pd.read_excel(source_path)
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")

        stream_name = os.path.splitext(os.path.basename(source_path))[0]
        records = []
        for _, row in df.iterrows():
            record = {
                'type': 'RECORD',
                'stream': stream_name,
                'record': {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()},
                'time_extracted': datetime.utcnow().isoformat() + 'Z'
            }
            records.append(record)

        self.records_extracted = len(records)

        # Singer STATE message
        state = {
            'type': 'STATE',
            'value': {
                'bookmarks': {
                    stream_name: {
                        'last_extracted': datetime.utcnow().isoformat(),
                        'row_count': len(records)
                    }
                }
            }
        }
        records.append(state)
        return records


class MeltanoLoader:
    """Singer-compatible loader that writes to SQLite."""

    def __init__(self, name, db_path=None):
        self.name = name
        self.db_path = db_path or Config.DATABASE
        self.records_loaded = 0

    def load(self, records, target_table=None):
        """Load Singer RECORD messages into SQLite (like target-sqlite)."""
        data_records = [r for r in records if r.get('type') == 'RECORD']
        if not data_records:
            return {'rows_loaded': 0, 'table': target_table}

        stream_name = data_records[0]['stream']
        table_name = target_table or f"raw_{stream_name}"

        df = pd.DataFrame([r['record'] for r in data_records])

        conn = sqlite3.connect(self.db_path)
        try:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            self.records_loaded = len(df)
            return {
                'rows_loaded': len(df),
                'table': table_name,
                'columns': list(df.columns)
            }
        finally:
            conn.close()


class MeltanoTransformer:
    """SQL-based transformer (simulates dbt-style transforms)."""

    def __init__(self):
        self.transforms = {}
        self.results = {}

    def add_model(self, name, sql, depends_on=None):
        """Register a dbt-style SQL model."""
        self.transforms[name] = {
            'sql': sql,
            'depends_on': depends_on or [],
            'status': 'pending'
        }

    def run(self, model_name=None):
        """Execute transforms (like `meltano invoke dbt-sqlite:run`)."""
        conn = get_connection()
        try:
            models_to_run = [model_name] if model_name else list(self.transforms.keys())
            for name in models_to_run:
                if name not in self.transforms:
                    self.results[name] = {'status': 'error', 'error': f'Model {name} not found'}
                    continue

                model = self.transforms[name]
                # Check dependencies
                for dep in model['depends_on']:
                    if dep in self.results and self.results[dep].get('status') == 'error':
                        self.results[name] = {'status': 'skipped', 'error': f'Dependency {dep} failed'}
                        continue

                try:
                    conn.execute(model['sql'])
                    conn.commit()
                    # Count rows in resulting table/view
                    try:
                        row_count = conn.execute(
                            f"SELECT COUNT(*) FROM {name}"
                        ).fetchone()[0]
                    except Exception:
                        row_count = 0

                    model['status'] = 'completed'
                    self.results[name] = {
                        'status': 'completed',
                        'rows': row_count
                    }
                except Exception as e:
                    model['status'] = 'failed'
                    self.results[name] = {'status': 'error', 'error': str(e)}

            return self.results
        finally:
            conn.close()


class MeltanoELTPipeline:
    """Full Meltano-style ELT pipeline orchestrator."""

    def __init__(self, name='meltano_elt'):
        self.name = name
        self.extractor = None
        self.loader = None
        self.transformer = MeltanoTransformer()
        self.run_log = []

    def _log(self, step, status, detail=None):
        entry = {
            'step': step,
            'status': status,
            'detail': detail,
            'timestamp': datetime.now().isoformat()
        }
        self.run_log.append(entry)
        try:
            execute_db(
                "INSERT INTO pipeline_runs (pipeline_name, step_name, status, started_at, completed_at, rows_processed, error_message) VALUES (?,?,?,?,?,?,?)",
                (self.name, step, status, entry['timestamp'],
                 entry['timestamp'] if status in ('completed', 'failed') else None,
                 detail.get('rows_loaded', 0) if isinstance(detail, dict) else 0,
                 detail.get('error') if isinstance(detail, dict) else None)
            )
        except Exception:
            pass

    def configure_extract(self, source_path, source_type='csv'):
        """Configure the extraction source."""
        name = os.path.splitext(os.path.basename(source_path))[0]
        self.extractor = MeltanoExtractor(f'tap-{name}', source_type)
        self._source_path = source_path

    def configure_load(self, target_table=None, db_path=None):
        """Configure the load target."""
        self.loader = MeltanoLoader('target-sqlite', db_path)
        self._target_table = target_table

    def add_transform(self, name, sql, depends_on=None):
        """Add a dbt-style SQL transformation model."""
        self.transformer.add_model(name, sql, depends_on)

    def run(self):
        """Execute the full ELT pipeline: Extract -> Load -> Transform."""
        results = {
            'pipeline': self.name,
            'started_at': datetime.now().isoformat(),
            'steps': {}
        }

        # === EXTRACT ===
        self._log('extract', 'running')
        try:
            schema = self.extractor.discover(self._source_path)
            records = self.extractor.extract(self._source_path)
            results['steps']['extract'] = {
                'status': 'completed',
                'records': self.extractor.records_extracted,
                'stream': schema['stream'],
                'schema': schema['schema']
            }
            self._log('extract', 'completed', {'rows_loaded': self.extractor.records_extracted})
        except Exception as e:
            results['steps']['extract'] = {'status': 'failed', 'error': str(e)}
            self._log('extract', 'failed', {'error': str(e)})
            results['completed_at'] = datetime.now().isoformat()
            return results

        # === LOAD ===
        self._log('load', 'running')
        try:
            load_result = self.loader.load(records, self._target_table)
            results['steps']['load'] = {
                'status': 'completed',
                **load_result
            }
            self._log('load', 'completed', load_result)
        except Exception as e:
            results['steps']['load'] = {'status': 'failed', 'error': str(e)}
            self._log('load', 'failed', {'error': str(e)})
            results['completed_at'] = datetime.now().isoformat()
            return results

        # === TRANSFORM ===
        if self.transformer.transforms:
            self._log('transform', 'running')
            try:
                transform_results = self.transformer.run()
                all_ok = all(r.get('status') == 'completed' for r in transform_results.values())
                results['steps']['transform'] = {
                    'status': 'completed' if all_ok else 'partial',
                    'models': transform_results
                }
                total_rows = sum(r.get('rows', 0) for r in transform_results.values())
                self._log('transform', 'completed' if all_ok else 'partial',
                          {'rows_loaded': total_rows})
            except Exception as e:
                results['steps']['transform'] = {'status': 'failed', 'error': str(e)}
                self._log('transform', 'failed', {'error': str(e)})

        results['completed_at'] = datetime.now().isoformat()
        return results


# ============================================================
# Pre-built ELT pipelines for Singapore datasets
# ============================================================

def build_hdb_elt_pipeline():
    """Build Meltano ELT pipeline for HDB resale data."""
    pipeline = MeltanoELTPipeline('hdb_elt')
    pipeline.configure_extract(
        os.path.join(Config.DATA_DIR, 'hdb_resale.csv'), 'csv'
    )
    pipeline.configure_load('raw_hdb_resale')

    # dbt-style transforms
    pipeline.add_transform(
        'stg_hdb_monthly_avg',
        """CREATE TABLE IF NOT EXISTS stg_hdb_monthly_avg AS
           SELECT month, town, flat_type,
                  COUNT(*) as transaction_count,
                  ROUND(AVG(resale_price), 2) as avg_price,
                  ROUND(MIN(resale_price), 2) as min_price,
                  ROUND(MAX(resale_price), 2) as max_price,
                  ROUND(AVG(floor_area_sqm), 2) as avg_area
           FROM raw_hdb_resale
           GROUP BY month, town, flat_type"""
    )
    pipeline.add_transform(
        'stg_hdb_town_summary',
        """CREATE TABLE IF NOT EXISTS stg_hdb_town_summary AS
           SELECT town,
                  COUNT(*) as total_transactions,
                  ROUND(AVG(resale_price), 2) as avg_price,
                  ROUND(AVG(floor_area_sqm), 2) as avg_area,
                  ROUND(AVG(resale_price / NULLIF(floor_area_sqm, 0)), 2) as avg_price_per_sqm
           FROM raw_hdb_resale
           GROUP BY town""",
        depends_on=['stg_hdb_monthly_avg']
    )
    return pipeline


def build_transport_elt_pipeline():
    """Build Meltano ELT pipeline for transport data."""
    pipeline = MeltanoELTPipeline('transport_elt')
    pipeline.configure_extract(
        os.path.join(Config.DATA_DIR, 'transport_usage.csv'), 'csv'
    )
    pipeline.configure_load('raw_transport_usage')

    pipeline.add_transform(
        'stg_transport_line_summary',
        """CREATE TABLE IF NOT EXISTS stg_transport_line_summary AS
           SELECT line_name, line_color,
                  COUNT(DISTINCT station_code) as station_count,
                  SUM(tap_in_count) as total_tap_in,
                  SUM(tap_out_count) as total_tap_out,
                  SUM(tap_in_count + tap_out_count) as total_trips,
                  ROUND(AVG(peak_hour_pct), 3) as avg_peak_pct
           FROM raw_transport_usage
           GROUP BY line_name, line_color"""
    )
    return pipeline


def build_energy_elt_pipeline():
    """Build Meltano ELT pipeline for energy data."""
    pipeline = MeltanoELTPipeline('energy_elt')
    pipeline.configure_extract(
        os.path.join(Config.DATA_DIR, 'energy.csv'), 'csv'
    )
    pipeline.configure_load('raw_energy')

    pipeline.add_transform(
        'stg_energy_yearly',
        """CREATE TABLE IF NOT EXISTS stg_energy_yearly AS
           SELECT year, sector, energy_type,
                  ROUND(SUM(consumption_gwh), 2) as total_consumption_gwh,
                  ROUND(SUM(cost_million_sgd), 2) as total_cost_million_sgd,
                  ROUND(SUM(carbon_emission_tonnes), 2) as total_carbon_tonnes
           FROM raw_energy
           GROUP BY year, sector, energy_type"""
    )
    return pipeline


def build_custom_elt_pipeline(source_path, table_name, source_type='csv', transforms=None):
    """Build a custom Meltano ELT pipeline for any dataset."""
    pipeline = MeltanoELTPipeline(f'custom_elt_{table_name}')
    pipeline.configure_extract(source_path, source_type)
    pipeline.configure_load(table_name)

    if transforms:
        for t in transforms:
            pipeline.add_transform(t['name'], t['sql'], t.get('depends_on'))

    return pipeline


def run_full_meltano_elt():
    """Run all pre-built Meltano ELT pipelines."""
    results = {}

    pipelines = [
        ('hdb', build_hdb_elt_pipeline),
        ('transport', build_transport_elt_pipeline),
        ('energy', build_energy_elt_pipeline),
    ]

    for name, builder in pipelines:
        try:
            pipeline = builder()
            result = pipeline.run()
            results[name] = result
        except Exception as e:
            results[name] = {'status': 'failed', 'error': str(e)}

    return results


def get_meltano_config():
    """Read and return the meltano.yml configuration."""
    config_path = os.path.join(Config.DATA_DIR, '..', 'meltano.yml')
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback: read as text
        try:
            with open(config_path) as f:
                return {'raw': f.read()}
        except FileNotFoundError:
            return {'error': 'meltano.yml not found'}
    except FileNotFoundError:
        return {'error': 'meltano.yml not found'}
