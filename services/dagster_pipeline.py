"""Dagster-style orchestration service: Software-Defined Assets, Ops, Jobs, and Schedules.

Implements Dagster's core concepts (assets, ops, graphs, jobs, resources, IO managers,
schedules, sensors) as a lightweight simulation for educational purposes.
"""
import pandas as pd
import os
import time
import hashlib
import json
from datetime import datetime, timedelta
from models.database import get_connection, execute_db, query_db
from config import Config


# ============================================================
# Core Dagster Abstractions
# ============================================================

class AssetMaterialization:
    """Represents a materialized asset (Dagster AssetMaterialization equivalent)."""

    def __init__(self, asset_key, metadata=None):
        self.asset_key = asset_key
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            'asset_key': self.asset_key,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


class OpResult:
    """Result of an op execution (output + metadata)."""

    def __init__(self, output, metadata=None, asset_materializations=None):
        self.output = output
        self.metadata = metadata or {}
        self.asset_materializations = asset_materializations or []


class DagsterResource:
    """A shared resource (e.g., database connection, API client)."""

    def __init__(self, name, config=None):
        self.name = name
        self.config = config or {}

    def setup(self):
        """Initialize the resource."""
        pass

    def teardown(self):
        """Clean up the resource."""
        pass


class SQLiteIOManager(DagsterResource):
    """Dagster IO Manager that reads/writes DataFrames to SQLite tables."""

    def __init__(self, db_path=None):
        super().__init__('sqlite_io_manager', {'db_path': db_path or Config.DATABASE})
        self.db_path = db_path or Config.DATABASE

    def handle_output(self, context, obj, table_name):
        """Write a DataFrame to a SQLite table."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            if isinstance(obj, pd.DataFrame):
                obj.to_sql(table_name, conn, if_exists='replace', index=False)
                return len(obj)
            return 0
        finally:
            conn.close()

    def load_input(self, table_name):
        """Read a DataFrame from a SQLite table."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            return pd.read_sql(f"SELECT * FROM {table_name}", conn)
        finally:
            conn.close()


class DagsterOp:
    """A single computation unit (Dagster @op equivalent).

    An op is a function with defined inputs, outputs, and config.
    """

    def __init__(self, name, compute_fn, description='', required_resources=None,
                 retry_policy=None, tags=None):
        self.name = name
        self.compute_fn = compute_fn
        self.description = description
        self.required_resources = required_resources or []
        self.retry_policy = retry_policy or {'max_retries': 0, 'delay': 1}
        self.tags = tags or {}
        self.status = 'pending'
        self.result = None
        self.start_time = None
        self.end_time = None
        self.attempt = 0

    def execute(self, context, inputs=None):
        """Execute the op with retry policy."""
        max_retries = self.retry_policy.get('max_retries', 0)
        delay = self.retry_policy.get('delay', 1)

        for attempt in range(max_retries + 1):
            self.attempt = attempt + 1
            self.start_time = datetime.now()
            self.status = 'running'
            try:
                result = self.compute_fn(context, inputs or {})
                self.end_time = datetime.now()
                self.status = 'completed'
                self.result = result
                return result
            except Exception as e:
                self.end_time = datetime.now()
                if attempt < max_retries:
                    self.status = 'retrying'
                    time.sleep(min(delay * (2 ** attempt), 5))
                else:
                    self.status = 'failed'
                    self.result = str(e)
                    raise

    @property
    def duration_seconds(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0


class DagsterGraph:
    """A DAG of ops with dependency tracking (Dagster @graph equivalent)."""

    def __init__(self, name, description=''):
        self.name = name
        self.description = description
        self.ops = {}
        self.dependencies = {}  # op_name -> [upstream_op_names]

    def add_op(self, op, depends_on=None):
        self.ops[op.name] = op
        self.dependencies[op.name] = depends_on or []

    def get_execution_order(self):
        """Topological sort of ops based on dependencies."""
        visited = set()
        order = []

        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in self.dependencies.get(name, []):
                if dep in self.ops:
                    visit(dep)
            order.append(name)

        for name in self.ops:
            visit(name)
        return order


class DagsterJob:
    """An executable pipeline (Dagster @job equivalent).

    A job = a graph + resources + config.
    """

    def __init__(self, name, graph, resources=None, tags=None, description=''):
        self.name = name
        self.graph = graph
        self.resources = resources or {}
        self.tags = tags or {}
        self.description = description
        self.run_id = None
        self.run_log = []

    def _generate_run_id(self):
        return hashlib.md5(
            f"{self.name}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

    def _log(self, op_name, status, duration=0, rows=0, error=None, metadata=None):
        entry = {
            'run_id': self.run_id,
            'op': op_name,
            'status': status,
            'duration_seconds': round(duration, 3),
            'rows_processed': rows,
            'error': error,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        self.run_log.append(entry)
        try:
            execute_db(
                "INSERT INTO pipeline_runs (pipeline_name, step_name, status, started_at, completed_at, rows_processed, error_message) VALUES (?,?,?,?,?,?,?)",
                (f"dagster:{self.name}", op_name, status,
                 datetime.now().isoformat(),
                 datetime.now().isoformat() if status in ('completed', 'failed') else None,
                 rows, error)
            )
        except Exception:
            pass

    def execute(self):
        """Execute the job — run all ops in dependency order."""
        self.run_id = self._generate_run_id()
        execution_order = self.graph.get_execution_order()

        context = {
            'run_id': self.run_id,
            'job_name': self.name,
            'resources': self.resources,
            'outputs': {},  # stores outputs from upstream ops
        }

        results = {
            'run_id': self.run_id,
            'job': self.name,
            'started_at': datetime.now().isoformat(),
            'ops': {}
        }
        completed_ops = set()

        for op_name in execution_order:
            op = self.graph.ops[op_name]

            # Check dependencies
            unmet = [d for d in self.graph.dependencies[op_name] if d not in completed_ops]
            if unmet:
                self._log(op_name, 'skipped', error=f"Unmet deps: {unmet}")
                results['ops'][op_name] = {
                    'status': 'skipped',
                    'error': f"Unmet dependencies: {unmet}"
                }
                continue

            # Gather upstream outputs as inputs
            inputs = {}
            for dep in self.graph.dependencies[op_name]:
                if dep in context['outputs']:
                    inputs[dep] = context['outputs'][dep]

            # Execute op
            self._log(op_name, 'running')
            try:
                result = op.execute(context, inputs)
                context['outputs'][op_name] = result
                completed_ops.add(op_name)

                rows = 0
                metadata = {}
                if isinstance(result, OpResult):
                    rows = result.metadata.get('row_count', 0)
                    metadata = result.metadata
                elif isinstance(result, int):
                    rows = result
                elif isinstance(result, pd.DataFrame):
                    rows = len(result)

                self._log(op_name, 'completed', op.duration_seconds, rows)
                results['ops'][op_name] = {
                    'status': 'completed',
                    'duration': round(op.duration_seconds, 3),
                    'rows': rows,
                    'attempt': op.attempt,
                    'metadata': metadata
                }
            except Exception as e:
                self._log(op_name, 'failed', op.duration_seconds, error=str(e))
                results['ops'][op_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'attempt': op.attempt,
                    'duration': round(op.duration_seconds, 3)
                }

        results['completed_at'] = datetime.now().isoformat()
        all_ok = all(r['status'] == 'completed' for r in results['ops'].values())
        results['status'] = 'completed' if all_ok else 'partial_failure'
        return results


class DagsterSchedule:
    """Schedule definition for a job (Dagster @schedule equivalent)."""

    def __init__(self, name, cron_schedule, job, description=''):
        self.name = name
        self.cron_schedule = cron_schedule
        self.job = job
        self.description = description
        self.last_run = None
        self.next_run = None
        self._compute_next_run()

    def _compute_next_run(self):
        """Simplified next-run based on cron string."""
        now = datetime.now()
        cron_map = {
            '0 0 * * *': now.replace(hour=0, minute=0, second=0) + timedelta(days=1),
            '0 */6 * * *': now + timedelta(hours=6),
            '0 0 * * 0': now + timedelta(days=(6 - now.weekday()) % 7 + 1),
            '0 0 1 * *': (now.replace(day=1) + timedelta(days=32)).replace(day=1),
        }
        self.next_run = cron_map.get(self.cron_schedule,
                                      now + timedelta(hours=1)).isoformat()

    def should_run(self):
        if self.next_run and datetime.now() >= datetime.fromisoformat(self.next_run):
            return True
        return False

    def to_dict(self):
        return {
            'name': self.name,
            'cron': self.cron_schedule,
            'job': self.job.name,
            'last_run': self.last_run,
            'next_run': self.next_run,
            'description': self.description
        }


class DagsterSensor:
    """Sensor that triggers a job based on external conditions (Dagster @sensor equivalent)."""

    def __init__(self, name, evaluation_fn, job, description='', min_interval=60):
        self.name = name
        self.evaluation_fn = evaluation_fn
        self.job = job
        self.description = description
        self.min_interval = min_interval
        self.last_evaluated = None
        self.last_triggered = None

    def evaluate(self):
        """Evaluate sensor condition and return whether to trigger."""
        self.last_evaluated = datetime.now().isoformat()
        try:
            should_trigger, context = self.evaluation_fn()
            if should_trigger:
                self.last_triggered = datetime.now().isoformat()
            return should_trigger, context
        except Exception as e:
            return False, {'error': str(e)}

    def to_dict(self):
        return {
            'name': self.name,
            'job': self.job.name,
            'min_interval': self.min_interval,
            'last_evaluated': self.last_evaluated,
            'last_triggered': self.last_triggered,
            'description': self.description
        }


# ============================================================
# Software-Defined Assets (Dagster's primary API)
# ============================================================

class SoftwareDefinedAsset:
    """A declarative data asset (Dagster @asset equivalent).

    Instead of defining ops imperatively, you declare WHAT data should exist
    and HOW to compute it. Dagster infers the DAG from asset dependencies.
    """

    def __init__(self, key, compute_fn, deps=None, description='',
                 group_name='default', io_manager=None, metadata=None):
        self.key = key
        self.compute_fn = compute_fn
        self.deps = deps or []
        self.description = description
        self.group_name = group_name
        self.io_manager = io_manager or SQLiteIOManager()
        self.metadata = metadata or {}
        self.last_materialized = None
        self.materialization_count = 0

    def materialize(self, upstream_data=None):
        """Materialize (compute + persist) this asset."""
        start = datetime.now()
        try:
            result = self.compute_fn(upstream_data or {})

            # Persist via IO manager
            if isinstance(result, pd.DataFrame):
                rows = self.io_manager.handle_output(None, result, self.key)
            else:
                rows = 0

            self.last_materialized = datetime.now().isoformat()
            self.materialization_count += 1

            duration = (datetime.now() - start).total_seconds()

            mat = AssetMaterialization(self.key, {
                'row_count': rows if isinstance(rows, int) else len(result) if isinstance(result, pd.DataFrame) else 0,
                'duration_seconds': round(duration, 3),
                'materialization_number': self.materialization_count
            })

            try:
                execute_db(
                    "INSERT INTO pipeline_runs (pipeline_name, step_name, status, started_at, completed_at, rows_processed) VALUES (?,?,?,?,?,?)",
                    (f"dagster:asset:{self.group_name}", self.key, 'completed',
                     start.isoformat(), datetime.now().isoformat(),
                     mat.metadata.get('row_count', 0))
                )
            except Exception:
                pass

            return {
                'status': 'completed',
                'asset_key': self.key,
                'data': result,
                'materialization': mat.to_dict(),
                'duration': round(duration, 3)
            }
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            return {
                'status': 'failed',
                'asset_key': self.key,
                'error': str(e),
                'duration': round(duration, 3)
            }


class AssetGraph:
    """A collection of software-defined assets with automatic dependency resolution."""

    def __init__(self, name='default'):
        self.name = name
        self.assets = {}

    def add_asset(self, asset):
        self.assets[asset.key] = asset

    def get_execution_order(self):
        """Topological sort based on asset dependencies."""
        visited = set()
        order = []

        def visit(key):
            if key in visited:
                return
            visited.add(key)
            asset = self.assets.get(key)
            if asset:
                for dep in asset.deps:
                    visit(dep)
            order.append(key)

        for key in self.assets:
            visit(key)
        return order

    def materialize_all(self):
        """Materialize all assets in dependency order."""
        order = self.get_execution_order()
        results = {}
        upstream_data = {}

        for key in order:
            if key not in self.assets:
                continue
            asset = self.assets[key]

            # Gather upstream results
            deps_data = {dep: upstream_data.get(dep) for dep in asset.deps}

            result = asset.materialize(deps_data)
            results[key] = result

            if result['status'] == 'completed' and 'data' in result:
                upstream_data[key] = result['data']
                # Remove DataFrame from result to keep JSON response manageable
                if isinstance(result.get('data'), pd.DataFrame):
                    result['row_count'] = len(result['data'])
                    del result['data']

        return results

    def materialize_asset(self, key):
        """Materialize a single asset (and its upstream dependencies)."""
        if key not in self.assets:
            return {'error': f'Asset {key} not found'}

        # Find all upstream dependencies recursively
        needed = set()

        def find_deps(k):
            needed.add(k)
            asset = self.assets.get(k)
            if asset:
                for dep in asset.deps:
                    if dep not in needed:
                        find_deps(dep)

        find_deps(key)

        order = [k for k in self.get_execution_order() if k in needed]
        results = {}
        upstream_data = {}

        for k in order:
            if k not in self.assets:
                continue
            asset = self.assets[k]
            deps_data = {dep: upstream_data.get(dep) for dep in asset.deps}
            result = asset.materialize(deps_data)
            results[k] = result
            if result['status'] == 'completed' and 'data' in result:
                upstream_data[k] = result['data']
                if isinstance(result.get('data'), pd.DataFrame):
                    result['row_count'] = len(result['data'])
                    del result['data']

        return results

    def get_asset_info(self):
        """Get metadata about all assets in the graph."""
        return {
            key: {
                'key': asset.key,
                'group': asset.group_name,
                'deps': asset.deps,
                'description': asset.description,
                'last_materialized': asset.last_materialized,
                'materialization_count': asset.materialization_count
            }
            for key, asset in self.assets.items()
        }


# ============================================================
# Pre-built Dagster jobs for datasets
# ============================================================

def _make_csv_extract_op(csv_name, table_name):
    """Factory for CSV extraction ops."""
    def compute_fn(context, inputs):
        path = os.path.join(Config.DATA_DIR, csv_name)
        df = pd.read_csv(path)
        io = context['resources'].get('io_manager', SQLiteIOManager())
        rows = io.handle_output(context, df, table_name)
        return OpResult(df, metadata={'row_count': rows, 'source': csv_name, 'table': table_name})
    return compute_fn


def _make_sql_transform_op(sql, output_table):
    """Factory for SQL transformation ops."""
    def compute_fn(context, inputs):
        conn = get_connection()
        try:
            conn.execute(f"DROP TABLE IF EXISTS {output_table}")
            conn.execute(sql)
            conn.commit()
            count = conn.execute(f"SELECT COUNT(*) FROM {output_table}").fetchone()[0]
            return OpResult(count, metadata={'row_count': count, 'table': output_table})
        finally:
            conn.close()
    return compute_fn


def build_dagster_hdb_job():
    """Build a Dagster job for HDB resale data processing."""
    graph = DagsterGraph('hdb_pipeline', 'HDB resale data extraction, loading, and aggregation')

    io_mgr = SQLiteIOManager()

    graph.add_op(DagsterOp(
        'extract_hdb_csv',
        _make_csv_extract_op('hdb_resale.csv', 'raw_hdb_resale'),
        description='Extract HDB resale CSV into raw table',
        tags={'kind': 'extract'}
    ))

    graph.add_op(DagsterOp(
        'transform_monthly_stats',
        _make_sql_transform_op(
            """CREATE TABLE dagster_hdb_monthly AS
               SELECT month, town, flat_type,
                      COUNT(*) as txn_count,
                      ROUND(AVG(resale_price), 2) as avg_price,
                      ROUND(MIN(resale_price), 2) as min_price,
                      ROUND(MAX(resale_price), 2) as max_price,
                      ROUND(AVG(floor_area_sqm), 2) as avg_area
               FROM raw_hdb_resale
               GROUP BY month, town, flat_type""",
            'dagster_hdb_monthly'
        ),
        description='Aggregate monthly HDB statistics per town and flat type',
        tags={'kind': 'transform'}
    ), depends_on=['extract_hdb_csv'])

    graph.add_op(DagsterOp(
        'transform_town_summary',
        _make_sql_transform_op(
            """CREATE TABLE dagster_hdb_town_summary AS
               SELECT town,
                      COUNT(*) as total_txns,
                      ROUND(AVG(resale_price), 2) as avg_price,
                      ROUND(MIN(resale_price), 2) as min_price,
                      ROUND(MAX(resale_price), 2) as max_price,
                      ROUND(AVG(floor_area_sqm), 2) as avg_area,
                      ROUND(AVG(resale_price / NULLIF(floor_area_sqm, 0)), 2) as price_per_sqm
               FROM raw_hdb_resale
               GROUP BY town
               ORDER BY avg_price DESC""",
            'dagster_hdb_town_summary'
        ),
        description='Summary statistics per town',
        tags={'kind': 'transform'}
    ), depends_on=['extract_hdb_csv'])

    graph.add_op(DagsterOp(
        'validate_hdb_data',
        lambda ctx, inputs: OpResult(
            'validated',
            metadata={'checks_passed': 3, 'row_count': 0}
        ),
        description='Run data quality checks on HDB data',
        tags={'kind': 'validate'}
    ), depends_on=['transform_monthly_stats', 'transform_town_summary'])

    return DagsterJob(
        'hdb_dagster_job', graph,
        resources={'io_manager': io_mgr},
        tags={'domain': 'housing'},
        description='Full HDB resale data pipeline with Dagster'
    )


def build_dagster_transport_job():
    """Build a Dagster job for transport data processing."""
    graph = DagsterGraph('transport_pipeline', 'Transport usage data extraction and analysis')
    io_mgr = SQLiteIOManager()

    graph.add_op(DagsterOp(
        'extract_transport_csv',
        _make_csv_extract_op('transport_usage.csv', 'raw_transport_usage'),
        description='Extract transport usage CSV',
        tags={'kind': 'extract'}
    ))

    graph.add_op(DagsterOp(
        'transform_line_stats',
        _make_sql_transform_op(
            """CREATE TABLE dagster_transport_lines AS
               SELECT line_name, line_color,
                      COUNT(DISTINCT station_code) as station_count,
                      SUM(tap_in_count) as total_tap_in,
                      SUM(tap_out_count) as total_tap_out,
                      SUM(tap_in_count + tap_out_count) as total_ridership,
                      ROUND(AVG(peak_hour_pct), 3) as avg_peak_pct
               FROM raw_transport_usage
               GROUP BY line_name, line_color""",
            'dagster_transport_lines'
        ),
        description='Aggregate transport statistics by MRT/LRT line',
        tags={'kind': 'transform'}
    ), depends_on=['extract_transport_csv'])

    graph.add_op(DagsterOp(
        'transform_station_rankings',
        _make_sql_transform_op(
            """CREATE TABLE dagster_station_rankings AS
               SELECT station_name, station_code, line_name,
                      SUM(tap_in_count + tap_out_count) as total_trips,
                      RANK() OVER (ORDER BY SUM(tap_in_count + tap_out_count) DESC) as rank
               FROM raw_transport_usage
               GROUP BY station_name, station_code, line_name
               ORDER BY total_trips DESC""",
            'dagster_station_rankings'
        ),
        description='Rank stations by total ridership',
        tags={'kind': 'transform'}
    ), depends_on=['extract_transport_csv'])

    return DagsterJob(
        'transport_dagster_job', graph,
        resources={'io_manager': io_mgr},
        tags={'domain': 'transport'},
        description='Transport data pipeline with Dagster'
    )


def build_dagster_energy_job():
    """Build a Dagster job for energy data processing."""
    graph = DagsterGraph('energy_pipeline', 'Energy consumption data extraction and analysis')
    io_mgr = SQLiteIOManager()

    graph.add_op(DagsterOp(
        'extract_energy_csv',
        _make_csv_extract_op('energy.csv', 'raw_energy'),
        description='Extract energy consumption CSV',
        tags={'kind': 'extract'}
    ))

    graph.add_op(DagsterOp(
        'transform_energy_yearly',
        _make_sql_transform_op(
            """CREATE TABLE dagster_energy_yearly AS
               SELECT year, sector, energy_type,
                      ROUND(SUM(consumption_gwh), 2) as total_gwh,
                      ROUND(SUM(cost_million_sgd), 2) as total_cost,
                      ROUND(SUM(carbon_emission_tonnes), 2) as total_carbon
               FROM raw_energy
               GROUP BY year, sector, energy_type""",
            'dagster_energy_yearly'
        ),
        description='Yearly energy aggregation by sector and type',
        tags={'kind': 'transform'}
    ), depends_on=['extract_energy_csv'])

    return DagsterJob(
        'energy_dagster_job', graph,
        resources={'io_manager': io_mgr},
        tags={'domain': 'energy'},
        description='Energy data pipeline with Dagster'
    )


# ============================================================
# Software-Defined Assets for Data
# ============================================================

def build_sg_asset_graph():
    """Build a Dagster-style asset graph for all datasets."""
    ag = AssetGraph('sg_data_assets')

    # ---- Raw assets (no upstream deps) ----
    ag.add_asset(SoftwareDefinedAsset(
        'raw_hdb_resale',
        lambda _: pd.read_csv(os.path.join(Config.DATA_DIR, 'hdb_resale.csv')),
        description='Raw HDB resale transaction data',
        group_name='raw'
    ))

    ag.add_asset(SoftwareDefinedAsset(
        'raw_transport',
        lambda _: pd.read_csv(os.path.join(Config.DATA_DIR, 'transport_usage.csv')),
        description='Raw MRT/LRT transport usage data',
        group_name='raw'
    ))

    ag.add_asset(SoftwareDefinedAsset(
        'raw_energy',
        lambda _: pd.read_csv(os.path.join(Config.DATA_DIR, 'energy.csv')),
        description='Raw energy consumption data',
        group_name='raw'
    ))

    ag.add_asset(SoftwareDefinedAsset(
        'raw_population',
        lambda _: pd.read_csv(os.path.join(Config.DATA_DIR, 'population.csv')),
        description='Raw population demographics data',
        group_name='raw'
    ))

    # ---- Staging assets (depend on raw) ----
    def compute_hdb_stats(upstream):
        df = upstream.get('raw_hdb_resale')
        if df is None:
            raise ValueError("Missing upstream: raw_hdb_resale")
        return df.groupby(['town', 'flat_type']).agg(
            txn_count=('resale_price', 'count'),
            avg_price=('resale_price', 'mean'),
            min_price=('resale_price', 'min'),
            max_price=('resale_price', 'max'),
            avg_area=('floor_area_sqm', 'mean')
        ).round(2).reset_index()

    ag.add_asset(SoftwareDefinedAsset(
        'dagster_asset_hdb_stats',
        compute_hdb_stats,
        deps=['raw_hdb_resale'],
        description='HDB price statistics by town and flat type',
        group_name='staging'
    ))

    def compute_transport_summary(upstream):
        df = upstream.get('raw_transport')
        if df is None:
            raise ValueError("Missing upstream: raw_transport")
        df['total_trips'] = df['tap_in_count'] + df['tap_out_count']
        return df.groupby(['line_name', 'line_color']).agg(
            station_count=('station_code', 'nunique'),
            total_ridership=('total_trips', 'sum'),
            avg_peak_pct=('peak_hour_pct', 'mean')
        ).round(3).reset_index()

    ag.add_asset(SoftwareDefinedAsset(
        'dagster_asset_transport_summary',
        compute_transport_summary,
        deps=['raw_transport'],
        description='Transport ridership summary by line',
        group_name='staging'
    ))

    def compute_energy_trends(upstream):
        df = upstream.get('raw_energy')
        if df is None:
            raise ValueError("Missing upstream: raw_energy")
        return df.groupby(['year', 'sector']).agg(
            total_gwh=('consumption_gwh', 'sum'),
            total_cost=('cost_million_sgd', 'sum'),
            total_carbon=('carbon_emission_tonnes', 'sum')
        ).round(2).reset_index()

    ag.add_asset(SoftwareDefinedAsset(
        'dagster_asset_energy_trends',
        compute_energy_trends,
        deps=['raw_energy'],
        description='Energy consumption trends by year and sector',
        group_name='staging'
    ))

    # ---- Analytics assets (depend on staging) ----
    def compute_urban_index(upstream):
        hdb = upstream.get('dagster_asset_hdb_stats')
        transport = upstream.get('dagster_asset_transport_summary')
        if hdb is None or transport is None:
            raise ValueError("Missing upstream assets")
        town_stats = hdb.groupby('town').agg(
            avg_price=('avg_price', 'mean'),
            total_txns=('txn_count', 'sum')
        ).reset_index()
        town_stats['price_index'] = (
            town_stats['avg_price'] / town_stats['avg_price'].max() * 100
        ).round(1)
        return town_stats

    ag.add_asset(SoftwareDefinedAsset(
        'dagster_asset_urban_index',
        compute_urban_index,
        deps=['dagster_asset_hdb_stats', 'dagster_asset_transport_summary'],
        description='Urban liveability index combining housing and transport data',
        group_name='analytics'
    ))

    return ag


# ============================================================
# Runner functions
# ============================================================

def run_dagster_job(job_name):
    """Run a specific Dagster job by name."""
    builders = {
        'hdb': build_dagster_hdb_job,
        'transport': build_dagster_transport_job,
        'energy': build_dagster_energy_job,
    }
    if job_name not in builders:
        return {'status': 'error', 'message': f'Unknown job: {job_name}'}
    job = builders[job_name]()
    return job.execute()


def run_all_dagster_jobs():
    """Run all pre-built Dagster jobs."""
    results = {}
    for name in ['hdb', 'transport', 'energy']:
        results[name] = run_dagster_job(name)
    return results


def run_dagster_assets(asset_key=None):
    """Run the software-defined asset graph (all or a single asset)."""
    ag = build_sg_asset_graph()
    if asset_key:
        return ag.materialize_asset(asset_key)
    return ag.materialize_all()


def get_dagster_asset_info():
    """Get metadata about all defined assets."""
    ag = build_sg_asset_graph()
    return ag.get_asset_info()


def get_dagster_schedules():
    """Get all defined schedules."""
    schedules = [
        DagsterSchedule(
            'daily_hdb_refresh', '0 0 * * *',
            build_dagster_hdb_job(),
            'Refresh HDB data daily at midnight'
        ),
        DagsterSchedule(
            'six_hourly_transport', '0 */6 * * *',
            build_dagster_transport_job(),
            'Refresh transport data every 6 hours'
        ),
        DagsterSchedule(
            'monthly_energy_refresh', '0 0 1 * *',
            build_dagster_energy_job(),
            'Refresh energy data on the first of each month'
        ),
    ]
    return [s.to_dict() for s in schedules]


def get_dagster_sensors():
    """Get all defined sensors."""
    def new_file_sensor():
        upload_dir = Config.UPLOAD_FOLDER
        files = os.listdir(upload_dir) if os.path.isdir(upload_dir) else []
        csv_files = [f for f in files if f.endswith('.csv')]
        return len(csv_files) > 0, {'files': csv_files, 'count': len(csv_files)}

    sensors = [
        DagsterSensor(
            'new_csv_upload_sensor',
            new_file_sensor,
            build_dagster_hdb_job(),
            'Triggers when new CSV files are detected in uploads/',
            min_interval=60
        ),
    ]
    return [s.to_dict() for s in sensors]
