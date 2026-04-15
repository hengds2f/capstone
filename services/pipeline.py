"""Pipeline orchestration service."""
import time
from datetime import datetime
from models.database import execute_db, query_db
from services.ingestion import (
    ingest_hdb_data, ingest_transport_data, ingest_population_data,
    ingest_school_data, ingest_energy_data, ingest_feedback_data
)
from services.validation import run_all_validations
from config import Config
import os


class PipelineOrchestrator:
    """Simple pipeline orchestrator with dependency ordering and logging."""

    def __init__(self):
        self.steps = []
        self.run_id = None

    def add_step(self, name, func, depends_on=None):
        self.steps.append({
            'name': name,
            'func': func,
            'depends_on': depends_on or [],
            'status': 'pending',
            'result': None
        })

    def _log_step(self, pipeline_name, step_name, status, rows=0, error=None):
        execute_db(
            "INSERT INTO pipeline_runs (pipeline_name, step_name, status, started_at, completed_at, rows_processed, error_message) VALUES (?,?,?,?,?,?,?)",
            (pipeline_name, step_name, status, datetime.now().isoformat(),
             datetime.now().isoformat() if status in ('completed', 'failed') else None,
             rows, error)
        )

    def run(self, pipeline_name='full_pipeline'):
        """Execute pipeline steps in dependency order."""
        completed = set()
        results = {}

        for step in self.steps:
            # Check dependencies
            unmet = [d for d in step['depends_on'] if d not in completed]
            if unmet:
                step['status'] = 'skipped'
                self._log_step(pipeline_name, step['name'], 'skipped',
                               error=f"Unmet dependencies: {unmet}")
                continue

            self._log_step(pipeline_name, step['name'], 'running')
            try:
                result = step['func']()
                step['status'] = 'completed'
                step['result'] = result
                completed.add(step['name'])
                rows = result if isinstance(result, int) else 0
                self._log_step(pipeline_name, step['name'], 'completed', rows=rows)
                results[step['name']] = {'status': 'completed', 'result': result}
            except Exception as e:
                step['status'] = 'failed'
                self._log_step(pipeline_name, step['name'], 'failed', error=str(e))
                results[step['name']] = {'status': 'failed', 'error': str(e)}

        return results


def build_full_pipeline():
    """Build and return the full data pipeline."""
    data_dir = Config.DATA_DIR
    orch = PipelineOrchestrator()

    orch.add_step('ingest_hdb', lambda: ingest_hdb_data(os.path.join(data_dir, 'hdb_resale.csv')))
    orch.add_step('ingest_transport', lambda: ingest_transport_data(os.path.join(data_dir, 'transport_usage.csv')))
    orch.add_step('ingest_population', lambda: ingest_population_data(os.path.join(data_dir, 'population.csv')))
    orch.add_step('ingest_schools', lambda: ingest_school_data(os.path.join(data_dir, 'schools.csv')))
    orch.add_step('ingest_energy', lambda: ingest_energy_data(os.path.join(data_dir, 'energy.csv')))
    orch.add_step('ingest_feedback', lambda: ingest_feedback_data(os.path.join(data_dir, 'sample_feedback.csv')))
    orch.add_step('validate_data', lambda: run_all_validations(),
                  depends_on=['ingest_hdb', 'ingest_transport', 'ingest_population', 'ingest_schools', 'ingest_energy'])

    return orch


def get_pipeline_history():
    """Get recent pipeline run history."""
    return query_db(
        "SELECT * FROM pipeline_runs ORDER BY run_id DESC LIMIT 50"
    )
