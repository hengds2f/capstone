import pytest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from models.database import get_connection, query_db, execute_db


@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestDatabaseConnection:
    def test_connection(self):
        conn = get_connection()
        assert conn is not None
        conn.close()

    def test_query_db(self):
        result = query_db("SELECT 1 as test")
        assert result is not None

    def test_get_all_tables(self):
        from models.database import get_all_tables
        tables = get_all_tables()
        assert isinstance(tables, list)


class TestSchema:
    def test_tables_exist_after_init(self, app):
        with app.app_context():
            from models.database import table_exists
            expected = ['raw_hdb_resale', 'dim_location', 'dim_time',
                        'fact_hdb_transactions', 'model_metrics',
                        'pipeline_runs', 'api_logs', 'stream_events']
            for table in expected:
                assert table_exists(table), f"Table {table} should exist"

    def test_row_count(self, app):
        with app.app_context():
            from models.database import get_row_count
            count = get_row_count('raw_hdb_resale')
            assert isinstance(count, int)
            assert count >= 0
