import pytest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app


@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestPipeline:
    def test_pipeline_run(self, client):
        response = client.post('/api/pipeline/run')
        assert response.status_code == 200
        data = response.get_json()
        assert 'results' in data

    def test_pipeline_history(self, client):
        response = client.get('/api/pipeline/history')
        assert response.status_code == 200
        data = response.get_json()
        assert 'history' in data

    def test_pipeline_populates_data(self, client):
        """After running pipeline, tables should have data."""
        client.post('/api/pipeline/run')
        response = client.get('/api/summary')
        data = response.get_json()
        table_names = [t['name'] for t in data['tables']]
        assert 'raw_hdb_resale' in table_names


class TestValidation:
    def test_validation_runs(self, client):
        # Load data first
        client.post('/api/pipeline/run')
        response = client.get('/api/validate')
        assert response.status_code == 200
        data = response.get_json()
        assert 'results' in data
