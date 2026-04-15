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


class TestPageRoutes:
    """Test that all page routes return 200."""
    pages = ['/', '/ingestion', '/sql', '/python', '/eda', '/visualization',
             '/bigdata', '/architecture', '/dataflow', '/webscraping',
             '/warehouse', '/pipeline', '/outofcore', '/distributed',
             '/streaming', '/probability', '/supervised', '/unsupervised',
             '/timeseries', '/deployment', '/admin']

    @pytest.mark.parametrize('page', pages)
    def test_page_loads(self, client, page):
        response = client.get(page)
        assert response.status_code == 200, f"Page {page} returned {response.status_code}"


class TestAPIEndpoints:
    def test_summary(self, client):
        response = client.get('/api/summary')
        assert response.status_code == 200
        data = response.get_json()
        assert 'tables' in data

    def test_housing(self, client):
        response = client.get('/api/housing')
        assert response.status_code == 200
        data = response.get_json()
        assert 'data' in data

    def test_transport(self, client):
        response = client.get('/api/transport')
        assert response.status_code == 200

    def test_eda(self, client):
        response = client.get('/api/eda')
        assert response.status_code == 200

    def test_validate(self, client):
        response = client.get('/api/validate')
        assert response.status_code == 200

    def test_sql_select(self, client):
        response = client.post('/api/sql/execute',
                               json={'query': 'SELECT 1 as test'})
        assert response.status_code == 200
        data = response.get_json()
        assert 'results' in data

    def test_sql_blocks_write(self, client):
        """SQL endpoint must block write operations."""
        dangerous_queries = [
            'DROP TABLE raw_hdb_resale',
            'DELETE FROM raw_hdb_resale',
            'INSERT INTO raw_hdb_resale VALUES (1)',
            'UPDATE raw_hdb_resale SET town="X"',
        ]
        for q in dangerous_queries:
            response = client.post('/api/sql/execute', json={'query': q})
            assert response.status_code == 400, f"Should block: {q}"

    def test_feedback(self, client):
        response = client.post('/api/feedback', json={
            'module': 'test',
            'rating': 5,
            'comment': 'Great module!'
        })
        assert response.status_code == 200

    def test_stream_generate(self, client):
        response = client.post('/api/stream/generate', json={'count': 5})
        assert response.status_code == 200
        data = response.get_json()
        assert 'events' in data

    def test_stream_stats(self, client):
        response = client.get('/api/stream/stats')
        assert response.status_code == 200
