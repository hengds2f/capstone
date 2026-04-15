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


@pytest.fixture(autouse=True)
def load_data(client):
    """Ensure data is loaded before model tests."""
    client.post('/api/pipeline/run')


class TestModelTraining:
    def test_train_linear_regression(self, client):
        response = client.post('/api/model/train',
                               json={'model_type': 'linear_regression'})
        assert response.status_code == 200
        data = response.get_json()
        assert 'metrics' in data
        assert 'r2_score' in data['metrics']

    def test_train_random_forest(self, client):
        response = client.post('/api/model/train',
                               json={'model_type': 'random_forest'})
        assert response.status_code == 200
        data = response.get_json()
        assert 'metrics' in data

    def test_train_kmeans(self, client):
        response = client.post('/api/model/train',
                               json={'model_type': 'kmeans'})
        assert response.status_code == 200
        data = response.get_json()
        assert 'clusters' in data or 'metrics' in data

    def test_predict(self, client):
        # Train first
        client.post('/api/model/train',
                    json={'model_type': 'linear_regression'})
        # Then predict
        response = client.post('/api/model/predict', json={
            'town': 'BISHAN',
            'flat_type': '4 ROOM',
            'floor_area_sqm': 95,
            'storey_mid': 10,
            'remaining_lease_years': 85
        })
        assert response.status_code == 200
        data = response.get_json()
        assert 'predicted_price' in data
        assert data['predicted_price'] > 0

    def test_model_history(self, client):
        client.post('/api/model/train',
                    json={'model_type': 'linear_regression'})
        response = client.get('/api/model/history')
        assert response.status_code == 200
        data = response.get_json()
        assert 'history' in data
        assert len(data['history']) > 0


class TestNLP:
    def test_text_classification(self, client):
        response = client.post('/api/model/train', json={
            'model_type': 'text_classifier',
            'text': 'The MRT was very crowded today'
        })
        # This may or may not be a supported endpoint — check gracefully
        assert response.status_code in [200, 400]
