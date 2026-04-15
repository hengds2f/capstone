"""Stream processing simulation for Singapore data events."""
import json
import random
import time
from datetime import datetime, timedelta
from models.database import execute_db, query_db


# Simulated event generators
MRT_STATIONS = ['Raffles Place', 'City Hall', 'Orchard', 'Ang Mo Kio', 'Bishan',
                'Jurong East', 'Tampines', 'Woodlands', 'Punggol', 'Sengkang']

def generate_mrt_event():
    """Generate a simulated MRT crowding event."""
    return {
        'event_type': 'mrt_crowding',
        'source': 'transport_feed',
        'payload': json.dumps({
            'station': random.choice(MRT_STATIONS),
            'crowd_level': random.choice(['Low', 'Medium', 'High', 'Very High']),
            'passenger_count': random.randint(50, 500),
            'timestamp': datetime.now().isoformat()
        })
    }


def generate_housing_event():
    """Generate a simulated housing listing event."""
    towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'TAMPINES', 'WOODLANDS',
             'JURONG WEST', 'SENGKANG', 'PUNGGOL', 'HOUGANG', 'YISHUN']
    flat_types = ['3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']
    return {
        'event_type': 'housing_listing',
        'source': 'hdb_feed',
        'payload': json.dumps({
            'town': random.choice(towns),
            'flat_type': random.choice(flat_types),
            'price': random.randint(280000, 900000),
            'floor_area_sqm': random.randint(60, 140),
            'timestamp': datetime.now().isoformat()
        })
    }


def generate_weather_event():
    """Generate a simulated weather/air quality event."""
    regions = ['North', 'South', 'East', 'West', 'Central']
    return {
        'event_type': 'weather_update',
        'source': 'nea_feed',
        'payload': json.dumps({
            'region': random.choice(regions),
            'temperature_c': round(random.uniform(24, 35), 1),
            'humidity_pct': random.randint(60, 95),
            'psi': random.randint(20, 120),
            'rainfall_mm': round(random.uniform(0, 50), 1),
            'timestamp': datetime.now().isoformat()
        })
    }


def generate_event_batch(batch_size=10):
    """Generate a batch of mixed events."""
    generators = [generate_mrt_event, generate_housing_event, generate_weather_event]
    events = []
    for _ in range(batch_size):
        gen = random.choice(generators)
        events.append(gen())
    return events


def ingest_events(events):
    """Store events in the stream_events table."""
    for event in events:
        execute_db(
            "INSERT INTO stream_events (event_type, source, payload) VALUES (?,?,?)",
            (event['event_type'], event['source'], event['payload'])
        )
    return len(events)


def process_window(window_minutes=5):
    """Process events within a time window for aggregation."""
    cutoff = (datetime.now() - timedelta(minutes=window_minutes)).isoformat()

    # Get unprocessed events
    events = query_db(
        "SELECT * FROM stream_events WHERE processed = 0 AND event_time >= ?",
        (cutoff,)
    )

    aggregations = {
        'mrt_crowding': {'count': 0, 'stations': {}, 'total_passengers': 0},
        'housing_listing': {'count': 0, 'avg_price': 0, 'prices': []},
        'weather_update': {'count': 0, 'avg_temp': 0, 'temps': [], 'avg_psi': 0, 'psis': []}
    }

    event_ids = []
    for event in events:
        event_ids.append(event['event_id'])
        payload = json.loads(event['payload'])
        etype = event['event_type']

        if etype == 'mrt_crowding':
            aggregations[etype]['count'] += 1
            station = payload.get('station', 'Unknown')
            aggregations[etype]['stations'][station] = aggregations[etype]['stations'].get(station, 0) + 1
            aggregations[etype]['total_passengers'] += payload.get('passenger_count', 0)
        elif etype == 'housing_listing':
            aggregations[etype]['count'] += 1
            aggregations[etype]['prices'].append(payload.get('price', 0))
        elif etype == 'weather_update':
            aggregations[etype]['count'] += 1
            aggregations[etype]['temps'].append(payload.get('temperature_c', 0))
            aggregations[etype]['psis'].append(payload.get('psi', 0))

    # Calculate averages
    if aggregations['housing_listing']['prices']:
        aggregations['housing_listing']['avg_price'] = round(
            sum(aggregations['housing_listing']['prices']) / len(aggregations['housing_listing']['prices']), 2)
        del aggregations['housing_listing']['prices']

    if aggregations['weather_update']['temps']:
        aggregations['weather_update']['avg_temp'] = round(
            sum(aggregations['weather_update']['temps']) / len(aggregations['weather_update']['temps']), 1)
        aggregations['weather_update']['avg_psi'] = round(
            sum(aggregations['weather_update']['psis']) / len(aggregations['weather_update']['psis']), 0)
        del aggregations['weather_update']['temps']
        del aggregations['weather_update']['psis']

    # Mark events as processed
    if event_ids:
        placeholders = ','.join('?' for _ in event_ids)
        from models.database import get_connection
        conn = get_connection()
        conn.execute(f"UPDATE stream_events SET processed = 1 WHERE event_id IN ({placeholders})", event_ids)
        conn.commit()
        conn.close()

    # Anomaly detection
    anomalies = detect_anomalies(aggregations)
    aggregations['anomalies'] = anomalies

    return aggregations


def detect_anomalies(aggregations):
    """Simple anomaly detection on streaming data."""
    anomalies = []

    # Check for very high MRT crowding
    if aggregations['mrt_crowding']['total_passengers'] > 2000:
        anomalies.append({
            'type': 'high_crowding',
            'message': f"High total passenger count: {aggregations['mrt_crowding']['total_passengers']}",
            'severity': 'warning'
        })

    # Check for housing price spike
    if aggregations['housing_listing']['avg_price'] > 700000:
        anomalies.append({
            'type': 'price_spike',
            'message': f"Average listing price above threshold: ${aggregations['housing_listing']['avg_price']:,.0f}",
            'severity': 'info'
        })

    # Check for poor air quality
    if aggregations['weather_update']['avg_psi'] > 80:
        anomalies.append({
            'type': 'poor_air_quality',
            'message': f"Average PSI above 80: {aggregations['weather_update']['avg_psi']:.0f}",
            'severity': 'warning'
        })

    return anomalies


def get_stream_stats():
    """Get stream processing statistics."""
    total = query_db("SELECT COUNT(*) as cnt FROM stream_events", one=True)
    processed = query_db("SELECT COUNT(*) as cnt FROM stream_events WHERE processed = 1", one=True)
    by_type = query_db(
        "SELECT event_type, COUNT(*) as cnt FROM stream_events GROUP BY event_type"
    )
    return {
        'total_events': total['cnt'] if total else 0,
        'processed_events': processed['cnt'] if processed else 0,
        'by_type': {r['event_type']: r['cnt'] for r in by_type}
    }
