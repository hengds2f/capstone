import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'datascience-lab-dev-key')
    DATABASE = os.path.join(BASE_DIR, 'database', 'datascience.db')
    SCHEMA_FILE = os.path.join(BASE_DIR, 'database', 'schema.sql')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    DEBUG = True
