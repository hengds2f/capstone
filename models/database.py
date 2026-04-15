"""Database connection and helper utilities."""
import sqlite3
import os
from config import Config


def get_db_path():
    return Config.DATABASE


def get_connection():
    os.makedirs(os.path.dirname(get_db_path()), exist_ok=True)
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    with open(Config.SCHEMA_FILE, 'r') as f:
        conn.executescript(f.read())
    conn.close()


def query_db(query, args=(), one=False):
    conn = get_connection()
    cur = conn.execute(query, args)
    rv = cur.fetchall()
    conn.close()
    return (rv[0] if rv else None) if one else rv


def execute_db(query, args=()):
    conn = get_connection()
    try:
        cur = conn.execute(query, args)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def executemany_db(query, args_list):
    conn = get_connection()
    try:
        conn.executemany(query, args_list)
        conn.commit()
    finally:
        conn.close()


def execute_script(script):
    conn = get_connection()
    try:
        conn.executescript(script)
    finally:
        conn.close()


def table_exists(table_name):
    row = query_db(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,), one=True
    )
    return row is not None


def get_table_info(table_name):
    return query_db(f"PRAGMA table_info({table_name})")


def get_all_tables():
    rows = query_db("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [r['name'] for r in rows]


def get_row_count(table_name):
    row = query_db(f"SELECT COUNT(*) as cnt FROM {table_name}", one=True)
    return row['cnt'] if row else 0
