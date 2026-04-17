"""Web scraping service: Extract tabular data from web pages, HTML tables, and APIs."""
import pandas as pd
import requests
import json
import re
import os
from datetime import datetime
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from models.database import get_connection, execute_db
from config import Config

# Common data source domains
ALLOWED_DOMAINS = [
    'data.gov.sg', 'tablesgenerator.com', 'en.wikipedia.org',
    'www.singstat.gov.sg', 'www.hdb.gov.sg', 'datamall.lta.gov.sg',
    'raw.githubusercontent.com', 'github.com', 'kaggle.com',
    'www.kaggle.com', 'archive.ics.uci.edu', 'people.sc.fsu.edu',
    'ourworldindata.org', 'www.worldometers.info',
    'data.worldbank.org', 'api.worldbank.org', 'data.un.org',
]

REQUEST_TIMEOUT = 15
MAX_TABLES = 20
USER_AGENT = 'DataScience-Lab/1.0 (Educational Purpose)'


def _is_domain_allowed(url):
    """Check if domain is in the allowed list; allow all for educational flexibility."""
    # For educational purposes we allow all domains but flag unknown ones
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    return domain in ALLOWED_DOMAINS, domain


def _fetch_page(url):
    """Fetch a web page with proper headers and timeout."""
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError("Only http/https URLs are supported")

    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


def scrape_html_tables(url, table_index=None):
    """Scrape HTML tables from a web page using pandas.read_html().

    Args:
        url: The web page URL to scrape
        table_index: Optional specific table index to extract (0-based).
                     If None, returns all tables found.

    Returns:
        dict with tables, metadata, and preview
    """
    is_known, domain = _is_domain_allowed(url)

    # Use pandas read_html for robust table extraction
    try:
        tables = pd.read_html(url, storage_options={'User-Agent': USER_AGENT})
    except Exception:
        # Fallback: fetch page manually and parse
        resp = _fetch_page(url)
        tables = pd.read_html(resp.text)

    if not tables:
        return {'status': 'no_tables', 'message': 'No HTML tables found on this page', 'tables': []}

    tables = tables[:MAX_TABLES]  # Limit

    if table_index is not None:
        if table_index < 0 or table_index >= len(tables):
            return {'status': 'error', 'message': f'Table index {table_index} out of range (0-{len(tables)-1})'}
        tables = [tables[table_index]]

    result = {
        'status': 'ok',
        'url': url,
        'domain': domain,
        'known_domain': is_known,
        'tables_found': len(tables),
        'tables': []
    }

    for i, df in enumerate(tables):
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        # Drop fully empty rows/cols
        df = df.dropna(how='all').dropna(axis=1, how='all')

        table_info = {
            'index': i if table_index is None else table_index,
            'rows': len(df),
            'columns': list(df.columns),
            'preview': df.head(5).to_dict(orient='records'),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        result['tables'].append(table_info)

    return result


def scrape_and_load_table(url, table_index, table_name, if_exists='replace'):
    """Scrape an HTML table and load it directly into the database.

    Args:
        url: Web page URL
        table_index: Which table to extract (0-based index)
        table_name: Target SQLite table name
        if_exists: 'replace' or 'append'

    Returns:
        dict with load results
    """
    try:
        tables = pd.read_html(url, storage_options={'User-Agent': USER_AGENT})
    except Exception:
        resp = _fetch_page(url)
        tables = pd.read_html(resp.text)

    if not tables:
        raise ValueError("No tables found on page")
    if table_index < 0 or table_index >= len(tables):
        raise ValueError(f"Table index {table_index} out of range (0-{len(tables)-1})")

    df = tables[table_index]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how='all').dropna(axis=1, how='all')

    if df.empty:
        raise ValueError("Selected table is empty after cleaning")

    conn = get_connection()
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        execute_db(
            "INSERT INTO pipeline_runs (pipeline_name, step_name, status, started_at, completed_at, rows_processed) VALUES (?,?,?,?,?,?)",
            ('web_scraping', f'scrape_{table_name}', 'completed',
             datetime.now().isoformat(), datetime.now().isoformat(), len(df))
        )
        return {
            'status': 'ok',
            'rows_loaded': len(df),
            'columns': list(df.columns),
            'table_name': table_name,
            'source_url': url,
            'table_index': table_index
        }
    finally:
        conn.close()


def scrape_page_links(url, pattern=None):
    """Scrape all links from a page, optionally filtering by regex pattern.

    Useful for discovering dataset download links on index pages.
    """
    resp = _fetch_page(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(url, href)
        text = a.get_text(strip=True)

        if pattern and not re.search(pattern, full_url, re.IGNORECASE):
            continue

        links.append({
            'url': full_url,
            'text': text[:200],
        })

    # Deduplicate by URL
    seen = set()
    unique_links = []
    for link in links:
        if link['url'] not in seen:
            seen.add(link['url'])
            unique_links.append(link)

    return {
        'status': 'ok',
        'url': url,
        'total_links': len(unique_links),
        'links': unique_links[:100],  # Limit to 100
        'pattern': pattern
    }


def scrape_page_text(url, selector=None):
    """Scrape structured text content from a page.

    Args:
        url: Page URL
        selector: Optional CSS selector to target specific elements

    Returns:
        Extracted text data
    """
    resp = _fetch_page(url)
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Remove script and style tags
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()

    if selector:
        elements = soup.select(selector)
        texts = [el.get_text(strip=True) for el in elements if el.get_text(strip=True)]
    else:
        # Get main content
        main = soup.find('main') or soup.find('article') or soup.find('body')
        texts = [p.get_text(strip=True) for p in main.find_all(['p', 'li', 'h1', 'h2', 'h3', 'h4', 'td', 'th'])
                 if p.get_text(strip=True)]

    return {
        'status': 'ok',
        'url': url,
        'element_count': len(texts),
        'content': texts[:200],  # Limit
        'selector': selector
    }


def scrape_api_json(url, params=None):
    """Fetch data from a JSON API endpoint and convert to a DataFrame.

    Args:
        url: API endpoint URL
        params: Optional query parameters dict

    Returns:
        dict with data preview and metadata
    """
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError("Only http/https URLs are supported")

    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'application/json',
    }
    resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    data = resp.json()

    # Try to find a list of records in the response
    records = None
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # Common API patterns: data.result.records, data.records, data.data, data.results
        for key in ['records', 'data', 'results', 'items', 'result', 'value', 'rows']:
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
            if 'result' in data and isinstance(data['result'], dict) and key in data['result']:
                if isinstance(data['result'][key], list):
                    records = data['result'][key]
                    break

    if records is None:
        return {
            'status': 'ok',
            'format': 'non_tabular',
            'url': url,
            'raw_keys': list(data.keys()) if isinstance(data, dict) else f'list[{len(data)}]',
            'preview': str(data)[:500]
        }

    df = pd.DataFrame(records)
    return {
        'status': 'ok',
        'format': 'tabular',
        'url': url,
        'rows': len(df),
        'columns': list(df.columns),
        'preview': df.head(5).to_dict(orient='records'),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def scrape_api_and_load(url, table_name, params=None, if_exists='replace'):
    """Fetch data from a JSON API and load into the database."""
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError("Only http/https URLs are supported")

    headers = {'User-Agent': USER_AGENT, 'Accept': 'application/json'}
    resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # Find records
    records = None
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ['records', 'data', 'results', 'items', 'result', 'value', 'rows']:
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
            if 'result' in data and isinstance(data['result'], dict) and key in data['result']:
                if isinstance(data['result'][key], list):
                    records = data['result'][key]
                    break

    if records is None:
        raise ValueError("Could not find tabular data in API response")

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("API returned no records")

    conn = get_connection()
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        execute_db(
            "INSERT INTO pipeline_runs (pipeline_name, step_name, status, started_at, completed_at, rows_processed) VALUES (?,?,?,?,?,?)",
            ('api_scraping', f'api_{table_name}', 'completed',
             datetime.now().isoformat(), datetime.now().isoformat(), len(df))
        )
        return {
            'status': 'ok',
            'rows_loaded': len(df),
            'columns': list(df.columns),
            'table_name': table_name,
            'source_url': url
        }
    finally:
        conn.close()
