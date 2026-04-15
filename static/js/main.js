/* Singapore Data Science Lab - Main JavaScript */

// Tab functionality
function openTab(tabGroup, tabName) {
    const container = document.getElementById(tabGroup);
    if (!container) return;

    container.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    container.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

    const tab = document.getElementById(tabName);
    if (tab) tab.classList.add('active');

    event.target.classList.add('active');
}

// API call helper
async function apiCall(url, method = 'GET', body = null) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' }
    };
    if (body) opts.body = JSON.stringify(body);

    const res = await fetch(url, opts);
    return await res.json();
}

// SQL query executor
async function executeSQL() {
    const query = document.getElementById('sql-input').value.trim();
    const output = document.getElementById('sql-output');
    if (!query) {
        output.innerHTML = '<span style="color: var(--warning)">Please enter a SQL query.</span>';
        return;
    }

    output.innerHTML = '<div class="spinner"></div> Executing...';

    try {
        const result = await apiCall('/api/sql/execute', 'POST', { query });

        if (result.status === 'error') {
            output.innerHTML = `<span style="color: var(--danger)">Error: ${result.message}</span>`;
            return;
        }

        if (result.data && result.data.length > 0) {
            let html = `<div style="color: var(--info); margin-bottom: 8px;">${result.row_count} row(s) returned${result.truncated ? ' (truncated to 500)' : ''}</div>`;
            html += '<div style="overflow-x: auto;"><table class="table table-striped">';
            html += '<tr>' + Object.keys(result.data[0]).map(k => `<th>${k}</th>`).join('') + '</tr>';
            result.data.forEach(row => {
                html += '<tr>' + Object.values(row).map(v => `<td>${v !== null ? v : 'NULL'}</td>`).join('') + '</tr>';
            });
            html += '</table></div>';
            output.innerHTML = html;
        } else {
            output.innerHTML = '<span style="color: var(--text-secondary)">Query returned no results.</span>';
        }
    } catch (err) {
        output.innerHTML = `<span style="color: var(--danger)">Error: ${err.message}</span>`;
    }
}

// Train model
async function trainModel(modelType) {
    const output = document.getElementById('model-output');
    if (output) output.innerHTML = '<div class="spinner"></div> Training model...';

    try {
        const result = await apiCall('/api/model/train', 'POST', { model_type: modelType });

        if (output) {
            if (result.status === 'ok') {
                output.innerHTML = '<pre>' + JSON.stringify(result.result, null, 2) + '</pre>';
            } else {
                output.innerHTML = `<span style="color: var(--danger)">Error: ${result.message}</span>`;
            }
        }
    } catch (err) {
        if (output) output.innerHTML = `<span style="color: var(--danger)">Error: ${err.message}</span>`;
    }
}

// Predict
async function predictPrice() {
    const output = document.getElementById('predict-output');
    if (output) output.innerHTML = '<div class="spinner"></div> Predicting...';

    const data = {
        model_type: 'hdb_price',
        floor_area: parseFloat(document.getElementById('pred-area').value) || 93,
        storey: parseFloat(document.getElementById('pred-storey').value) || 7,
        town_code: parseInt(document.getElementById('pred-town').value) || 0,
        flat_type_code: parseInt(document.getElementById('pred-flat').value) || 1,
        building_age: parseInt(document.getElementById('pred-age').value) || 20
    };

    try {
        const result = await apiCall('/api/model/predict', 'POST', data);
        if (output) {
            if (result.status === 'ok') {
                const price = result.result.predicted_price;
                output.innerHTML = `<div style="font-size: 1.5rem; color: var(--success); font-weight: bold;">Predicted Price: $${price.toLocaleString()}</div>
                    <pre style="margin-top: 12px;">${JSON.stringify(result.result.input, null, 2)}</pre>`;
            } else {
                output.innerHTML = `<span style="color: var(--danger)">Error: ${result.message}</span>`;
            }
        }
    } catch (err) {
        if (output) output.innerHTML = `<span style="color: var(--danger)">Error: ${err.message}</span>`;
    }
}

// Stream events
async function generateEvents() {
    const output = document.getElementById('stream-output');
    if (output) output.innerHTML = '<div class="spinner"></div> Generating events...';

    try {
        const result = await apiCall('/api/stream/generate', 'POST', { batch_size: 15 });
        if (output) {
            output.innerHTML = `<span style="color: var(--success)">Generated ${result.events_generated} events.</span>`;
        }
        refreshStreamStats();
    } catch (err) {
        if (output) output.innerHTML = `<span style="color: var(--danger)">Error: ${err.message}</span>`;
    }
}

async function processEvents() {
    const output = document.getElementById('stream-output');
    if (output) output.innerHTML = '<div class="spinner"></div> Processing events...';

    try {
        const result = await apiCall('/api/stream/process', 'POST', { window_minutes: 60 });
        if (output) {
            output.innerHTML = '<pre>' + JSON.stringify(result.result, null, 2) + '</pre>';
        }
    } catch (err) {
        if (output) output.innerHTML = `<span style="color: var(--danger)">Error: ${err.message}</span>`;
    }
}

async function refreshStreamStats() {
    try {
        const result = await apiCall('/api/stream/stats');
        const statsEl = document.getElementById('stream-stats');
        if (statsEl && result.status === 'ok') {
            statsEl.innerHTML = `Total: ${result.stats.total_events} | Processed: ${result.stats.processed_events}`;
        }
    } catch (err) {
        // Silent fail for stats refresh
    }
}

// Classify text
async function classifyText() {
    const text = document.getElementById('nlp-input').value.trim();
    const output = document.getElementById('nlp-output');

    if (!text) {
        output.innerHTML = '<span style="color: var(--warning)">Please enter text to classify.</span>';
        return;
    }

    output.innerHTML = '<div class="spinner"></div> Classifying...';

    try {
        const result = await apiCall('/api/model/predict', 'POST', { model_type: 'text_classify', text });
        if (result.status === 'ok') {
            output.innerHTML = '<pre>' + JSON.stringify(result.result, null, 2) + '</pre>';
        } else {
            output.innerHTML = `<span style="color: var(--danger)">Error: ${result.message}</span>`;
        }
    } catch (err) {
        output.innerHTML = `<span style="color: var(--danger)">Error: ${err.message}</span>`;
    }
}

// Run pipeline
async function runPipeline() {
    const output = document.getElementById('pipeline-output');
    if (output) output.innerHTML = '<div class="spinner"></div> Running pipeline... this may take a moment.';

    try {
        const result = await apiCall('/api/pipeline/run', 'POST');
        if (output) {
            if (result.status === 'ok') {
                let html = '<div style="color: var(--success); margin-bottom: 12px;">Pipeline completed!</div>';
                for (const [step, info] of Object.entries(result.result)) {
                    const color = info.status === 'completed' ? 'var(--success)' : 'var(--danger)';
                    html += `<div class="pipeline-step ${info.status}">
                        <span class="step-name">${step}</span>
                        <span class="badge badge-${info.status === 'completed' ? 'success' : 'danger'}">${info.status}</span>
                    </div>`;
                }
                output.innerHTML = html;
            } else {
                output.innerHTML = `<span style="color: var(--danger)">Error: ${result.message}</span>`;
            }
        }
    } catch (err) {
        if (output) output.innerHTML = `<span style="color: var(--danger)">Error: ${err.message}</span>`;
    }
}

// Submit feedback
async function submitFeedback() {
    const module = document.getElementById('fb-module').value;
    const rating = parseInt(document.getElementById('fb-rating').value);
    const comment = document.getElementById('fb-comment').value;
    const output = document.getElementById('feedback-output');

    try {
        const result = await apiCall('/api/feedback', 'POST', {
            module_name: module, rating, comment
        });
        if (output) {
            output.innerHTML = `<span style="color: var(--success)">${result.message || 'Feedback submitted!'}</span>`;
        }
    } catch (err) {
        if (output) output.innerHTML = `<span style="color: var(--danger)">Error: ${err.message}</span>`;
    }
}

// Sidebar toggle for mobile
function toggleSidebar() {
    document.querySelector('.sidebar').classList.toggle('open');
}

// Load sample SQL
function loadSampleSQL(query) {
    const input = document.getElementById('sql-input');
    if (input) {
        input.value = query;
        executeSQL();
    }
}

// Initialize Plotly charts
function renderPlotly(elementId, chartJSON) {
    const el = document.getElementById(elementId);
    if (!el || !chartJSON) return;

    try {
        const data = JSON.parse(chartJSON);
        Plotly.newPlot(el, data.data, {
            ...data.layout,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(26,26,46,0.5)',
            font: { color: '#e0e0e0' },
            margin: { t: 40, r: 20, b: 60, l: 60 }
        }, { responsive: true });
    } catch (err) {
        console.error('Chart render error:', err);
    }
}

// DOM ready
document.addEventListener('DOMContentLoaded', () => {
    // Activate current nav link
    const path = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === path) {
            link.classList.add('active');
        }
    });

    // Initialize any charts on the page
    document.querySelectorAll('[data-chart]').forEach(el => {
        const chartData = el.getAttribute('data-chart');
        if (chartData) renderPlotly(el.id, chartData);
    });
});
