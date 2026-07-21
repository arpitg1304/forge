"""Generate an interactive HTML page for browsing the dataset registry."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path

from forge.registry.models import DatasetEntry

TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Forge Dataset Registry</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg: #080b12;
    --bg-subtle: #0c1018;
    --surface: #111827;
    --surface-hover: #162032;
    --surface-raised: #1a2540;
    --border: #1e2d4a;
    --border-hover: #2d4a7a;
    --text: #f0f4fc;
    --text-secondary: #94a3c0;
    --text-muted: #5e6e8a;
    --accent: #6c9fff;
    --accent-bright: #89b4ff;
    --accent-glow: rgba(108, 159, 255, 0.12);
    --accent-glow-strong: rgba(108, 159, 255, 0.25);
    --green: #34d399;
    --green-dim: rgba(52, 211, 153, 0.12);
    --orange: #fbbf24;
    --orange-dim: rgba(251, 191, 36, 0.12);
    --purple: #a78bfa;
    --purple-dim: rgba(167, 139, 250, 0.12);
    --rose: #fb7185;
    --rose-dim: rgba(251, 113, 133, 0.10);
    --teal: #2dd4bf;
    --teal-dim: rgba(45, 212, 191, 0.12);
    --radius: 12px;
    --radius-sm: 8px;
    --radius-xs: 6px;
    --shadow: 0 2px 8px rgba(0,0,0,0.3), 0 1px 3px rgba(0,0,0,0.2);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.4), 0 2px 8px rgba(0,0,0,0.3);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    min-height: 100vh;
  }

  .page-wrapper {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem 2.5rem;
  }

  /* ---- Header ---- */
  .header {
    text-align: center;
    padding: 3rem 1rem 2.5rem;
    position: relative;
  }
  .header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 10%;
    right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
  }
  .logo {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
  }
  .header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #fff 0%, var(--accent-bright) 50%, var(--purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .header .subtitle {
    color: var(--text-secondary);
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 400;
  }
  .facts-link {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 1.1rem;
    font-size: 0.9rem;
    color: var(--accent-bright);
    text-decoration: none;
    border: 1px solid var(--border-hover);
    padding: 0.4rem 0.85rem;
    border-radius: var(--radius-sm);
    transition: background 0.15s ease, border-color 0.15s ease;
  }
  .facts-link:hover { background: var(--accent-glow); border-color: var(--accent); }

  /* Stats */
  .stats-bar {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 2rem;
    flex-wrap: wrap;
  }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1.5rem;
    text-align: center;
    min-width: 120px;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .stat-card:hover {
    border-color: var(--border-hover);
    box-shadow: 0 0 20px var(--accent-glow);
  }
  .stat-card .num {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--accent-bright);
    line-height: 1.2;
  }
  .stat-card .label {
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-top: 0.15rem;
  }

  /* ---- Filters ---- */
  .filters {
    display: flex;
    gap: 0.6rem;
    margin: 2rem 0 1.5rem;
    flex-wrap: wrap;
    align-items: center;
    position: sticky;
    top: 0;
    z-index: 100;
    background: var(--bg);
    padding: 0.75rem 0;
    border-bottom: 1px solid transparent;
    transition: border-color 0.2s;
  }
  .filters.scrolled {
    border-bottom-color: var(--border);
    background: rgba(8, 11, 18, 0.95);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
  }
  .search-wrap {
    flex: 1;
    min-width: 220px;
    position: relative;
  }
  .search-icon {
    position: absolute;
    left: 0.85rem;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-muted);
    font-size: 0.9rem;
    pointer-events: none;
  }
  .filters input[type="text"] {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.6rem 0.85rem 0.6rem 2.4rem;
    border-radius: var(--radius-sm);
    font-size: 0.875rem;
    font-family: inherit;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .filters input[type="text"]:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }
  .filters input[type="text"]::placeholder { color: var(--text-muted); }
  .filter-select {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    padding: 0.6rem 2rem 0.6rem 0.75rem;
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    font-family: inherit;
    outline: none;
    cursor: pointer;
    appearance: none;
    -webkit-appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%235e6e8a' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10l-5 5z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 0.6rem center;
    transition: border-color 0.2s;
    min-width: 140px;
  }
  .filter-select:focus { border-color: var(--accent); }
  .filter-select:hover { border-color: var(--border-hover); }
  .result-count {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-left: auto;
    white-space: nowrap;
  }

  /* ---- Grid ---- */
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
    gap: 0.85rem;
  }

  /* ---- Card ---- */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.15s;
    position: relative;
  }
  .card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--purple));
    opacity: 0;
    transition: opacity 0.2s;
  }
  .card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow);
  }
  .card:hover::before { opacity: 1; }
  .card.open { box-shadow: var(--shadow-lg); }
  .card.open::before { opacity: 1; }

  .card-header {
    padding: 1.1rem 1.25rem 0.6rem;
    cursor: pointer;
    user-select: none;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.75rem;
  }
  .card-header:hover { background: var(--surface-hover); }
  .card-title-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  .card-title {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.01em;
  }
  .card-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    background: rgba(94, 110, 138, 0.12);
    padding: 0.1rem 0.45rem;
    border-radius: 4px;
    font-weight: 500;
  }
  .card-desc {
    color: var(--text-secondary);
    font-size: 0.825rem;
    margin-top: 0.35rem;
    line-height: 1.45;
  }
  .card-badges {
    display: flex;
    gap: 0.35rem;
    flex-wrap: wrap;
    margin-top: 0.6rem;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.2rem;
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    border: 1px solid transparent;
  }
  .badge-format { color: var(--green); background: var(--green-dim); border-color: rgba(52,211,153,0.2); }
  .badge-embodiment { color: var(--purple); background: var(--purple-dim); border-color: rgba(167,139,250,0.2); }
  .badge-demo { color: var(--orange); background: var(--orange-dim); border-color: rgba(251,191,36,0.2); }
  .badge-tag { color: var(--text-secondary); background: rgba(94,110,138,0.1); border-color: rgba(94,110,138,0.15); }
  .badge-scale { color: var(--teal); background: var(--teal-dim); border-color: rgba(45,212,191,0.2); }

  .chevron {
    color: var(--text-muted);
    transition: transform 0.25s ease;
    flex-shrink: 0;
    margin-top: 0.2rem;
    width: 20px;
    height: 20px;
  }
  .card.open .chevron { transform: rotate(180deg); }

  .card-meta {
    display: flex;
    gap: 0.5rem;
    padding: 0 1.25rem 0.85rem;
    font-size: 0.78rem;
    color: var(--text-muted);
    flex-wrap: wrap;
  }
  .meta-item {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(94, 110, 138, 0.06);
    padding: 0.2rem 0.6rem;
    border-radius: var(--radius-xs);
  }
  .meta-item strong { color: var(--text-secondary); font-weight: 600; }
  .meta-dot { color: var(--text-muted); opacity: 0.3; }

  /* ---- Accordion body ---- */
  .card-body {
    display: grid;
    grid-template-rows: 0fr;
    transition: grid-template-rows 0.3s ease;
  }
  .card.open .card-body {
    grid-template-rows: 1fr;
  }
  .card-body-inner {
    overflow: hidden;
    border-top: 1px solid transparent;
    transition: border-color 0.3s;
  }
  .card.open .card-body-inner {
    border-top-color: var(--border);
  }
  .card-body-content {
    padding: 1.15rem 1.25rem;
    display: grid;
    gap: 1rem;
  }

  /* ---- Detail sections ---- */
  .detail-section {}
  .detail-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
    font-weight: 700;
  }

  .source-list { display: flex; flex-direction: column; gap: 0.35rem; }
  .source-item {
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius-xs);
    padding: 0.55rem 0.75rem;
    font-size: 0.825rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    transition: border-color 0.15s;
  }
  .source-item:hover { border-color: var(--border-hover); }
  .source-item a {
    color: var(--accent);
    text-decoration: none;
    font-weight: 500;
  }
  .source-item a:hover { color: var(--accent-bright); text-decoration: underline; }
  .source-type {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-muted);
    background: rgba(94,110,138,0.15);
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  .source-note {
    color: var(--text-muted);
    font-size: 0.75rem;
    font-style: italic;
  }

  /* ---- Commands ---- */
  .cmd-grid { display: flex; flex-direction: column; gap: 0.3rem; }
  .cmd-row {
    display: flex;
    align-items: center;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius-xs);
    overflow: hidden;
    transition: border-color 0.15s;
  }
  .cmd-row:hover { border-color: var(--border-hover); }
  .cmd-row code {
    flex: 1;
    padding: 0.5rem 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--green);
    font-weight: 500;
    white-space: nowrap;
    overflow-x: auto;
  }
  .copy-btn {
    background: none;
    border: none;
    border-left: 1px solid var(--border);
    color: var(--text-muted);
    cursor: pointer;
    padding: 0.5rem 0.75rem;
    font-size: 0.75rem;
    font-family: inherit;
    font-weight: 500;
    transition: all 0.15s;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }
  .copy-btn:hover { color: var(--accent); background: var(--accent-glow); }
  .copy-btn.copied { color: var(--green); background: var(--green-dim); }

  /* ---- Paper link ---- */
  .paper-link {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--accent);
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.35rem 0.65rem;
    background: var(--accent-glow);
    border-radius: var(--radius-xs);
    border: 1px solid rgba(108, 159, 255, 0.15);
    transition: all 0.15s;
  }
  .paper-link:hover {
    background: var(--accent-glow-strong);
    border-color: rgba(108, 159, 255, 0.3);
    color: var(--accent-bright);
  }

  .no-results {
    text-align: center;
    color: var(--text-muted);
    padding: 4rem 1rem;
    font-size: 1.1rem;
  }
  .no-results .big { font-size: 2.5rem; margin-bottom: 0.5rem; }

  .footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.75rem;
    padding: 3rem 0 1rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
  }
  .footer a { color: var(--accent); text-decoration: none; }
  .footer a:hover { text-decoration: underline; }

  /* SVG icons inline */
  .icon { display: inline-flex; vertical-align: middle; }

  @media (max-width: 540px) {
    .page-wrapper { padding: 1rem; }
    .grid { grid-template-columns: 1fr; }
    .header h1 { font-size: 1.75rem; }
    .stat-card { min-width: 90px; padding: 0.65rem 1rem; }
    .stat-card .num { font-size: 1.35rem; }
  }
</style>
</head>
<body>

<div class="page-wrapper">

<div class="header">
  <div class="logo">Forge</div>
  <h1>Dataset Registry</h1>
  <p class="subtitle">Curated robotics datasets for training, evaluation, and research</p>
  <a class="facts-link" href="labels.html">⚒ Dataset facts labels &rarr;</a>
  <div class="stats-bar">
    <div class="stat-card">
      <div class="num">{{ total }}</div>
      <div class="label">Datasets</div>
    </div>
    <div class="stat-card">
      <div class="num">{{ demo_count }}</div>
      <div class="label">Demo-Ready</div>
    </div>
    <div class="stat-card">
      <div class="num">{{ total_episodes }}</div>
      <div class="label">Total Episodes</div>
    </div>
    <div class="stat-card">
      <div class="num">{{ formats | length }}</div>
      <div class="label">Formats</div>
    </div>
    <div class="stat-card">
      <div class="num">{{ embodiment_count }}</div>
      <div class="label">Robots</div>
    </div>
  </div>
</div>

<div class="filters" id="filterBar">
  <div class="search-wrap">
    <span class="search-icon">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
    </span>
    <input type="text" id="search" placeholder="Search by name, robot, task, tag..." oninput="filterCards()">
  </div>
  <select class="filter-select" id="formatFilter" onchange="filterCards()">
    <option value="">All Formats</option>
    {% for f in formats %}<option value="{{ f }}">{{ f }}</option>
    {% endfor %}
  </select>
  <select class="filter-select" id="tagFilter" onchange="filterCards()">
    <option value="">All Tags</option>
    {% for t in all_tags %}<option value="{{ t }}">{{ t.replace('_', ' ') }}</option>
    {% endfor %}
  </select>
  <select class="filter-select" id="embodimentFilter" onchange="filterCards()">
    <option value="">All Robots</option>
    {% for emb in all_embodiments %}<option value="{{ emb }}">{{ emb }}</option>
    {% endfor %}
  </select>
  <select class="filter-select" id="demoFilter" onchange="filterCards()">
    <option value="">All Datasets</option>
    <option value="demo">Demo-Suitable Only</option>
  </select>
  <span class="result-count" id="resultCount">{{ total }} datasets</span>
</div>

<div class="grid" id="grid">
{% for entry in entries %}
<div class="card" data-id="{{ entry.id }}" data-format="{{ entry.format }}" data-tags="{{ entry.tags | join(',') }}" data-demo="{{ 'true' if entry.demo_suitable else 'false' }}" data-embodiment="{{ entry.embodiment | join(',') }}" data-search="{{ entry.id }} {{ entry.name }} {{ entry.description }} {{ entry.embodiment | join(' ') }} {{ entry.tags | join(' ') }} {{ entry.task_types | join(' ') }} {{ entry.format }}">
  <div class="card-header" onclick="toggleCard(this)">
    <div style="flex:1;min-width:0;">
      <div class="card-title-row">
        <span class="card-title">{{ entry.name }}</span>
        <span class="card-id">{{ entry.id }}</span>
      </div>
      <div class="card-desc">{{ entry.description }}</div>
      <div class="card-badges">
        <span class="badge badge-format">{{ entry.format }}</span>
        {% for emb in entry.embodiment %}<span class="badge badge-embodiment">{{ emb }}</span>{% endfor %}
        {% if entry.demo_suitable %}<span class="badge badge-demo">demo</span>{% endif %}
        {% if entry.scale and entry.scale.episodes is not none %}
          {% if entry.scale.episodes >= 50000 %}<span class="badge badge-scale">large</span>{% endif %}
        {% endif %}
      </div>
    </div>
    <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
  </div>

  <div class="card-meta">
    {% if entry.scale and entry.scale.episodes is not none %}
    <span class="meta-item">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
      <strong>{{ "{:,}".format(entry.scale.episodes) }}</strong> episodes
    </span>
    {% endif %}
    {% if entry.scale and entry.scale.hours is not none %}
    <span class="meta-item">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      <strong>{{ entry.scale.hours | int }}</strong>h
    </span>
    {% endif %}
    {% if entry.license %}
    <span class="meta-item">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
      <strong>{{ entry.license }}</strong>
    </span>
    {% endif %}
  </div>

  <div class="card-body">
    <div class="card-body-inner">
      <div class="card-body-content">

        {% if entry.paper_url %}
        <div class="detail-section">
          <div class="detail-label">Paper</div>
          <a href="{{ entry.paper_url }}" target="_blank" rel="noopener" class="paper-link">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
            {{ entry.paper_url | replace('https://arxiv.org/abs/', 'arXiv:') }}
          </a>
        </div>
        {% endif %}

        {% if entry.tags %}
        <div class="detail-section">
          <div class="detail-label">Tags</div>
          <div class="card-badges">
            {% for tag in entry.tags %}<span class="badge badge-tag">{{ tag.replace('_', ' ') }}</span>{% endfor %}
          </div>
        </div>
        {% endif %}

        {% if entry.task_types %}
        <div class="detail-section">
          <div class="detail-label">Task Types</div>
          <div class="card-badges">
            {% for t in entry.task_types %}<span class="badge badge-tag">{{ t.replace('_', ' ') }}</span>{% endfor %}
          </div>
        </div>
        {% endif %}

        <div class="detail-section">
          <div class="detail-label">Sources</div>
          <div class="source-list">
            {% for source in entry.sources %}
            <div class="source-item">
              <span class="source-type">{{ source.type }}</span>
              {% if source.type == 'hf_hub' %}
              <a href="https://huggingface.co/datasets/{{ source.uri }}" target="_blank" rel="noopener">{{ source.uri }}</a>
              {% elif source.type == 'http' %}
              <a href="{{ source.uri }}" target="_blank" rel="noopener">{{ source.uri }}</a>
              {% else %}
              <span style="color:var(--text-secondary)">{{ source.uri }}</span>
              {% endif %}
              {% if source.notes %}<span class="source-note">{{ source.notes }}</span>{% endif %}
              {% if loop.index0 == entry.demo_source_index %}<span class="badge badge-demo" style="font-size:0.6rem">demo subset</span>{% endif %}
            </div>
            {% endfor %}
          </div>
        </div>

        <div class="detail-section">
          <div class="detail-label">Forge Commands</div>
          <div class="cmd-grid">
            <div class="cmd-row">
              <code>forge inspect {{ entry.id }}</code>
              <button class="copy-btn" onclick="copyCmd(event, this, 'forge inspect {{ entry.id }}')">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                Copy
              </button>
            </div>
            <div class="cmd-row">
              <code>forge quality {{ entry.id }}</code>
              <button class="copy-btn" onclick="copyCmd(event, this, 'forge quality {{ entry.id }}')">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                Copy
              </button>
            </div>
            <div class="cmd-row">
              <code>forge convert {{ entry.id }} ./output --format lerobot-v3</code>
              <button class="copy-btn" onclick="copyCmd(event, this, 'forge convert {{ entry.id }} ./output --format lerobot-v3')">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                Copy
              </button>
            </div>
            {% if entry.demo_suitable %}
            <div class="cmd-row">
              <code>forge inspect {{ entry.id }} --demo</code>
              <button class="copy-btn" onclick="copyCmd(event, this, 'forge inspect {{ entry.id }} --demo')">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                Copy
              </button>
            </div>
            {% endif %}
          </div>
        </div>

        {% if entry.notes %}
        <div class="detail-section">
          <div class="detail-label">Notes</div>
          <div style="color: var(--text-muted); font-size: 0.85rem; line-height: 1.5;">{{ entry.notes }}</div>
        </div>
        {% endif %}

      </div>
    </div>
  </div>
</div>
{% endfor %}
</div>

<div class="no-results" id="noResults" style="display:none">
  <div class="big">No matches</div>
  <div>Try adjusting your search or filters.</div>
</div>

<div class="footer">
  Generated by <a href="https://github.com/arpitg1304/forge">Forge</a> &mdash; the normalization layer for robotics data
</div>

</div>

<script>
function toggleCard(header) {
  var card = header.closest('.card');
  card.classList.toggle('open');
}

function copyCmd(e, btn, text) {
  e.stopPropagation();
  navigator.clipboard.writeText(text).then(function() {
    var orig = btn.innerHTML;
    btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
    btn.classList.add('copied');
    setTimeout(function() {
      btn.innerHTML = orig;
      btn.classList.remove('copied');
    }, 1500);
  });
}

function filterCards() {
  var query = document.getElementById('search').value.toLowerCase();
  var fmt = document.getElementById('formatFilter').value;
  var tag = document.getElementById('tagFilter').value;
  var emb = document.getElementById('embodimentFilter').value;
  var demo = document.getElementById('demoFilter').value;
  var cards = document.querySelectorAll('.card');
  var visible = 0;

  cards.forEach(function(card) {
    var show = true;
    if (query && card.dataset.search.toLowerCase().indexOf(query) === -1) show = false;
    if (fmt && card.dataset.format !== fmt) show = false;
    if (tag && card.dataset.tags.indexOf(tag) === -1) show = false;
    if (emb && card.dataset.embodiment.indexOf(emb) === -1) show = false;
    if (demo === 'demo' && card.dataset.demo !== 'true') show = false;
    card.style.display = show ? '' : 'none';
    if (show) visible++;
  });

  document.getElementById('noResults').style.display = visible === 0 ? '' : 'none';
  document.getElementById('resultCount').textContent = visible + ' dataset' + (visible !== 1 ? 's' : '');
}

// Sticky filter bar shadow on scroll
var filterBar = document.getElementById('filterBar');
window.addEventListener('scroll', function() {
  filterBar.classList.toggle('scrolled', window.scrollY > 200);
});

// Keyboard shortcut: / to focus search
document.addEventListener('keydown', function(e) {
  if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
    e.preventDefault();
    document.getElementById('search').focus();
  }
  if (e.key === 'Escape') {
    document.getElementById('search').blur();
  }
});
</script>

</body>
</html>
"""


def generate_registry_html(entries: list[DatasetEntry]) -> str:
    """Generate an HTML page for browsing the dataset registry.

    Args:
        entries: List of DatasetEntry objects to display.

    Returns:
        HTML string.

    Raises:
        MissingDependencyError: If jinja2 is not installed.
    """
    try:
        from jinja2 import Environment
    except ImportError:
        from forge.core.exceptions import MissingDependencyError

        raise MissingDependencyError(
            "jinja2", "registry HTML view", "pip install forge-robotics[registry]"
        )

    env = Environment(autoescape=True)
    template = env.from_string(TEMPLATE)

    # Compute stats
    formats = sorted({e.format for e in entries})
    all_tags = sorted({t for e in entries for t in e.tags})
    all_embodiments = {emb for e in entries for emb in e.embodiment}
    total_episodes = sum(
        e.scale.episodes for e in entries
        if e.scale and e.scale.episodes is not None
    )
    demo_count = sum(1 for e in entries if e.demo_suitable)

    if total_episodes >= 1_000_000:
        total_episodes_str = f"{total_episodes / 1_000_000:.1f}M"
    elif total_episodes >= 1_000:
        total_episodes_str = f"{total_episodes / 1_000:.0f}K"
    else:
        total_episodes_str = str(total_episodes)

    return template.render(
        entries=entries,
        total=len(entries),
        demo_count=demo_count,
        total_episodes=total_episodes_str,
        formats=formats,
        all_tags=all_tags,
        all_embodiments=sorted(all_embodiments),
        embodiment_count=len(all_embodiments),
    )


def open_registry_html(entries: list[DatasetEntry]) -> Path:
    """Generate and open the registry HTML page in the default browser.

    Args:
        entries: List of DatasetEntry objects to display.

    Returns:
        Path to the generated HTML file.
    """
    html = generate_registry_html(entries)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", prefix="forge_registry_", delete=False, mode="w"
    )
    tmp.write(html)
    tmp.close()
    path = Path(tmp.name)
    webbrowser.open(f"file://{path}")
    return path
