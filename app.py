"""SmartDesk AI - Intelligent Help Desk Management System.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import textwrap
from datetime import datetime
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from database import db
from ml import predictor
from ml.train_model import train
from services.classification import CONFIDENCE_THRESHOLD, classify_complaint
from services.priority import detect_priority
from services.response import get_suggested_response
from services.validation import validate_complaint

st.set_page_config(
    page_title="SmartDesk AI",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

db.init_db()

NAV_PAGES = ("Dashboard", "Submit Complaint", "Manage Tickets", "About Project")

CHART_BLUE = "#2563EB"
CHART_RED = "#DC2626"
CHART_AMBER = "#D97706"
CHART_GREEN = "#059669"
CHART_STATUS_AMBER = "#F59E0B"
CHART_STATUS_BLUE = "#3B82F6"
CHART_STATUS_GREEN = "#10B981"

ICONS: dict[str, str] = {
    "ticket": (
        '<path d="M2 9a3 3 0 0 1 0 6v2a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-2a3 3 0 0 1 0-6V7a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2Z"/>'
        '<path d="M13 5v2"/><path d="M13 17v2"/><path d="M13 11v2"/>'
    ),
    "inbox": (
        '<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/>'
        '<path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>'
    ),
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "progress": '<path d="M21 12a9 9 0 1 1-6.219-8.56"/>',
    "check-circle": (
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        '<polyline points="22 4 12 14.01 9 11.01"/>'
    ),
    "layers": (
        '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
        '<polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>'
    ),
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "cpu": (
        '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
        '<path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/>'
        '<path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>'
    ),
    "route": (
        '<circle cx="6" cy="19" r="3"/>'
        '<path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/>'
        '<circle cx="18" cy="5" r="3"/>'
    ),
    "message": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "edit": '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    "shield": (
        '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
        '<path d="m9 12 2 2 4-4"/>'
    ),
    "database": (
        '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/>'
        '<path d="M3 12A9 3 0 0 0 21 12"/>'
    ),
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "alert": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
        '<path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    "x-circle": (
        '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>'
    ),
    "calendar": (
        '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>'
        '<line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/>'
        '<line x1="3" y1="10" x2="21" y2="10"/>'
    ),
    "chart": (
        '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>'
    ),
    "user": (
        '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/>'
        '<circle cx="12" cy="7" r="4"/>'
    ),
    "mail": (
        '<rect x="2" y="4" width="20" height="16" rx="2"/>'
        '<path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>'
    ),
    "tag": (
        '<path d="M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 .586 1.414l8.704 8.704a2.426 2.426 0 0 0 3.42 0l6.58-6.58a2.426 2.426 0 0 0 0-3.42z"/>'
        '<circle cx="7.5" cy="7.5" r=".5" fill="currentColor"/>'
    ),
    "list": (
        '<path d="m3 17 2 2 4-4"/><path d="m3 7 2 2 4-4"/>'
        '<path d="M13 6h8"/><path d="M13 12h8"/><path d="M13 18h8"/>'
    ),
    "refresh": (
        '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>'
        '<path d="M21 3v5h-5"/>'
        '<path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>'
        '<path d="M8 16H3v5"/>'
    ),
    "target": (
        '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/>'
        '<circle cx="12" cy="12" r="2"/>'
    ),
    "book": (
        '<path d="M12 7v14"/>'
        '<path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/>'
    ),
    "file": (
        '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>'
        '<polyline points="14 2 14 8 20 8"/>'
        '<line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>'
        '<line x1="10" y1="9" x2="8" y2="9"/>'
    ),
}

CSS = """
:root {
    --sd-bg: #F4F6FB;
    --sd-surface: #FFFFFF;
    --sd-surface-soft: #F8FAFC;
    --sd-surface-muted: #F1F5F9;
    --sd-border: #E2E8F0;
    --sd-border-strong: #CBD5E1;
    --sd-text: #0F172A;
    --sd-text-secondary: #334155;
    --sd-text-muted: #64748B;
    --sd-text-faint: #94A3B8;
    --sd-primary: #2563EB;
    --sd-primary-strong: #1D4ED8;
    --sd-primary-soft: #EFF6FF;
    --sd-primary-border: #BFDBFE;
    --sd-success: #059669;
    --sd-success-soft: #ECFDF5;
    --sd-success-border: #A7F3D0;
    --sd-warning: #D97706;
    --sd-warning-soft: #FFFBEB;
    --sd-warning-border: #FDE68A;
    --sd-danger: #DC2626;
    --sd-danger-soft: #FEF2F2;
    --sd-danger-border: #FECACA;
    --sd-info: #0284C7;
    --sd-info-soft: #F0F9FF;
    --sd-info-border: #BAE6FD;
    --sd-violet: #7C3AED;
    --sd-violet-soft: #F5F3FF;
    --sd-violet-border: #DDD6FE;
    --sd-radius: 14px;
    --sd-radius-sm: 10px;
    --sd-space: 8px;
    --sd-shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 14px 34px -24px rgba(15, 23, 42, .22);
    --sd-font: "Segoe UI", "Inter", -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
}

html, body, [class*="css"] { font-family: var(--sd-font); }
.stApp, [data-testid="stAppViewContainer"] { background: var(--sd-bg); color: var(--sd-text); }
.block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 3.5rem; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stAppDeployButton"], .stDeployButton { display: none; }

h1, h2, h3, h4 { color: var(--sd-text); letter-spacing: -.02em; }
p, li { color: var(--sd-text-secondary); }
a { color: var(--sd-primary); }
hr { border: none; border-top: 1px solid var(--sd-border); margin: 1.5rem 0; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1220 0%, #0F1B33 55%, #111F3D 100%);
    border-right: 1px solid rgba(148, 163, 184, .14);
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: .4rem 1.15rem 1.5rem 1.15rem;
}
section[data-testid="stSidebar"] hr {
    border-top-color: rgba(148, 163, 184, .16);
    margin: 1rem 0;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #94A3B8;
    font-size: .72rem;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label,
section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
    display: none;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: .35rem;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] {
    width: 100%;
    margin: 0;
    padding: .62rem .75rem;
    border: 1px solid transparent;
    border-radius: 10px;
    background: transparent;
    transition: background .15s ease, border-color .15s ease, transform .15s ease;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    background: rgba(148, 163, 184, .10);
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] p {
    color: #CBD5E1;
    font-size: .92rem;
    font-weight: 550;
    line-height: 1.2;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
    background: rgba(37, 99, 235, .20);
    border-color: rgba(96, 165, 250, .32);
    box-shadow: inset 3px 0 0 #60A5FA;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
    color: #FFFFFF;
    font-weight: 650;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {
    color: #CBD5E1;
}

.brand { display: flex; align-items: center; gap: .75rem; padding: .15rem 0 .1rem; }
.brand-mark {
    width: 44px; height: 44px; border-radius: 13px; display: grid; place-items: center;
    color: #fff; background: linear-gradient(135deg, #3B82F6 0%, #2563EB 55%, #1D4ED8 100%);
    box-shadow: 0 8px 18px -8px rgba(37, 99, 235, .7); flex-shrink: 0;
}
.brand-name { color: #F8FAFC; font-size: 1.02rem; font-weight: 750; letter-spacing: -.02em; line-height: 1.1; }
.brand-sub {
    color: #94A3B8; font-size: .68rem; font-weight: 500; letter-spacing: .04em;
    text-transform: uppercase; margin-top: .18rem;
}
.nav-label {
    color: #64748B; font-size: .66rem; font-weight: 700; letter-spacing: .16em;
    text-transform: uppercase; margin: 1.35rem 0 .45rem .2rem;
}
.sidebar-divider { height: 1px; background: rgba(148, 163, 184, .16); margin: 1.1rem 0; }
.sidebar-status {
    border: 1px solid rgba(148, 163, 184, .18); background: rgba(255, 255, 255, .045);
    border-radius: 12px; padding: .8rem .85rem;
}
.sidebar-status-top {
    display: flex; align-items: center; gap: .45rem; color: #94A3B8;
    font-size: .68rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.status-dot--green { background: #34D399; box-shadow: 0 0 0 4px rgba(52, 211, 153, .12); }
.status-dot--amber { background: #FBBF24; box-shadow: 0 0 0 4px rgba(251, 191, 36, .12); }
.sidebar-status-value { color: #F1F5F9; font-size: .88rem; font-weight: 650; margin-top: .45rem; }
.sidebar-status-meta { color: #94A3B8; font-size: .72rem; margin-top: .15rem; line-height: 1.45; }
.sidebar-footer { color: #64748B; font-size: .68rem; line-height: 1.55; margin-top: 1rem; }
.sidebar-footer strong { color: #94A3B8; font-weight: 600; }

.page-hero {
    display: flex; align-items: flex-end; justify-content: space-between; gap: 1.5rem;
    padding: 1.45rem 1.5rem; margin-bottom: 1.25rem;
    background: radial-gradient(circle at 100% 0%, rgba(37, 99, 235, .09), transparent 42%),
        linear-gradient(180deg, #FFFFFF, #FBFCFE);
    border: 1px solid var(--sd-border); border-radius: 18px; box-shadow: var(--sd-shadow);
}
.page-hero h1 {
    font-size: 2rem; line-height: 1.15; font-weight: 750; letter-spacing: -.035em;
    color: var(--sd-text); margin: 0;
}
.page-hero p {
    margin: .45rem 0 0; color: var(--sd-text-muted); font-size: .94rem;
    max-width: 660px; line-height: 1.55;
}
.eyebrow {
    font-size: .68rem; text-transform: uppercase; letter-spacing: .14em;
    font-weight: 800; color: var(--sd-primary); margin-bottom: .35rem;
}
.hero-aside { display: flex; flex-direction: column; align-items: flex-end; gap: .45rem; flex-shrink: 0; }
.live-pill, .soft-pill {
    display: inline-flex; align-items: center; gap: .45rem; height: 28px; padding: 0 .7rem;
    border-radius: 999px; font-size: .72rem; font-weight: 700; white-space: nowrap;
}
.live-pill { background: var(--sd-success-soft); border: 1px solid var(--sd-success-border); color: #047857; }
.soft-pill { background: var(--sd-primary-soft); border: 1px solid var(--sd-primary-border); color: var(--sd-primary-strong); }
.live-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #10B981;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, .15); animation: sd-pulse 2s ease-in-out infinite;
}
@keyframes sd-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .45; } }
.hero-meta { font-size: .73rem; color: var(--sd-text-faint); }

.stat-card {
    --accent: var(--sd-primary);
    --accent-soft: var(--sd-primary-soft);
    position: relative; height: 100%; min-height: 138px; background: var(--sd-surface);
    border: 1px solid var(--sd-border); border-radius: var(--sd-radius);
    padding: 1.15rem 1.2rem 1.05rem; box-shadow: var(--sd-shadow); overflow: hidden;
    transition: transform .15s ease, box-shadow .15s ease;
}
.stat-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(15, 23, 42, .05), 0 18px 40px -26px rgba(15, 23, 42, .28);
}
.stat-card::after {
    content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
    background: var(--accent); opacity: .9;
}
.stat-card--blue { --accent: #2563EB; --accent-soft: #EFF6FF; }
.stat-card--amber { --accent: #D97706; --accent-soft: #FFFBEB; }
.stat-card--green { --accent: #059669; --accent-soft: #ECFDF5; }
.stat-card--violet { --accent: #7C3AED; --accent-soft: #F5F3FF; }
.stat-card--slate { --accent: #475569; --accent-soft: #F1F5F9; }
.stat-top { display: flex; align-items: center; gap: .7rem; }
.stat-icon {
    width: 38px; height: 38px; border-radius: 10px; display: grid; place-items: center;
    color: var(--accent); background: var(--accent-soft); flex-shrink: 0;
}
.stat-label {
    color: var(--sd-text-muted); font-size: .72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .07em; line-height: 1.25;
}
.stat-value {
    color: var(--sd-text); font-size: 1.95rem; font-weight: 750; line-height: 1.05;
    letter-spacing: -.035em; margin-top: .8rem;
}
.stat-foot { color: var(--sd-text-muted); font-size: .76rem; line-height: 1.35; margin-top: .35rem; }

div[class*="st-key-card_"] {
    background: var(--sd-surface) !important;
    border: 1px solid var(--sd-border) !important;
    border-radius: var(--sd-radius) !important;
    box-shadow: var(--sd-shadow);
    padding: 1.15rem 1.25rem !important;
}
div[class*="st-key-card_"] > div { gap: .65rem; }
.card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; margin-bottom: .35rem; }
.card-title { color: var(--sd-text); font-size: .98rem; font-weight: 700; letter-spacing: -.01em; line-height: 1.3; }
.card-subtitle { color: var(--sd-text-muted); font-size: .77rem; line-height: 1.4; margin-top: .18rem; }
.card-icon {
    width: 34px; height: 34px; border-radius: 9px; display: grid; place-items: center;
    color: var(--sd-text-muted); background: var(--sd-surface-muted); flex-shrink: 0;
}

.section-heading { margin: 1.6rem 0 .85rem; }
.section-heading h2 { margin: 0; font-size: 1.18rem; font-weight: 750; letter-spacing: -.02em; color: var(--sd-text); }
.section-heading p { margin: .2rem 0 0; color: var(--sd-text-muted); font-size: .82rem; }
.section-heading .eyebrow { margin-bottom: .3rem; }

.table-block {
    background: var(--sd-surface); border: 1px solid var(--sd-border); border-radius: var(--sd-radius);
    box-shadow: var(--sd-shadow); overflow: hidden; margin-top: 1rem;
}
.table-head {
    display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem;
    padding: 1.1rem 1.25rem; border-bottom: 1px solid var(--sd-border);
    background: linear-gradient(180deg, #FBFCFE, #F8FAFC);
}
.table-head .card-title { font-size: .98rem; }
.table-count {
    display: inline-flex; align-items: center; height: 26px; padding: 0 .65rem; border-radius: 999px;
    background: var(--sd-surface-muted); border: 1px solid var(--sd-border);
    color: var(--sd-text-muted); font-size: .72rem; font-weight: 650; white-space: nowrap;
}
.table-scroll { overflow: auto; max-height: 430px; }
.data-table { width: 100%; border-collapse: collapse; font-size: .875rem; }
.data-table thead th {
    position: sticky; top: 0; z-index: 1; background: #F8FAFC; color: var(--sd-text-muted);
    font-size: .68rem; text-transform: uppercase; letter-spacing: .075em; font-weight: 750;
    text-align: left; padding: .72rem 1rem; border-bottom: 1px solid var(--sd-border); white-space: nowrap;
}
.data-table tbody td {
    padding: .78rem 1rem; border-bottom: 1px solid #F1F5F9;
    color: var(--sd-text-secondary); vertical-align: middle;
}
.data-table tbody tr:last-child td { border-bottom: none; }
.data-table tbody tr:hover { background: #F8FAFC; }
.ticket-code {
    font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
    font-size: .76rem; color: var(--sd-text-secondary); font-weight: 650;
    letter-spacing: -.01em; white-space: nowrap;
}
.subject-cell {
    display: inline-block; max-width: 380px; overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; color: var(--sd-text); font-weight: 550; vertical-align: middle;
}
.muted-cell { color: var(--sd-text-muted); white-space: nowrap; font-size: .79rem; }

.badge {
    display: inline-flex; align-items: center; gap: .35rem; height: 24px; padding: 0 .6rem;
    border-radius: 999px; font-size: .71rem; font-weight: 650; border: 1px solid transparent;
    white-space: nowrap; line-height: 1;
}
.badge-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
.badge--blue { background: var(--sd-primary-soft); color: var(--sd-primary-strong); border-color: var(--sd-primary-border); }
.badge--green { background: var(--sd-success-soft); color: #047857; border-color: var(--sd-success-border); }
.badge--amber { background: var(--sd-warning-soft); color: #B45309; border-color: var(--sd-warning-border); }
.badge--red { background: var(--sd-danger-soft); color: #B91C1C; border-color: var(--sd-danger-border); }
.badge--violet { background: var(--sd-violet-soft); color: #6D28D9; border-color: var(--sd-violet-border); }
.badge--slate { background: var(--sd-surface-muted); color: var(--sd-text-secondary); border-color: var(--sd-border); }

.empty-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; padding: 2.2rem 1.25rem;
}
.empty-state--compact { padding: 1.6rem 1rem; }
.empty-icon {
    width: 48px; height: 48px; border-radius: 14px; display: grid; place-items: center;
    background: var(--sd-primary-soft); color: var(--sd-primary); margin-bottom: .8rem;
}
.empty-title { color: var(--sd-text); font-size: .98rem; font-weight: 700; }
.empty-text {
    color: var(--sd-text-muted); font-size: .82rem; line-height: 1.5;
    max-width: 360px; margin-top: .25rem;
}

.sd-alert {
    display: flex; align-items: flex-start; gap: .7rem; padding: .85rem 1rem;
    border-radius: 12px; border: 1px solid; margin: .5rem 0; font-size: .86rem; line-height: 1.5;
}
.sd-alert-icon { flex-shrink: 0; margin-top: .05rem; }
.sd-alert-title { font-weight: 700; margin-bottom: .1rem; }
.sd-alert-body { font-size: .84rem; line-height: 1.55; }
.sd-alert--success { background: var(--sd-success-soft); border-color: var(--sd-success-border); color: #065F46; }
.sd-alert--warning { background: var(--sd-warning-soft); border-color: var(--sd-warning-border); color: #92400E; }
.sd-alert--error { background: var(--sd-danger-soft); border-color: var(--sd-danger-border); color: #991B1B; }
.sd-alert--info { background: var(--sd-info-soft); border-color: var(--sd-info-border); color: #075985; }
.alert-list { margin: .3rem 0 0; padding-left: 1.1rem; }
.alert-list li { margin-bottom: .18rem; }

.result-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; }
.result-id { color: var(--sd-text); font-size: 1.5rem; font-weight: 750; letter-spacing: -.03em; margin-top: .2rem; }
.result-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin-top: 1rem; }
.result-item {
    background: var(--sd-surface-soft); border: 1px solid var(--sd-border);
    border-radius: 11px; padding: .75rem .85rem;
}
.result-label {
    color: var(--sd-text-muted); font-size: .66rem; font-weight: 750;
    letter-spacing: .075em; text-transform: uppercase; margin-bottom: .35rem;
}
.result-value { color: var(--sd-text); font-size: .88rem; font-weight: 650; }
.meter { height: 8px; border-radius: 999px; background: var(--sd-surface-muted); overflow: hidden; margin-top: 1rem; }
.meter-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #3B82F6, #2563EB); }
.meter--warning .meter-fill { background: linear-gradient(90deg, #FBBF24, #D97706); }
.result-note { color: var(--sd-text-muted); font-size: .78rem; line-height: 1.5; margin-top: .45rem; }

.response-box {
    display: flex; gap: .8rem; background: var(--sd-surface-soft); border: 1px solid var(--sd-border);
    border-left: 4px solid var(--sd-primary); border-radius: 10px; padding: .9rem 1rem;
    color: var(--sd-text-secondary); font-size: .87rem; line-height: 1.6;
}
.response-icon { color: var(--sd-primary); flex-shrink: 0; margin-top: .1rem; }

.detail-header {
    display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem;
    padding-bottom: .9rem; border-bottom: 1px solid var(--sd-border);
}
.detail-id {
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .74rem;
    font-weight: 650; color: var(--sd-text-muted); letter-spacing: .02em;
}
.detail-title {
    color: var(--sd-text); font-size: 1.18rem; font-weight: 700;
    letter-spacing: -.02em; line-height: 1.3; margin-top: .25rem;
}
.detail-badges { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .55rem; }
.meta-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin-top: 1rem; }
.meta-label {
    color: var(--sd-text-muted); font-size: .66rem; font-weight: 750;
    letter-spacing: .075em; text-transform: uppercase; margin-bottom: .25rem;
}
.meta-value { color: var(--sd-text); font-size: .86rem; font-weight: 550; overflow-wrap: anywhere; }
.detail-section { margin-top: 1rem; }
.detail-label {
    color: var(--sd-text-muted); font-size: .68rem; font-weight: 750;
    letter-spacing: .075em; text-transform: uppercase; margin-bottom: .4rem;
}
.inset {
    background: var(--sd-surface-soft); border: 1px solid var(--sd-border); border-radius: 10px;
    padding: .85rem 1rem; color: var(--sd-text-secondary); font-size: .87rem; line-height: 1.6;
}

.step-list { display: flex; flex-direction: column; gap: .8rem; }
.step { display: flex; gap: .7rem; align-items: flex-start; }
.step-num {
    width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center;
    background: var(--sd-primary-soft); color: var(--sd-primary-strong);
    font-size: .7rem; font-weight: 750; flex-shrink: 0; margin-top: .05rem;
}
.step-title { color: var(--sd-text); font-size: .84rem; font-weight: 650; }
.step-text { color: var(--sd-text-muted); font-size: .78rem; line-height: 1.5; margin-top: .1rem; }

.stack-list { display: flex; flex-direction: column; }
.stack-row {
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    padding: .5rem 0; border-bottom: 1px solid #F1F5F9; font-size: .82rem;
}
.stack-row:last-child { border-bottom: none; }
.stack-layer { color: var(--sd-text-muted); font-weight: 550; }
.stack-tech { color: var(--sd-text); font-weight: 650; text-align: right; }

.check-list { list-style: none; margin: .2rem 0 0; padding: 0; display: flex; flex-direction: column; gap: .5rem; }
.check-list li {
    display: flex; gap: .55rem; align-items: flex-start; color: var(--sd-text-secondary);
    font-size: .84rem; line-height: 1.5;
}
.check-icon { color: var(--sd-success); flex-shrink: 0; margin-top: .08rem; }

.chart-legend { display: flex; flex-wrap: wrap; justify-content: center; gap: .9rem; margin-top: .35rem; }
.legend-item {
    display: inline-flex; align-items: center; gap: .35rem; color: var(--sd-text-muted);
    font-size: .72rem; font-weight: 550;
}
.legend-dot { width: 7px; height: 7px; border-radius: 50%; }
.legend-dot--amber { background: #F59E0B; }
.legend-dot--blue { background: #3B82F6; }
.legend-dot--green { background: #10B981; }
.legend-count { color: var(--sd-text); font-weight: 700; margin-left: .1rem; }

.form-note {
    display: flex; gap: .5rem; align-items: flex-start; background: var(--sd-surface-soft);
    border: 1px solid var(--sd-border); border-radius: 9px; padding: .65rem .8rem;
    color: var(--sd-text-muted); font-size: .76rem; line-height: 1.5; margin-top: .25rem;
}
.form-note svg { color: var(--sd-primary); flex-shrink: 0; margin-top: .05rem; }

.control-spacer { height: 1.7rem; }
.app-footer {
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    border-top: 1px solid var(--sd-border); margin-top: 2.5rem; padding-top: 1rem;
    color: var(--sd-text-faint); font-size: .72rem;
}
.app-footer strong { color: var(--sd-text-muted); font-weight: 650; }

div[data-testid="stWidgetLabel"] p { font-size: .82rem; font-weight: 650; color: var(--sd-text-secondary); }
div[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-testid="stTextArea"] div[data-baseweb="textarea"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    border: 1px solid var(--sd-border-strong);
    border-radius: var(--sd-radius-sm);
    background: var(--sd-surface);
    box-shadow: none;
    transition: border-color .15s ease, box-shadow .15s ease;
}
div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
div[data-testid="stTextArea"] div[data-baseweb="textarea"]:focus-within,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    border-color: var(--sd-primary);
    box-shadow: 0 0 0 3px rgba(37, 99, 235, .12);
}
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    color: var(--sd-text); font-size: .9rem; background: transparent;
}
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder { color: var(--sd-text-faint); }
div[data-testid="stSelectbox"] div[data-baseweb="select"] div { font-size: .9rem; }
div[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }

.stButton button, div[data-testid="stFormSubmitButton"] button {
    min-height: 40px;
    border-radius: var(--sd-radius-sm);
    font-weight: 650;
    font-size: .88rem;
    padding: .45rem 1.05rem;
    border: 1px solid var(--sd-border-strong);
    background: var(--sd-surface);
    color: var(--sd-text-secondary);
    box-shadow: 0 1px 2px rgba(15, 23, 42, .04);
    transition: all .15s ease;
}
.stButton button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    border-color: #94A3B8;
    color: var(--sd-text);
    transform: translateY(-1px);
}
.stButton button[kind="primary"], div[data-testid="stFormSubmitButton"] button[kind="primary"],
.stButton button[data-testid="stBaseButton-primary"], div[data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"] {
    background: var(--sd-primary);
    border-color: var(--sd-primary);
    color: #FFFFFF;
    box-shadow: 0 2px 6px -2px rgba(37, 99, 235, .5);
}
.stButton button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
.stButton button[data-testid="stBaseButton-primary"]:hover, div[data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"]:hover {
    background: var(--sd-primary-strong);
    border-color: var(--sd-primary-strong);
    color: #FFFFFF;
}

div[data-testid="stAlert"] { border-radius: 12px; }
div[data-testid="stDataFrame"] {
    border: 1px solid var(--sd-border); border-radius: 12px; overflow: hidden; background: var(--sd-surface);
}
details[data-testid="stExpander"] {
    border: 1px solid var(--sd-border); border-radius: 12px; background: var(--sd-surface); box-shadow: none;
}
details[data-testid="stExpander"] summary { font-weight: 650; color: var(--sd-text-secondary); }
div[data-testid="stVegaLiteChart"], div[data-testid="stArrowVegaLiteChart"] {
    padding: 0 !important; background: transparent !important;
}
div[data-testid="stSpinner"] { color: var(--sd-text-muted); }
[data-testid="stCaptionContainer"] p { color: var(--sd-text-muted); font-size: .76rem; line-height: 1.5; }

@media (max-width: 1100px) {
    .result-grid, .meta-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 900px) {
    .block-container { padding-top: 1.2rem; }
    .page-hero { flex-direction: column; align-items: flex-start; padding: 1.2rem; }
    .hero-aside { align-items: flex-start; }
    .page-hero h1 { font-size: 1.65rem; }
    .subject-cell { max-width: 220px; }
}
@media (max-width: 640px) {
    .result-grid, .meta-grid { grid-template-columns: 1fr; }
    .app-footer { flex-direction: column; align-items: flex-start; }
    .table-head { flex-direction: column; }
}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def icon(name: str, size: int = 18, class_name: str = "") -> str:
    """Return an inline SVG icon from the local icon set."""
    class_attr = f' class="{class_name}"' if class_name else ""
    return (
        f'<svg{class_attr} xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )


def esc(value: Any) -> str:
    """Escape dynamic text before placing it inside custom HTML."""
    return html.escape(str(value))


def render(markup: str) -> None:
    """Render custom HTML while preserving inline SVG icons."""
    compact = "".join(
        line.strip() for line in textwrap.dedent(markup).strip().splitlines()
    )
    st.markdown(compact, unsafe_allow_html=True)


def page_hero(eyebrow: str, title: str, subtitle: str, aside: str = "") -> None:
    aside_html = f'<div class="hero-aside">{aside}</div>' if aside else ""
    render(
        f"""
        <div class="page-hero">
            <div class="page-hero-copy">
                <div class="eyebrow">{esc(eyebrow)}</div>
                <h1>{esc(title)}</h1>
                <p>{esc(subtitle)}</p>
            </div>
            {aside_html}
        </div>
        """
    )


def live_pill(label: str = "Live data") -> str:
    return f'<span class="live-pill"><span class="live-dot"></span>{esc(label)}</span>'


def soft_pill(label: str) -> str:
    return f'<span class="soft-pill">{icon("shield", 14)}{esc(label)}</span>'


def stat_card(label: str, value: Any, hint: str, icon_name: str, tone: str = "blue") -> None:
    render(
        f"""
        <div class="stat-card stat-card--{tone}">
            <div class="stat-top">
                <div class="stat-icon">{icon(icon_name, 18)}</div>
                <div class="stat-label">{esc(label)}</div>
            </div>
            <div class="stat-value">{esc(value)}</div>
            <div class="stat-foot">{esc(hint)}</div>
        </div>
        """
    )


def card_header(title: str, subtitle: str, icon_name: str) -> None:
    render(
        f"""
        <div class="card-head">
            <div>
                <div class="card-title">{esc(title)}</div>
                <div class="card-subtitle">{esc(subtitle)}</div>
            </div>
            <div class="card-icon">{icon(icon_name, 17)}</div>
        </div>
        """
    )


def section_heading(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eyebrow_html = f'<div class="eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    subtitle_html = f"<p>{esc(subtitle)}</p>" if subtitle else ""
    render(
        f'<div class="section-heading">{eyebrow_html}<h2>{esc(title)}</h2>{subtitle_html}</div>'
    )


def badge(text: str, tone: str = "slate", dot: bool = False) -> str:
    dot_html = '<span class="badge-dot"></span>' if dot else ""
    return f'<span class="badge badge--{tone}">{dot_html}{esc(text)}</span>'


def priority_badge(priority: str) -> str:
    tone = {"High": "red", "Medium": "amber", "Low": "green"}.get(priority, "slate")
    return badge(priority, tone, dot=True)


def status_badge(status: str) -> str:
    tone = {"Pending": "amber", "In Progress": "blue", "Resolved": "green"}.get(status, "slate")
    return badge(status, tone, dot=True)


def category_badge(category: str) -> str:
    tone = {
        "Network Issue": "blue",
        "Account Issue": "violet",
        "Hardware Issue": "amber",
        "Software Issue": "blue",
        "General Inquiry": "slate",
    }.get(category, "slate")
    return badge(category, tone)


def alert(kind: str, title: str, body: str = "") -> None:
    icon_name = {
        "success": "check-circle",
        "warning": "alert",
        "error": "x-circle",
        "info": "info",
    }.get(kind, "info")
    body_html = f'<div class="sd-alert-body">{body}</div>' if body else ""
    render(
        f"""
        <div class="sd-alert sd-alert--{kind}">
            <div class="sd-alert-icon">{icon(icon_name, 18)}</div>
            <div>
                <div class="sd-alert-title">{esc(title)}</div>
                {body_html}
            </div>
        </div>
        """
    )


def empty_state(title: str, message: str, icon_name: str = "inbox", compact: bool = False) -> None:
    extra = " empty-state--compact" if compact else ""
    render(
        f"""
        <div class="empty-state{extra}">
            <div class="empty-icon">{icon(icon_name, 22)}</div>
            <div class="empty-title">{esc(title)}</div>
            <div class="empty-text">{esc(message)}</div>
        </div>
        """
    )


def ticket_table_html(
    frame: pd.DataFrame,
    title: str,
    subtitle: str,
    count_label: str,
) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            f"""
            <tr>
                <td><span class="ticket-code">{esc(db.format_ticket_id(row.id))}</span></td>
                <td><span class="subject-cell" title="{esc(row.subject)}">{esc(row.subject)}</span></td>
                <td>{category_badge(str(row.category))}</td>
                <td>{priority_badge(str(row.priority))}</td>
                <td>{status_badge(str(row.status))}</td>
                <td class="muted-cell">{esc(row.created_at)}</td>
            </tr>
            """
        )
    return f"""
    <div class="table-block">
        <div class="table-head">
            <div>
                <div class="card-title">{esc(title)}</div>
                <div class="card-subtitle">{esc(subtitle)}</div>
            </div>
            <span class="table-count">{esc(count_label)}</span>
        </div>
        <div class="table-scroll">
            <table class="data-table">
                <thead>
                    <tr>
                        <th scope="col">Ticket</th>
                        <th scope="col">Subject</th>
                        <th scope="col">Category</th>
                        <th scope="col">Priority</th>
                        <th scope="col">Status</th>
                        <th scope="col">Created</th>
                    </tr>
                </thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
        </div>
    </div>
    """


def evaluation_table_html(frame: pd.DataFrame) -> str:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            f"""
            <tr>
                <td><span class="subject-cell">{esc(row["Category"])}</span></td>
                <td>{float(row["Precision"]):.4f}</td>
                <td>{float(row["Recall"]):.4f}</td>
                <td>{float(row["F1-score"]):.4f}</td>
                <td>{int(row["Support"])}</td>
            </tr>
            """
        )
    return f"""
    <div class="table-block">
        <div class="table-head">
            <div>
                <div class="card-title">Per-category evaluation</div>
                <div class="card-subtitle">Precision, recall and F1-score on the held-out test set.</div>
            </div>
            <span class="table-count">{len(frame)} categories</span>
        </div>
        <div class="table-scroll">
            <table class="data-table">
                <thead>
                    <tr>
                        <th scope="col">Category</th>
                        <th scope="col">Precision</th>
                        <th scope="col">Recall</th>
                        <th scope="col">F1-score</th>
                        <th scope="col">Support</th>
                    </tr>
                </thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
        </div>
    </div>
    """


def style_chart(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    return (
        chart.configure(background="transparent", padding=0, font="Segoe UI")
        .configure_view(strokeWidth=0)
    )


def category_chart(counts: dict[str, int]) -> alt.TopLevelMixin:
    order = [category for category, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)]
    frame = pd.DataFrame({"Category": order, "Tickets": [int(counts[category]) for category in order]})
    max_value = max(int(frame["Tickets"].max()), 1)
    base = alt.Chart(frame).encode(
        y=alt.Y(
            "Category:N",
            sort=order,
            title=None,
            axis=alt.Axis(
                labelColor="#475569",
                labelFontSize=11,
                labelLimit=140,
                labelPadding=8,
                ticks=False,
                domain=False,
            ),
        ),
        x=alt.X(
            "Tickets:Q",
            title=None,
            scale=alt.Scale(domain=[0, max_value * 1.2]),
            axis=alt.Axis(
                grid=True,
                gridColor="#EEF2F7",
                tickCount=4,
                tickMinStep=1,
                format="d",
                domain=False,
                ticks=False,
                labelColor="#94A3B8",
                labelFontSize=10,
            ),
        ),
    )
    bars = base.mark_bar(cornerRadiusEnd=6, color=CHART_BLUE, height=18)
    labels = base.mark_text(
        align="left", dx=6, color="#475569", fontSize=11, fontWeight=600
    ).encode(text="Tickets:Q")
    return style_chart((bars + labels).properties(height=220))


def priority_chart(counts: dict[str, int]) -> alt.TopLevelMixin:
    order = ["High", "Medium", "Low"]
    frame = pd.DataFrame({"Priority": order, "Tickets": [int(counts.get(p, 0)) for p in order]})
    max_value = max(int(frame["Tickets"].max()), 1)
    base = alt.Chart(frame).encode(
        x=alt.X(
            "Priority:N",
            sort=order,
            title=None,
            axis=alt.Axis(
                labelColor="#475569",
                labelFontSize=11,
                labelPadding=6,
                ticks=False,
                domain=False,
            ),
        ),
        y=alt.Y(
            "Tickets:Q",
            title=None,
            scale=alt.Scale(domain=[0, max_value * 1.25]),
            axis=alt.Axis(
                grid=True,
                gridColor="#EEF2F7",
                tickCount=4,
                tickMinStep=1,
                format="d",
                domain=False,
                ticks=False,
                labelColor="#94A3B8",
                labelFontSize=10,
            ),
        ),
    )
    bars = base.mark_bar(
        cornerRadiusTopLeft=6, cornerRadiusTopRight=6, width=46
    ).encode(
        color=alt.Color(
            "Priority:N",
            scale=alt.Scale(domain=order, range=[CHART_RED, CHART_AMBER, CHART_GREEN]),
            legend=None,
        ),
        tooltip=[alt.Tooltip("Priority:N"), alt.Tooltip("Tickets:Q")],
    )
    labels = base.mark_text(dy=-9, color="#475569", fontSize=11, fontWeight=600).encode(
        text="Tickets:Q"
    )
    return style_chart((bars + labels).properties(height=220))


def status_chart(status_counts: dict[str, int], total: int) -> alt.TopLevelMixin:
    order = ["Pending", "In Progress", "Resolved"]
    frame = pd.DataFrame(
        {"Status": order, "Tickets": [int(status_counts.get(status, 0)) for status in order]}
    )
    arc = alt.Chart(frame).mark_arc(
        innerRadius=58, outerRadius=86, cornerRadius=4, padAngle=0.02
    ).encode(
        theta=alt.Theta("Tickets:Q", stack=True),
        color=alt.Color(
            "Status:N",
            scale=alt.Scale(
                domain=order,
                range=[CHART_STATUS_AMBER, CHART_STATUS_BLUE, CHART_STATUS_GREEN],
            ),
            legend=None,
        ),
        tooltip=[alt.Tooltip("Status:N"), alt.Tooltip("Tickets:Q")],
    )
    total_text = (
        alt.Chart(pd.DataFrame({"label": [str(total)]}))
        .mark_text(fontSize=28, fontWeight=700, color="#0F172A", dy=-2)
        .encode(text="label:N")
    )
    sub_text = (
        alt.Chart(pd.DataFrame({"label": ["TOTAL"]}))
        .mark_text(fontSize=9, fontWeight=700, color="#94A3B8", dy=18)
        .encode(text="label:N")
    )
    return style_chart(alt.layer(arc, total_text, sub_text).properties(height=220))


def chart_legend(status_counts: dict[str, int]) -> None:
    items = (
        ("Pending", "amber"),
        ("In Progress", "blue"),
        ("Resolved", "green"),
    )
    html_items = "".join(
        f'<span class="legend-item"><span class="legend-dot legend-dot--{tone}"></span>'
        f'{esc(label)}<span class="legend-count">{int(status_counts.get(label, 0))}</span></span>'
        for label, tone in items
    )
    render(f'<div class="chart-legend">{html_items}</div>')


def train_model_from_ui() -> None:
    with st.spinner("Training the TF-IDF + Naive Bayes model..."):
        train()
        predictor.clear_model_cache()


def navigate_to(page: str) -> None:
    st.session_state["nav_page"] = page


def navigate_to_ticket(ticket_id: int) -> None:
    st.session_state["nav_page"] = "Manage Tickets"
    st.session_state["manage_pending_ticket"] = int(ticket_id)


def clear_last_ticket() -> None:
    st.session_state.pop("last_ticket", None)


def sidebar() -> str:
    with st.sidebar:
        render(
            f"""
            <div class="brand">
                <div class="brand-mark">{icon("ticket", 22)}</div>
                <div>
                    <div class="brand-name">SmartDesk AI</div>
                    <div class="brand-sub">Help desk workspace</div>
                </div>
            </div>
            """
        )
        render('<div class="nav-label">Workspace</div>')
        page = st.radio(
            "Navigation",
            NAV_PAGES,
            label_visibility="collapsed",
            key="nav_page",
        )
        render('<div class="sidebar-divider"></div>')

        if predictor.model_exists():
            render(
                """
                <div class="sidebar-status">
                    <div class="sidebar-status-top">
                        <span class="status-dot status-dot--green"></span>AI model
                    </div>
                    <div class="sidebar-status-value">Ready to classify</div>
                    <div class="sidebar-status-meta">TF-IDF + Naive Bayes</div>
                </div>
                """
            )
        else:
            render(
                """
                <div class="sidebar-status">
                    <div class="sidebar-status-top">
                        <span class="status-dot status-dot--amber"></span>AI model
                    </div>
                    <div class="sidebar-status-value">Not trained</div>
                    <div class="sidebar-status-meta">Train the model to enable AI routing.</div>
                </div>
                """
            )
            if st.button("Train AI model", type="primary", width="stretch"):
                train_model_from_ui()
                st.rerun()

        render(
            """
            <div class="sidebar-divider"></div>
            <div class="sidebar-footer">
                <strong>Local-first help desk</strong><br>
                SQLite database · scikit-learn model
            </div>
            """
        )
    return page


def dashboard_page() -> None:
    stats = db.get_statistics()
    status_counts = stats["status_counts"]
    total = int(stats["total"])
    resolved = int(status_counts["Resolved"])
    resolution_rate = (resolved / total * 100) if total else 0.0

    page_hero(
        "Operations overview",
        "Dashboard",
        "A live view of every complaint, triage decision and resolution in the help desk.",
        aside=live_pill("Live data")
        + f'<span class="hero-meta">Updated {esc(datetime.now().strftime("%d %b %Y, %H:%M"))}</span>',
    )

    columns = st.columns(4, gap="medium")
    with columns[0]:
        stat_card("Total tickets", total, "All complaints on record", "ticket", "blue")
    with columns[1]:
        stat_card("Pending", int(status_counts["Pending"]), "Waiting for first review", "clock", "amber")
    with columns[2]:
        stat_card("In progress", int(status_counts["In Progress"]), "Currently being worked on", "progress", "violet")
    with columns[3]:
        stat_card("Resolved", resolved, f"{resolution_rate:.0f}% resolution rate", "check-circle", "green")

    chart_columns = st.columns(3, gap="medium")
    with chart_columns[0]:
        with st.container(key="card_chart_category"):
            card_header("Tickets by category", "Where complaint volume is concentrated", "layers")
            if stats["category_counts"]:
                st.altair_chart(category_chart(stats["category_counts"]), width="stretch", theme=None)
            else:
                empty_state(
                    "No category data yet",
                    "Submit a complaint to start building the distribution.",
                    "layers",
                    compact=True,
                )
    with chart_columns[1]:
        with st.container(key="card_chart_priority"):
            card_header("Tickets by priority", "Rule-based urgency across the queue", "activity")
            if stats["priority_counts"]:
                st.altair_chart(priority_chart(stats["priority_counts"]), width="stretch", theme=None)
            else:
                empty_state(
                    "No priority data yet",
                    "Priority appears after the first complaint is submitted.",
                    "activity",
                    compact=True,
                )
    with chart_columns[2]:
        with st.container(key="card_chart_status"):
            card_header("Tickets by status", "The current shape of the workflow", "target")
            if total:
                st.altair_chart(status_chart(status_counts, total), width="stretch", theme=None)
                chart_legend(status_counts)
            else:
                empty_state(
                    "No status data yet",
                    "Status counts appear as tickets move through the workflow.",
                    "target",
                    compact=True,
                )

    recent = db.get_all_tickets()
    if recent.empty:
        with st.container(key="card_dashboard_empty"):
            empty_state(
                "Your help desk is ready",
                "No complaints have been submitted yet. Start the happy flow by creating the first ticket.",
                "inbox",
            )
        st.button(
            "Submit your first complaint",
            type="primary",
            on_click=navigate_to,
            args=("Submit Complaint",),
        )
    else:
        render(
            ticket_table_html(
                recent.head(5),
                title="Recent tickets",
                subtitle="The five latest complaints submitted to the help desk.",
                count_label=f"{min(5, len(recent))} of {len(recent)}",
            )
        )
        st.caption("Open Manage Tickets to search, filter and update the full queue.")


def render_submission_result(ticket: dict[str, Any]) -> None:
    ticket_code = db.format_ticket_id(ticket["id"])
    confidence = float(ticket["confidence"])
    confidence_pct = max(0.0, min(confidence * 100, 100.0))
    low_confidence = bool(ticket["low_confidence"])

    section_heading(
        "Submission complete",
        "The ticket is in the queue and ready for review.",
        "Happy flow",
    )
    alert(
        "success",
        "Complaint submitted successfully",
        f"Your ticket ID is <strong>{esc(ticket_code)}</strong>. Keep it handy when following up with support.",
    )

    with st.container(key="card_result"):
        render(
            f"""
            <div class="result-head">
                <div>
                    <div class="eyebrow">Ticket created</div>
                    <div class="result-id">{esc(ticket_code)}</div>
                </div>
                {status_badge("Pending")}
            </div>
            <div class="result-grid">
                <div class="result-item">
                    <div class="result-label">Category</div>
                    <div class="result-value">{category_badge(str(ticket["category"]))}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Priority</div>
                    <div class="result-value">{priority_badge(str(ticket["priority"]))}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Confidence</div>
                    <div class="result-value">{confidence_pct:.1f}%</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Assigned queue</div>
                    <div class="result-value">{esc(ticket["category"])}</div>
                </div>
            </div>
            <div class="meter{' meter--warning' if low_confidence else ''}">
                <div class="meter-fill" style="width: {confidence_pct:.1f}%"></div>
            </div>
            <div class="result-note">{esc(ticket["priority_reason"])}</div>
            """
        )

    if low_confidence:
        alert("warning", "Manual review recommended", esc(ticket["note"]))

    with st.container(key="card_response"):
        card_header(
            "Suggested first response",
            "An automated reply the support team can review and send.",
            "message",
        )
        render(
            f'<div class="response-box"><div class="response-icon">{icon("message", 18)}</div>'
            f'<div>{esc(ticket["response"])}</div></div>'
        )

    actions = st.columns(2, gap="medium")
    with actions[0]:
        st.button(
            "View in Manage Tickets",
            type="primary",
            width="stretch",
            on_click=navigate_to_ticket,
            args=(int(ticket["id"]),),
        )
    with actions[1]:
        st.button("Submit another complaint", width="stretch", on_click=clear_last_ticket)


def submit_page() -> None:
    page_hero(
        "AI-assisted intake",
        "Submit a complaint",
        "Describe the issue once. SmartDesk AI classifies it, assigns a transparent priority and drafts a first-line response.",
        aside=soft_pill("Local & secure"),
    )

    if not predictor.model_exists():
        alert(
            "warning",
            "The AI model is not trained yet",
            "Train the model from the sidebar or run <code>python -m ml.train_model</code>, then reload this page.",
        )
        if st.button("Train AI model now", type="primary"):
            train_model_from_ui()
            st.rerun()
        return

    left, right = st.columns([1.6, 1], gap="large")

    with left:
        with st.container(key="card_complaint_form"):
            card_header(
                "Complaint details",
                "Give the support team enough context to act quickly.",
                "edit",
            )
            with st.form("complaint_form", clear_on_submit=False):
                row = st.columns(2, gap="medium")
                with row[0]:
                    name = st.text_input("Full name", placeholder="Enter your full name")
                with row[1]:
                    email = st.text_input("Email address", placeholder="you@example.com")
                subject = st.text_input(
                    "Complaint subject", placeholder="Short summary of the issue"
                )
                description = st.text_area(
                    "Complaint description",
                    placeholder="Explain the issue in detail. Include error messages or symptoms if possible.",
                    height=160,
                )
                render(
                    f'<div class="form-note">{icon("shield", 15)}'
                    "<span>Your complaint is stored locally in SQLite. Nothing is sent to an external service.</span></div>"
                )
                submitted = st.form_submit_button(
                    "Submit complaint", type="primary", width="stretch"
                )

            if submitted:
                errors = validate_complaint(name, email, subject, description)
                if errors:
                    items = "".join(
                        f"<li>{esc(message)}</li>" for message in errors.values()
                    )
                    alert(
                        "error",
                        "Please correct the following before submitting",
                        f'<ul class="alert-list">{items}</ul>',
                    )
                else:
                    with st.spinner("Running AI classification and priority rules..."):
                        classification = classify_complaint(subject, description)
                        priority_result = detect_priority(f"{subject} {description}")
                        suggested_response = get_suggested_response(
                            str(classification["category"])
                        )
                        ticket_id = db.create_ticket(
                            name=name,
                            email=email,
                            subject=subject,
                            description=description,
                            category=str(classification["category"]),
                            priority=str(priority_result["priority"]),
                            suggested_response=suggested_response,
                            confidence=float(classification["confidence"]),
                        )

                    st.session_state["last_ticket"] = {
                        "id": ticket_id,
                        "category": classification["category"],
                        "confidence": classification["confidence"],
                        "low_confidence": classification["low_confidence"],
                        "note": classification["note"],
                        "priority": priority_result["priority"],
                        "priority_reason": priority_result["reason"],
                        "response": suggested_response,
                    }

    with right:
        with st.container(key="card_next_steps"):
            card_header(
                "What happens next",
                "A clear path from submission to resolution.",
                "route",
            )
            render(
                """
                <div class="step-list">
                    <div class="step">
                        <span class="step-num">1</span>
                        <div>
                            <div class="step-title">AI classifies the issue</div>
                            <div class="step-text">TF-IDF + Naive Bayes predicts the most likely category and shows its confidence.</div>
                        </div>
                    </div>
                    <div class="step">
                        <span class="step-num">2</span>
                        <div>
                            <div class="step-title">Priority is assigned</div>
                            <div class="step-text">Transparent keyword rules set High, Medium or Low urgency and explain the match.</div>
                        </div>
                    </div>
                    <div class="step">
                        <span class="step-num">3</span>
                        <div>
                            <div class="step-title">Support takes over</div>
                            <div class="step-text">The ticket lands in the queue with a suggested first-line response for review.</div>
                        </div>
                    </div>
                </div>
                """
            )
        with st.container(key="card_ai_routing"):
            card_header("AI routing rules", "How the system handles uncertainty.", "cpu")
            render(
                """
                <div class="stack-list">
                    <div class="stack-row"><span class="stack-layer">Classifier</span><span class="stack-tech">TF-IDF + Naive Bayes</span></div>
                    <div class="stack-row"><span class="stack-layer">Confidence floor</span><span class="stack-tech">30%</span></div>
                    <div class="stack-row"><span class="stack-layer">Low confidence</span><span class="stack-tech">Manual review</span></div>
                    <div class="stack-row"><span class="stack-layer">Priority logic</span><span class="stack-tech">Explainable rules</span></div>
                </div>
                """
            )

    last_ticket = st.session_state.get("last_ticket")
    if last_ticket:
        render_submission_result(last_ticket)


def render_ticket_detail(ticket: dict[str, Any]) -> None:
    ticket_code = db.format_ticket_id(ticket["id"])
    confidence = float(ticket["confidence"])

    with st.container(key="card_ticket_detail"):
        render(
            f"""
            <div class="detail-header">
                <div>
                    <div class="detail-id">{esc(ticket_code)}</div>
                    <div class="detail-title">{esc(ticket["subject"])}</div>
                    <div class="detail-badges">
                        {status_badge(str(ticket["status"]))}
                        {priority_badge(str(ticket["priority"]))}
                        {category_badge(str(ticket["category"]))}
                    </div>
                </div>
                <div class="card-icon">{icon("file", 17)}</div>
            </div>
            <div class="meta-grid">
                <div>
                    <div class="meta-label">Submitted by</div>
                    <div class="meta-value">{esc(ticket["name"])}</div>
                </div>
                <div>
                    <div class="meta-label">Email</div>
                    <div class="meta-value">{esc(ticket["email"])}</div>
                </div>
                <div>
                    <div class="meta-label">Created at</div>
                    <div class="meta-value">{esc(ticket["created_at"])}</div>
                </div>
                <div>
                    <div class="meta-label">AI confidence</div>
                    <div class="meta-value">{confidence * 100:.1f}%</div>
                </div>
            </div>
            <div class="detail-section">
                <div class="detail-label">Complaint description</div>
                <div class="inset">{esc(ticket["description"]).replace(chr(10), "<br>")}</div>
            </div>
            <div class="detail-section">
                <div class="detail-label">Suggested first response</div>
                <div class="response-box">
                    <div class="response-icon">{icon("message", 18)}</div>
                    <div>{esc(ticket["suggested_response"])}</div>
                </div>
            </div>
            """
        )

    if confidence < CONFIDENCE_THRESHOLD:
        alert(
            "warning",
            "Low-confidence prediction",
            "This prediction had low confidence and was routed to General Inquiry for manual review.",
        )

    with st.container(key="card_status_update"):
        card_header("Update status", "Move this ticket through the workflow.", "refresh")
        update_columns = st.columns([2, 1], gap="medium")
        with update_columns[0]:
            new_status = st.selectbox(
                "New status",
                db.STATUSES,
                index=db.STATUSES.index(ticket["status"]),
                key=f"status_{ticket['id']}",
            )
        with update_columns[1]:
            render('<div class="control-spacer"></div>')
            if st.button("Update status", type="primary", width="stretch"):
                if db.update_ticket_status(int(ticket["id"]), new_status):
                    st.session_state["manage_flash"] = (
                        f"{ticket_code} status updated to {new_status}."
                    )
                    st.rerun()
                else:
                    alert("error", "Update failed", "The ticket could not be updated.")


def manage_page() -> None:
    page_hero(
        "Ticket operations",
        "Manage tickets",
        "Search the queue, inspect AI decisions and move every complaint through the workflow.",
        aside=soft_pill("Search & update"),
    )

    flash = st.session_state.pop("manage_flash", None)
    if flash:
        alert("success", "Status updated", esc(flash))

    with st.container(key="card_filters"):
        card_header(
            "Filters",
            "Narrow the queue by keyword, category, priority or status.",
            "filter",
        )
        filter_columns = st.columns([2.2, 1, 1, 1], gap="medium")
        with filter_columns[0]:
            search = st.text_input(
                "Search by ticket ID or subject",
                placeholder="Example: TKT-000001 or printer",
            )
        with filter_columns[1]:
            category = st.selectbox("Category", ("All",) + db.CATEGORIES)
        with filter_columns[2]:
            priority = st.selectbox("Priority", ("All",) + db.PRIORITIES)
        with filter_columns[3]:
            status = st.selectbox("Status", ("All",) + db.STATUSES)

    tickets = db.get_all_tickets(
        search=search,
        category=category,
        priority=priority,
        status=status,
    )

    if tickets.empty:
        with st.container(key="card_manage_empty"):
            card_header(
                "Matching tickets",
                "No tickets match the current filter set.",
                "search",
            )
            empty_state(
                "Nothing in this view",
                "Try a broader search or reset one of the filter fields.",
                "search",
                compact=True,
            )
        return

    render(
        ticket_table_html(
            tickets,
            title="Matching tickets",
            subtitle="Every ticket that matches the current filter set.",
            count_label=f"{len(tickets)} matching",
        )
    )

    options = tickets["id"].tolist()
    pending = st.session_state.pop("manage_pending_ticket", None)
    if pending is not None and int(pending) in options:
        st.session_state["manage_ticket_select"] = int(pending)
    if (
        "manage_ticket_select" in st.session_state
        and st.session_state["manage_ticket_select"] not in options
    ):
        del st.session_state["manage_ticket_select"]

    with st.container(key="card_ticket_picker"):
        card_header(
            "Ticket inspector",
            "Choose a ticket to review its full history.",
            "search",
        )
        selected_id = st.selectbox(
            "Inspect a ticket",
            options=options,
            format_func=db.format_ticket_id,
            key="manage_ticket_select",
        )

    ticket = db.get_ticket(int(selected_id))
    if ticket is None:
        alert(
            "error",
            "Ticket not found",
            "The selected ticket could not be loaded. Try refreshing the page.",
        )
        return

    render_ticket_detail(ticket)


def about_page() -> None:
    page_hero(
        "Project overview",
        "About SmartDesk AI",
        "A third-year engineering mini-project that pairs a real machine-learning classifier with an explainable help desk workflow.",
        aside=soft_pill("Fundamentals of AI Lab"),
    )

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        with st.container(key="card_overview"):
            card_header(
                "Project overview",
                "What the application does and why it exists.",
                "book",
            )
            render(
                """
                <p>SmartDesk AI is a local help desk application that accepts employee complaints, predicts their category with a trained machine-learning model, assigns an explainable priority, suggests a first-line response and stores every ticket in SQLite.</p>
                <p>Administrators can search, filter and update tickets from the Manage Tickets page, while the dashboard keeps the queue visible at a glance.</p>
                <div class="detail-section">
                    <div class="detail-label">Problem statement</div>
                    <div class="inset">Manual complaint sorting is slow and inconsistent. Support staff must read each complaint, decide its category and priority, and write a first response. SmartDesk AI automates the first pass so tickets reach the right team faster.</div>
                </div>
                """
            )
        with st.container(key="card_objectives"):
            card_header(
                "Objectives",
                "The outcomes this project is designed to prove.",
                "target",
            )
            objectives = [
                "Accept and validate complaints through a web form.",
                "Classify complaint text with a real machine-learning model.",
                "Show prediction confidence instead of hiding uncertainty.",
                "Detect priority with transparent rule-based logic.",
                "Suggest a standard response from the predicted category.",
                "Store tickets permanently in SQLite.",
                "Give administrators a live dashboard and management page.",
            ]
            items = "".join(
                f'<li>{icon("check-circle", 15, "check-icon")}<span>{esc(item)}</span></li>'
                for item in objectives
            )
            render(f'<ul class="check-list">{items}</ul>')

    with right:
        with st.container(key="card_stack"):
            card_header("Technology stack", "The tools behind the workflow.", "layers")
            stack = [
                ("Language", "Python"),
                ("Frontend", "Streamlit"),
                ("ML library", "Scikit-learn"),
                ("Features", "TF-IDF"),
                ("Classifier", "Multinomial Naive Bayes"),
                ("Database", "SQLite"),
                ("Data processing", "Pandas"),
                ("Testing", "Pytest"),
                ("Model storage", "Joblib"),
            ]
            rows = "".join(
                f'<div class="stack-row"><span class="stack-layer">{esc(layer)}</span>'
                f'<span class="stack-tech">{esc(tech)}</span></div>'
                for layer, tech in stack
            )
            render(f'<div class="stack-list">{rows}</div>')
        with st.container(key="card_how_ai"):
            card_header("How the AI works", "From raw text to a ranked category.", "cpu")
            steps = [
                (
                    "Clean and tokenize",
                    "Complaint text is cleaned and split into words and word pairs.",
                ),
                (
                    "Weight the terms",
                    "TF-IDF converts terms into numerical weights. Terms that are common in one category but rare elsewhere get more weight.",
                ),
                (
                    "Learn the patterns",
                    "Multinomial Naive Bayes learns the probability of each term for each category during training.",
                ),
                (
                    "Predict with confidence",
                    "For a new complaint, the model combines those probabilities and returns the most likely category with a confidence score.",
                ),
            ]
            step_html = "".join(
                f'<div class="step"><span class="step-num">{index}</span><div>'
                f'<div class="step-title">{esc(title)}</div>'
                f'<div class="step-text">{esc(text)}</div></div></div>'
                for index, (title, text) in enumerate(steps, start=1)
            )
            render(f'<div class="step-list">{step_html}</div>')

    section_heading(
        "Model evaluation",
        "Real results from a single train/test split on the demonstration dataset.",
        "Measured, not guessed",
    )

    metrics = predictor.get_model_metrics()
    if metrics is None:
        alert(
            "warning",
            "No evaluation results found",
            "Train the model with <code>python -m ml.train_model</code> and reload this page.",
        )
        return

    metric_columns = st.columns(4, gap="medium")
    with metric_columns[0]:
        stat_card(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%",
            "Correct predictions on the test set",
            "target",
            "blue",
        )
    with metric_columns[1]:
        stat_card(
            "Precision",
            f"{metrics['precision_macro'] * 100:.2f}%",
            "Macro average across categories",
            "chart",
            "violet",
        )
    with metric_columns[2]:
        stat_card(
            "Recall",
            f"{metrics['recall_macro'] * 100:.2f}%",
            "Macro average across categories",
            "activity",
            "green",
        )
    with metric_columns[3]:
        stat_card(
            "F1-score",
            f"{metrics['f1_macro'] * 100:.2f}%",
            "Macro average across categories",
            "check-circle",
            "amber",
        )

    st.caption(
        f"Trained on {metrics['train_size']} records and evaluated on {metrics['test_size']} "
        f"held-out records (dataset size: {metrics['dataset_size']}, random state: "
        f"{metrics['random_state']}, trained at: {metrics['trained_at']})."
    )

    report = metrics["classification_report"]
    labels = metrics["labels"]
    rows = []
    for label in labels:
        label_metrics = report[label]
        rows.append(
            {
                "Category": label,
                "Precision": round(label_metrics["precision"], 4),
                "Recall": round(label_metrics["recall"], 4),
                "F1-score": round(label_metrics["f1-score"], 4),
                "Support": int(label_metrics["support"]),
            }
        )
    render(evaluation_table_html(pd.DataFrame(rows)))

    with st.expander("Confusion matrix and model details"):
        matrix_frame = pd.DataFrame(
            metrics["confusion_matrix"], index=labels, columns=labels
        )
        st.write("Rows are actual categories; columns are predicted categories.")
        st.dataframe(matrix_frame, width="stretch")
        st.json(
            {
                "model": metrics["model"],
                "alpha": metrics["alpha"],
                "test_split": metrics["test_split"],
                "random_state": metrics["random_state"],
                "dataset_size": metrics["dataset_size"],
            }
        )

    alert(
        "info",
        "A demonstration dataset, not a production benchmark",
        "This project uses a small demonstration dataset (100 sample complaints). "
        "The metrics above are real results from a single train/test split, but they "
        "do not prove performance on real-world complaints. A production system would "
        "need more representative, balanced and regularly updated data.",
    )

    if st.button("Retrain model now", type="primary"):
        train_model_from_ui()
        st.success("Model retrained successfully. Metrics on this page have been refreshed.")
        st.rerun()


def main() -> None:
    page = sidebar()

    if page == "Dashboard":
        dashboard_page()
    elif page == "Submit Complaint":
        submit_page()
    elif page == "Manage Tickets":
        manage_page()
    else:
        about_page()

    render(
        """
        <div class="app-footer">
            <span><strong>SmartDesk AI</strong> · Local-first help desk</span>
            <span>SQLite · scikit-learn · Streamlit</span>
        </div>
        """
    )


if __name__ == "__main__":
    main()
