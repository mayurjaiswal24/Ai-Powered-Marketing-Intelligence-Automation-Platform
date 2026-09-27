# AI-Powered Marketing Intelligence & Automation Platform

An AI-powered Marketing Intelligence and Decision Support Platform that automates the journey
from raw marketing data to verified analytics, AI-assisted interpretation, interactive
visualization and executive reporting.

Upload a marketing CSV or Excel export. Python profiles, maps, cleans and stores it (SQLite),
then calculates every KPI, trend, funnel, segment view and anomaly deterministically. Google
Gemini only interprets a compact, verified evidence summary; every AI statement cites its
evidence and every figure it quotes is checked. The dashboard, the PDF report and the Excel
workbook all read from the same analysis result.

## Setup (Windows)

1. Install Python 3.11 or newer.
2. In the project folder, create the environment and install the pinned packages:
   ```
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env`. The app works with AI switched off (`AI_ENABLED=false`).
   To use AI, add your own `GEMINI_API_KEY` and `GEMINI_MODEL` (a model available to your key in
   Google AI Studio) and set `AI_ENABLED=true`. Never commit `.env`.

## Usage

- Double-click `run_app.bat` (or run `.venv\Scripts\python -m streamlit run app.py`); the app
  opens at http://localhost:8501.
- **Upload & Profile:** upload a file or load a sample dataset. Check the field mapping (rules
  first, Gemini for the columns the rules could not map, your choice always wins), then press
  **Run analysis**.
- Explore the dashboard pages in the sidebar; filters recalculate every figure.
- **AI Insights:** generate an interpretation within the configured Gemini budget (cached, so
  the same analysis never costs a second call).
- **Reports:** download the executive PDF and the analytical Excel workbook.

## Tests

```
.venv\Scripts\python -m pytest
```
Tests never call the real Gemini API (a fake client is used; `tests/conftest.py` blocks real
calls). Timings: `.venv\Scripts\python scripts\benchmark.py`.

*Architecture, methodology and deployment notes are completed in the final documentation phase.*
