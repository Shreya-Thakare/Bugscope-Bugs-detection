# BugScope — Software Defect Prediction Command Center

AI-assisted defect risk scoring from static code metrics and process signals, with a REST API and a multi-page command-center dashboard.

> **Note:** BugScope provides **statistical estimates** for engineering prioritization. It is not a substitute for code review, testing, or professional judgment.

## Features

- **ML models** — Gradient Boosting + Random Forest (best model selected by F1)
- **REST API** — `GET /health`, `/metrics`, `/eda`, `/system` · `POST /predict`, `/predict/batch` (port 8000)
- **Command center** — Overview · Projects · Modules · Predict · Insights · Model · Alerts · Activity
- **Local explanations** — Factor contributions for every score (LOC, CC, churn, coverage, experience)
- **Batch scoring** — Paste CSV rows on Predict; export results
- **Prediction history** — Last 50 scores stored in the browser; export CSV
- **Global search** — Live hits for modules, projects, and alerts
- **Exports** — Modules, alerts, history, and batch results as CSV
- **Alert triage** — Acknowledge / resolve / reopen (client-side state)
- **Quality gate checks** — Pass/fail against portfolio thresholds on each score
- **CI / webhook ready** — Documented curl + batch endpoint for pipelines
- **Process + static signals** — Complexity, Halstead, churn, review/test coverage, team experience
- **SQL schema** — Projects, modules, predictions, alerts

## Tech stack

| Layer | Stack |
|-------|--------|
| ML | Python, scikit-learn, pandas, joblib |
| API | Pure Python HTTP server (optional FastAPI deps listed) |
| Frontend | Multi-page HTML/CSS/JS command center |
| Data | CSV + MySQL-compatible `schema.sql` |

## Project structure

```text
bugscope/
├── frontend/              # Command center UI
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── backend/
│   ├── run_api.py         # Recommended API entrypoint
│   └── requirements.txt
├── ml/
│   ├── train_model.py
│   ├── software_metrics.csv
│   ├── models/            # Trained artifacts
│   └── evaluation/        # metrics.json, eda_stats.json
├── database/
│   ├── schema.sql
│   └── software_metrics.csv
├── LICENSE
├── .gitignore
└── README.md
```

## Quick start

```bash
git clone <your-repo-url> bugscope
cd bugscope

# API deps (minimal runtime uses stdlib + optional joblib/sklearn)
pip install -r backend/requirements.txt

# Optional: retrain
pip install -r ml/requirements.txt
python ml/train_model.py

# 1. Start API
cd backend && python run_api.py
# → http://localhost:8000

# 2. Open dashboard (new terminal, from repo root)
python -m http.server 5500 --directory frontend
# → http://localhost:5500
```

The UI works offline with a local heuristic; with the API running it uses the trained model.

## API example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loc": 2400,
    "cyclomatic_complexity": 32,
    "code_churn": 48,
    "review_coverage": 0.4,
    "test_coverage": 0.35,
    "team_experience": 0.45,
    "language": "Java",
    "module_type": "Core"
  }'
```

Example response:

```json
{
  "defective": true,
  "defect_probability": 0.71,
  "probability": 71.0,
  "risk_band": "High",
  "recommendation": "Prioritize review, tests, and refactoring",
  "mode": "ml"
}
```

## Command center pages

| Page | Purpose |
|------|---------|
| **Overview** | KPIs, risk mix, pipeline |
| **Modules** | Sample module table |
| **Predict** | Interactive scoring form + risk band |
| **Insights** | Dataset profile + feature importance |
| **Model** | Per-model accuracy / F1 / AUC |
| **Alerts** | Illustrative high-risk triage list |

## System data

Demo portfolio data lives in:

- `frontend/data/system_data.json` — projects, modules, alerts, activity, quality gates, daily volumes
- `database/seed/demo_seed.sql` — SQL seed for projects / modules / predictions / alerts
- `GET /system` on the API returns the same JSON when the server is running

## Database

See `database/schema.sql` for `projects`, `modules`, `predictions`, `alerts`.

## License

MIT — see [LICENSE](LICENSE).

## Roadmap ideas

- Persist predictions to SQL (history is browser-local for now)
- Auth for quality leads
- Server-side alert ownership and Slack/email notifications
- True SHAP values from the trained model (rule-based factors ship today)
- Git / VCS metric connectors
