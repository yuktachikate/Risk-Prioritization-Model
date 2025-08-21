## Risk-Based Asset Prioritization – Ready-to-Run Toolkit

This project helps utilities decide which grid assets to repair when budgets are limited using a transparent, risk-based methodology. It computes risk (Probability of Failure × Consequence of Failure), optimizes selections under a budget, and produces outputs for Power BI and ArcGIS. A Streamlit dashboard provides interactive visuals for business stakeholders.

### Key features
- Risk scoring: Risk = PoF × CoF; Risk-per-Cost for fair comparisons
- Optimizers: Fast greedy and exact ILP (0/1 knapsack via PuLP)
- Outputs: CSV and GeoJSON, ready for Power BI and ArcGIS
- Interactive dashboard: Streamlit app with maps, KPIs, and filters
- Security-first defaults: Local execution, env-based secrets, and least-privilege guidance to align with zero-trust principles

---

### 1) Quick start

Prerequisites:
- Python 3.10+
- macOS or Windows

Setup:
```bash
cd "Risk Prioritization Model"
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the model (synthetic data):
```bash
python src/asset_risk_prioritization.py --budget 1_000_000 --optimizer greedy
```
Outputs will be written to `outputs/`:
- `asset_prioritization_results.csv`
- `asset_prioritization_results.geojson`
- `summary.json`

Run the interactive dashboard:
```bash
# Optional: set a simple password gate (recommended)
export STREAMLIT_PASSWORD="change-me"  # Windows PowerShell: $env:STREAMLIT_PASSWORD="change-me"
streamlit run src/app/streamlit_app.py
```

---

### 2) Using in Power BI
1. Open Power BI Desktop → Get Data → Text/CSV → select `outputs/asset_prioritization_results.csv`.
2. Suggested visuals:
   - Bar chart: Asset `name` vs `risk` or `risk_per_cost`.
   - Scatter: X=`cost`, Y=`risk`, Size=`risk_per_cost`, Color=`selected`.
   - Slicer: `selected` = True to show chosen assets.
   - KPI/Metrics: sum of `cost_selected`, sum of `risk_selected`, `risk_reduced` from `summary.json`.
3. Refresh: Re-run the Python model to update the CSV, then refresh the report.

Optional theme (JSON) is included at `powerbi/theme.json`.

---

### 3) Using in ArcGIS Pro or ArcGIS Online
1. Add layer from file → choose `outputs/asset_prioritization_results.geojson`.
2. Symbology:
   - Graduated color by `risk`.
   - Unique value by `selected` to highlight chosen assets.
3. Configure popups to show: `condition_score`, `pof`, `cof`, `cost`, `risk`, `risk_per_cost`, `selected`.

---

### 4) Configuration
Edit `configs/config.yaml`:
- `budget`: Total available spend.
- `optimizer`: `greedy` or `ilp`.
- `random_seed`: Reproducibility for synthetic data.

Override via CLI flags:
```bash
python src/asset_risk_prioritization.py --budget 750000 --optimizer ilp --num-assets 250
```

Use your own asset data:
```bash
python src/asset_risk_prioritization.py --input-csv path/to/your_assets.csv --lat-col latitude --lon-col longitude --budget 900000
```
Required columns if providing your own CSV:
- `name`, `lat`, `lon`, `pof`, `cof`, `cost`
Optional:
- `condition_score`, `category`, `region`

---

### 5) Dashboard highlights (Streamlit)
- Budget slider and optimizer switch (greedy/ILP)
- KPIs: total risk, selected risk, risk reduction, budget used
- Interactive map (pydeck) with hover tooltips
- Bar and scatter visuals (Plotly) with filters
- Download buttons for CSV/GeoJSON

Run:
```bash
streamlit run src/app/streamlit_app.py
```

---

### 6) Zero-trust aligned practices
- Run locally by default; no external services required.
- Set `STREAMLIT_PASSWORD` to gate access for business demos.
- Keep raw data minimal: avoid PII and sensitive fields.
- Principle of least privilege: read-only folders for business users, separate write permissions for data engineers.
- Reproducibility & integrity: `summary.json` captures run metadata; outputs are deterministic with `random_seed`.

For enterprise SSO, front the dashboard behind your IdP (Azure AD, Okta) or publish only the CSV/GeoJSON to controlled BI/GIS workspaces.

---

### 7) Project structure
```
Risk Prioritization Model/
  configs/
    config.yaml
  outputs/  # generated
  powerbi/
    theme.json
  src/
    app/streamlit_app.py
    asset_risk_prioritization.py
    io/writers.py
    model/optimization.py
    model/risk.py
    utils/config.py
    security/security.py
  requirements.txt
  README.md
  .env.example
```

---

### 8) Troubleshooting
- If ILP is slow for large datasets, use `--optimizer greedy` or reduce `--num-assets`.
- If pydeck map fails, ensure internet access for base maps or switch to the Folium export provided by the app.
- For Power BI and ArcGIS, ensure the output files exist (re-run the model if missing).

---

### 9) License
Apache-2.0
