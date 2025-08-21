import os
import json
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

from src.utils.config import load_yaml_config, merge_config_with_args, AppConfig
from src.model.risk import generate_synthetic_assets, compute_risk_fields, map_columns
from src.model.optimization import greedy_select, ilp_select
from src.io.writers import write_csv, write_geojson, write_summary, ensure_dir
from src.security.security import validate_password

PRIMARY = "#7B61FF"      # vibrant purple
ACCENT = "#00D1B2"       # teal
DANGER = "#FF4D4F"       # red
WARNING = "#F59E0B"      # amber
OK = "#10B981"           # emerald
BG = "#0B1221"           # deep navy
SBG = "#121A2B"          # secondary bg
TXT = "#E6EAF2"          # light text


RISK_SVG = """
<svg width="180" height="160" viewBox="0 0 200 180" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="#FBBF24"/>
      <stop offset="100%" stop-color="#F59E0B"/>
    </linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <path d="M100 10 L190 170 H10 Z" fill="url(#g)" stroke="#111827" stroke-width="4" filter="url(#glow)"/>
  <rect x="92" y="60" width="16" height="55" rx="4" fill="#111827"/>
  <circle cx="100" cy="130" r="8" fill="#111827"/>
</svg>
"""


def inject_css():
    st.markdown(
        f"""
        <style>
            :root {{
                --primary: {PRIMARY};
                --accent: {ACCENT};
                --danger: {DANGER};
                --warning: {WARNING};
                --ok: {OK};
                --bg: {BG};
                --sbg: {SBG};
                --txt: {TXT};
                --radius: 14px;
            }}
            html, body, .stApp {{
                background: radial-gradient(1200px 800px at 20% -10%, #1C2A4A 0%, var(--bg) 45%, #060A14 100%) !important;
                color: var(--txt) !important;
            }}
            section[data-testid="stSidebar"] > div {{
                background: linear-gradient(180deg, #0E172C 0%, #0A1222 100%) !important;
                border-right: 1px solid rgba(255,255,255,0.08);
            }}
            /* Ensure all sidebar text is visible */
            section[data-testid="stSidebar"] * {{
                color: var(--txt) !important;
            }}
            /* Improve visibility of inputs/labels */
            label, .stText, .stMarkdown, .stSlider, .stSelectbox, .stNumberInput, .stRadio, .stMultiSelect {{
                color: var(--txt) !important;
            }}
            /* Slider ticks/values */
            div[data-baseweb="slider"] * {{
                color: var(--txt) !important;
            }}
            .kpi-card {{
                padding: 18px 18px;
                border-radius: var(--radius);
                background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
                border: 1px solid rgba(255,255,255,0.10);
                box-shadow: 0 8px 24px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.04);
            }}
            .kpi-label {{
                font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: #B9C2D0;
            }}
            .kpi-value {{
                font-size: 26px; font-weight: 700; color: #F3F6FC; margin-top: 6px;
            }}
            .stDownloadButton button, .stButton button {{
                background: linear-gradient(135deg, var(--primary), #4B7BFF);
                border: 0; color: white; border-radius: 12px; padding: 10px 14px;
                box-shadow: 0 8px 20px rgba(75,123,255,0.35);
            }}
            .stDownloadButton button:hover, .stButton button:hover {{
                filter: brightness(1.08);
            }}
            .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
            .stTabs [data-baseweb="tab"] {{
                background: rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 8px 14px;
                color: var(--txt);
            }}
            .stTabs [aria-selected="true"] {{
                background: linear-gradient(135deg, var(--accent), #00C2A0);
                color: #00151A !important; font-weight: 700;
            }}
            div[data-testid="stMetricValue"] > div {{ color: #F3F6FC !important; }}
            /* Login card */
            .login-card {{
                max-width: 560px; margin: 8vh auto; padding: 28px 28px 24px;
                background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 18px;
                box-shadow: 0 18px 50px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
                text-align: center;
            }}
            .login-title {{ font-size: 28px; font-weight: 800; margin: 6px 0 2px; color: #F3F6FC; }}
            .login-sub {{ font-size: 14px; color: #C9D3E3; margin-bottom: 14px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_risk_image():
    url = os.environ.get("RISK_LOGIN_IMAGE_URL")
    if url:
        st.image(url, use_column_width=True)
        return
    st.markdown(f"<div>{RISK_SVG}</div>", unsafe_allow_html=True)


def gate_access() -> bool:
    # If password not set, allow
    if validate_password(""):
        return True
    # Persist auth across reruns
    if st.session_state.get("auth_ok"):
        return True

    # Full-page login
    col = st.container()
    with col:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        render_risk_image()
        st.markdown("<div class='login-title'>Risk Prioritization Portal</div>", unsafe_allow_html=True)
        st.markdown("<div class='login-sub'>Please enter your access password to continue</div>", unsafe_allow_html=True)
        with st.form("login_form", clear_on_submit=False):
            pwd = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Enter")
            if submitted:
                if validate_password(pwd):
                    st.session_state["auth_ok"] = True
                    st.rerun()
                else:
                    st.error("Invalid password")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


def run_optimizer(df: pd.DataFrame, optimizer: str, budget: float):
    if optimizer == "greedy":
        return greedy_select(df, budget)
    return ilp_select(df, budget)


def kpi_card(label: str, value, emoji: str = "", color: str = PRIMARY):
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left: 5px solid {color}">
            <div class="kpi-label">{emoji} {label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_risk_matrix(df: pd.DataFrame) -> px.imshow:
    pof_bins = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0]
    pof_labels = ["VL", "L", "M", "H", "VH"]
    pof_cat = pd.cut(df["pof"], bins=pof_bins, labels=pof_labels, include_lowest=True, right=True)

    cof_quantiles = df["cof"].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).to_list()
    for i in range(1, len(cof_quantiles)):
        if cof_quantiles[i] <= cof_quantiles[i-1]:
            cof_quantiles[i] = cof_quantiles[i-1] + 1e-6
    cof_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    cof_cat = pd.cut(df["cof"], bins=cof_quantiles, labels=cof_labels, include_lowest=True, right=True)

    mat = (
        df.assign(pof_bin=pof_cat, cof_bin=cof_cat)
          .groupby(["pof_bin", "cof_bin"], dropna=False, observed=False)["risk"]
          .sum()
          .unstack("cof_bin")
          .fillna(0.0)
    )
    fig = px.imshow(
        mat.values,
        labels=dict(x="CoF bin", y="PoF bin", color="Risk sum"),
        x=mat.columns.astype(str),
        y=mat.index.astype(str),
        aspect="auto",
        color_continuous_scale=[
            [0.0, "#0ea5e9"],
            [0.25, "#22d3ee"],
            [0.5, "#34d399"],
            [0.75, "#f59e0b"],
            [1.0, "#ef4444"],
        ],
    )
    fig.update_layout(title="Risk Matrix (PoF vs CoF)", paper_bgcolor=SBG, plot_bgcolor=SBG, font_color=TXT)
    return fig


def simulate_uncertainty(df: pd.DataFrame, budget: float, optimizer: str, num_sims: int = 200, concentration: float = 50.0) -> Dict[str, float]:
    rng = np.random.default_rng(42)
    n = len(df)
    risk_reductions = []

    base_pof = df["pof"].to_numpy(dtype=float)
    base_cof = df["cof"].to_numpy(dtype=float)

    for _ in range(num_sims):
        alpha = np.clip(base_pof * concentration, 1e-3, None)
        beta = np.clip((1.0 - base_pof) * concentration, 1e-3, None)
        pof_sim = rng.beta(alpha, beta)
        cof_sim = np.exp(np.log(base_cof) + rng.normal(0.0, 0.3, size=n))

        df_sim = df.copy()
        df_sim["pof"] = pof_sim
        df_sim["cof"] = cof_sim
        df_sim["risk"] = df_sim["pof"] * df_sim["cof"]
        df_sim["risk_per_cost"] = df_sim["risk"] / df_sim["cost"]

        res = run_optimizer(df_sim, optimizer, budget)
        mask = df_sim.index.isin(res.selected_indices)
        risk_reductions.append(float(df_sim.loc[mask, "risk"].sum()))

    arr = np.array(risk_reductions, dtype=float)
    return {
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def no_deps_globe_component(assets: List[Dict], height: int = 640) -> None:
    assets_json = json.dumps(assets)
    html = (
        "<div id=\"gcontainer\" style=\"width:100%; height:" + str(height) + "px; position:relative;\"></div>\n"
        "<canvas id=\"globe\" style=\"width:100%; height:100%; display:block;\"></canvas>\n"
        "<div id=\"gtooltip\" style=\"position:absolute; top:10px; left:10px; padding:6px 10px; background:rgba(0,0,0,0.6); color:#fff; border-radius:8px; font: 12px/1.2 sans-serif; pointer-events:none; display:none; z-index: 5;\"></div>\n"
        "<script>\n"
        "(function(){\n"
        " const container = document.getElementById('gcontainer');\n"
        " const canvas = document.getElementById('globe');\n"
        " const tooltip = document.getElementById('gtooltip');\n"
        " const ctx = canvas.getContext('2d');\n"
        " const DPR = window.devicePixelRatio || 1;\n"
        " let W = container.clientWidth, H = container.clientHeight;\n"
        " function resize(){ W = container.clientWidth; H = container.clientHeight; canvas.width = W*DPR; canvas.height = H*DPR; canvas.style.width=W+'px'; canvas.style.height=H+'px'; ctx.setTransform(DPR,0,0,DPR,0,0);}\n"
        " resize(); window.addEventListener('resize', resize);\n"
        " const assets = " + assets_json + ";\n"
        " const R = Math.min(W,H)*0.32;\n"
        " let angle = 0;\n"
        " function latLonToXYZ(lat, lon, r){ const phi=(90-lat)*Math.PI/180, the=(lon+angle)*Math.PI/180; const x = -r*Math.sin(phi)*Math.cos(the); const z =  r*Math.sin(phi)*Math.sin(the); const y =  r*Math.cos(phi); return {x,y,z}; }\n"
        " function project(x,y,z){ const d = 3.5; const scale = R*0.9; const f = d/(d - z/(R)); return { sx: W/2 + x/ R * scale * f, sy: H/2 + y/ R * scale * f, f: f, z:z }; }\n"
        " function draw(){ ctx.clearRect(0,0,W,H);\n"
        "  // background gradient\n"
        "  const grd = ctx.createRadialGradient(W*0.2, H*0.1, Math.min(W,H)*0.1, W*0.5, H*0.5, Math.max(W,H)*0.8); grd.addColorStop(0,'#1C2A4A'); grd.addColorStop(1,'#0B1221'); ctx.fillStyle=grd; ctx.fillRect(0,0,W,H);\n"
        "  // sphere outline\n"
        "  ctx.beginPath(); ctx.arc(W/2,H/2,R,0,Math.PI*2); ctx.strokeStyle='rgba(255,255,255,0.15)'; ctx.lineWidth=1; ctx.stroke();\n"
        "  // lat/lon lines\n"
        "  ctx.strokeStyle='rgba(255,255,255,0.06)';\n"
        "  for(let t=-60;t<=60;t+=30){ ctx.beginPath(); for(let l=-180;l<=180;l+=5){ const p=project(...Object.values(latLonToXYZ(t,l,R))); if(l===-180) ctx.moveTo(p.sx,p.sy); else ctx.lineTo(p.sx,p.sy);} ctx.stroke(); }\n"
        "  for(let t=-150;t<=150;t+=30){ ctx.beginPath(); for(let l=-80;l<=80;l+=5){ const p=project(...Object.values(latLonToXYZ(l,t,R))); if(l===-80) ctx.moveTo(p.sx,p.sy); else ctx.lineTo(p.sx,p.sy);} ctx.stroke(); }\n"
        "  // points\n"
        "  screenPts.length = 0;\n"
        "  assets.forEach(a=>{ const rn = Math.max(0, Math.min(1, a.risk_norm || 0.5)); const pos = latLonToXYZ(a.lat,a.lon,R*0.98); if(pos.z < -R*0.2) return; const p = project(pos.x,pos.y,pos.z); const size = 3 + 6*rn; const r = Math.min(255, Math.floor(60 + 195*rn)); const g = Math.max(0, Math.floor(220 - 160*rn)); const b = 70; ctx.beginPath(); ctx.fillStyle = `rgba(${r},${g},${b},0.9)`; ctx.arc(p.sx,p.sy,size,0,Math.PI*2); ctx.fill(); screenPts.push({x:p.sx,y:p.sy,s:size,a:a}); });\n"
        "  angle += 0.004; requestAnimationFrame(draw);\n"
        " }\n"
        " const screenPts = [];\n"
        " canvas.addEventListener('mousemove', (ev)=>{ const rect = canvas.getBoundingClientRect(); const x = (ev.clientX-rect.left); const y = (ev.clientY-rect.top); let best=null, bd=9999; screenPts.forEach(p=>{ const d=(p.x-x)*(p.x-x)+(p.y-y)*(p.y-y); if(d<bd && d < (p.s*p.s*4)){ bd=d; best=p; } }); if(best){ tooltip.style.display='block'; tooltip.style.transform = 'translate('+(x+14)+'px,'+(y+14)+'px)'; tooltip.innerHTML = (best.a.name||'Asset') + '<br/>Risk idx: '+ (best.a.risk_norm||0).toFixed(2); } else { tooltip.style.display='none'; } });\n"
        " draw();\n"
        "})();\n"
        "</script>\n"
    )
    components.html(html, height=height)


def google_earth_globe_component(assets: List[Dict], api_key: Optional[str], height: int = 640) -> None:
    if not api_key:
        st.info("Set GOOGLE_MAPS_API_KEY in your environment to enable the live Google Earth globe.")
        return
    assets_json = json.dumps(assets)
    html = (
        "<div id=\"gmap\" style=\"width:100%; height:" + str(height) + "px;\"></div>\n"
        "<script>\n"
        "window.__ASSETS__ = " + assets_json + ";\n"
        "function initMap() {\n"
        "  const el = document.getElementById('gmap');\n"
        "  const meanLat = __ASSETS__.reduce((a,b)=>a+b.lat,0)/Math.max(1,__ASSETS__.length);\n"
        "  const meanLon = __ASSETS__.reduce((a,b)=>a+b.lon,0)/Math.max(1,__ASSETS__.length);\n"
        "  const map = new google.maps.Map(el, {\n"
        "    center: {lat: meanLat, lng: meanLon},\n"
        "    zoom: 3,\n"
        "    tilt: 67,\n"
        "    heading: 0,\n"
        "    mapTypeId: 'satellite',\n"
        "    gestureHandling: 'greedy'\n"
        "  });\n"
        "  __ASSETS__.forEach(a => {\n"
        "    new google.maps.Marker({position: {lat: a.lat, lng: a.lon}, map, title: a.name});\n"
        "  });\n"
        "  let h = 0;\n"
        "  function animate() { h = (h + 0.2) % 360; map.setHeading(h); requestAnimationFrame(animate); }\n"
        "  animate();\n"
        "}\n"
        "</script>\n"
        "<script async src=\"https://maps.googleapis.com/maps/api/js?key=" + api_key + "&v=weekly&callback=initMap\"></script>\n"
    )
    components.html(html, height=height)


def main():
    st.set_page_config(page_title="Risk Prioritization", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    gate_access()

    cfg = load_yaml_config(os.path.join("configs", "config.yaml"))

    st.sidebar.header("Controls")
    data_source = st.sidebar.radio("Data source", ["Synthetic", "Upload CSV"], index=0)
    optimizer = st.sidebar.selectbox("Optimizer", ["greedy", "ilp"], index=0 if cfg.optimizer == "greedy" else 1)
    budget = st.sidebar.slider("Budget", min_value=100000, max_value=5000000, step=50000, value=int(cfg.budget))

    with st.sidebar.expander("3D Globe settings", expanded=False):
        default_key = os.environ.get("GOOGLE_MAPS_API_KEY", st.session_state.get("GOOGLE_MAPS_API_KEY", ""))
        api_key_input = st.text_input("Google Maps API key", value=default_key, type="password")
        if api_key_input:
            st.session_state["GOOGLE_MAPS_API_KEY"] = api_key_input

    if data_source == "Synthetic":
        num_assets = st.sidebar.slider("Assets (synthetic)", min_value=50, max_value=1000, step=50, value=int(cfg.num_assets))
        seed = st.sidebar.number_input("Random seed", min_value=0, value=int(cfg.random_seed), step=1)
        df = generate_synthetic_assets(num_assets, seed)
    else:
        st.sidebar.markdown("Upload CSV with columns: name, lat, lon, pof, cof, cost (optional: condition_score)")
        uploaded = st.sidebar.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)
        if uploaded is None:
            st.info("Upload a CSV to continue, or switch to Synthetic.")
            st.stop()
        df_raw = pd.read_csv(uploaded)
        st.sidebar.subheader("Column mapping")
        col_map = {
            "name": st.sidebar.text_input("Name column", value=cfg.columns.get("name", "name")),
            "latitude": st.sidebar.text_input("Latitude column", value=cfg.columns.get("latitude", "lat")),
            "longitude": st.sidebar.text_input("Longitude column", value=cfg.columns.get("longitude", "lon")),
            "pof": st.sidebar.text_input("PoF column", value=cfg.columns.get("pof", "pof")),
            "cof": st.sidebar.text_input("CoF column", value=cfg.columns.get("cof", "cof")),
            "cost": st.sidebar.text_input("Cost column", value=cfg.columns.get("cost", "cost")),
            "condition": st.sidebar.text_input("Condition column (optional)", value=cfg.columns.get("condition", "condition_score")),
        }
        cols = map_columns(col_map)
        df = compute_risk_fields(df_raw, cols)
        rename_map = {}
        if cols.latitude != "lat":
            rename_map[cols.latitude] = "lat"
        if cols.longitude != "lon":
            rename_map[cols.longitude] = "lon"
        if cols.cost != "cost":
            rename_map[cols.cost] = "cost"
        if rename_map:
            df = df.rename(columns=rename_map)
        if "condition_score" not in df.columns and cols.condition in df.columns:
            df = df.rename(columns={cols.condition: "condition_score"})
        if "condition_score" not in df.columns:
            df["condition_score"] = np.nan

    with st.sidebar.expander("Filters", expanded=False):
        if "region" in df.columns:
            regions = sorted([x for x in df["region"].dropna().unique()])
            selected_regions = st.multiselect("Region", regions, default=regions)
            if selected_regions:
                df = df[df["region"].isin(selected_regions)]
        if "category" in df.columns:
            cats = sorted([x for x in df["category"].dropna().unique()])
            selected_cats = st.multiselect("Category", cats, default=cats)
            if selected_cats:
                df = df[df["category"].isin(selected_cats)]

    result = run_optimizer(df, optimizer, budget)
    selected_mask = df.index.isin(result.selected_indices)
    df_out = df.copy()
    df_out["selected"] = selected_mask

    total_risk = float(df_out["risk"].sum())
    selected_risk = float(df_out.loc[selected_mask, "risk"].sum())
    risk_reduction = selected_risk
    total_cost_selected = float(df_out.loc[selected_mask, "cost"].sum())

    st.markdown("# ⚡ Risk-Based Asset Prioritization")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Assets", f"{len(df_out):,}", "📦", ACCENT)
    with k2:
        kpi_card("Selected", f"{int(selected_mask.sum()):,}", "✅", OK)
    with k3:
        kpi_card("Risk reduction", f"${risk_reduction:,.0f}", "🛡️", PRIMARY)
    with k4:
        kpi_card("Budget used", f"${total_cost_selected:,.0f}", "💸", WARNING)

    tab_overview, tab_map, tab_analytics, tab_matrix, tab_sim, tab_table, tab_globe, tab_dl = st.tabs([
        "Overview", "Map", "Analytics", "Risk Matrix", "Simulation", "Table", "3D Globe", "Downloads"
    ])

    with tab_map:
        layer = pdk.Layer(
            "ScatterplotLayer",
            df_out,
            get_position="[lon, lat]",
            get_radius=1200,
            get_fill_color="[selected ? 123 : 0, selected ? 97 : 209, selected ? 255 : 178, 160]",
            pickable=True,
            auto_highlight=True,
        )
        view_state = pdk.ViewState(latitude=float(df_out["lat"].mean()), longitude=float(df_out["lon"].mean()), zoom=4)
        r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}\nRisk: ${risk}\nCost: ${cost}"})
        st.pydeck_chart(r)

    with tab_analytics:
        left, right = st.columns(2)
        with left:
            topn = df_out.nlargest(20, "risk")
            fig_bar = px.bar(topn, x="name", y="risk", color="selected", title="Top 20 Risk by Asset",
                             color_discrete_map={True: PRIMARY, False: "#334155"})
            fig_bar.update_layout(xaxis_tickangle=-40, height=450, paper_bgcolor=SBG, plot_bgcolor=SBG, font_color=TXT)
            st.plotly_chart(fig_bar, use_container_width=True)
        with right:
            fig_scatter = px.scatter(df_out, x="cost", y="risk", color="selected", size="risk_per_cost", hover_name="name",
                                     title="Cost vs Risk", color_discrete_map={True: ACCENT, False: "#64748b"})
            fig_scatter.update_layout(paper_bgcolor=SBG, plot_bgcolor=SBG, font_color=TXT)
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_matrix:
        st.plotly_chart(build_risk_matrix(df_out), use_container_width=True)

    with tab_sim:
        with st.expander("Uncertainty simulation (Monte Carlo)"):
            run_sim = st.checkbox("Run simulation", value=False)
            if run_sim:
                sims = st.slider("Simulations", min_value=50, max_value=300, step=50, value=200)
                conc = st.slider("PoF concentration (higher = less variance)", min_value=10, max_value=200, step=10, value=50)
                stats = simulate_uncertainty(df_out, budget, optimizer, num_sims=sims, concentration=float(conc))
                c1, c2, c3 = st.columns(3)
                c1.metric("P5 risk reduction", f"${stats['p05']:,.0f}")
                c2.metric("Median", f"${stats['p50']:,.0f}")
                c3.metric("P95", f"${stats['p95']:,.0f}")

    with tab_table:
        st.dataframe(
            df_out[["name", "condition_score", "pof", "cof", "cost", "risk", "risk_per_cost", "selected"]]
                .sort_values("risk_per_cost", ascending=False),
            use_container_width=True,
        )

    with tab_globe:
        # Offline-safe Canvas Globe (no external dependencies)
        total_risk = df_out["risk"].astype(float)
        risk_norm = (total_risk - total_risk.min()) / (total_risk.max() - total_risk.min() + 1e-9)
        assets = (
            df_out[["name", "lat", "lon"]]
            .assign(risk_norm=risk_norm.values)
            .to_dict(orient="records")
        )
        st.markdown("#### 3D Globe (No-deps)")
        no_deps_globe_component(assets, height=520)
        st.markdown("#### Google Earth (Optional)")
        google_api_key = st.session_state.get("GOOGLE_MAPS_API_KEY", os.environ.get("GOOGLE_MAPS_API_KEY"))
        google_earth_globe_component(assets, google_api_key, height=420)

    with tab_dl:
        ensure_dir("outputs")
        csv_path = write_csv(df_out, "outputs")
        geojson_path = write_geojson(df_out, "outputs")
        summary = {
            "optimizer": optimizer,
            "budget": budget,
            "total_assets": int(len(df_out)),
            "selected_assets": int(selected_mask.sum()),
            "total_cost_selected": total_cost_selected,
            "total_risk": float(df_out["risk"].sum()),
            "selected_risk": float(df_out.loc[selected_mask, "risk"].sum()),
            "risk_reduction": float(df_out.loc[selected_mask, "risk"].sum()),
        }
        summary_path = write_summary(summary, "outputs")

        with open(csv_path, "rb") as f:
            st.download_button("Download CSV", data=f, file_name=os.path.basename(csv_path), mime="text/csv")
        with open(geojson_path, "rb") as f:
            st.download_button("Download GeoJSON", data=f, file_name=os.path.basename(geojson_path), mime="application/geo+json")
        with open(summary_path, "rb") as f:
            st.download_button("Download Summary JSON", data=f, file_name=os.path.basename(summary_path), mime="application/json")


if __name__ == "__main__":
    main()
