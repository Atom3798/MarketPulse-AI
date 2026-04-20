import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error

def compute_rmse(y_true, y_pred): return float(root_mean_squared_error(y_true, y_pred))

st.set_page_config(
    page_title="MarketPulse AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    box-sizing: border-box;
}

/* ── App shell ── */
[data-testid="stAppViewContainer"] { background: #0a0a1f; }
[data-testid="stMain"] { background: transparent; }
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, .stDeployButton, header { visibility: hidden !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0d25 !important;
    border-right: 1px solid rgba(99,102,241,0.18) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

[data-testid="stSidebar"] label {
    color: #64748b !important;
    font-size: 10.5px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.06) !important;
    border: 1.5px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 14px !important;
    padding: 10px 12px !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    outline: none !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background: rgba(255,255,255,0.06) !important;
    border: 1.5px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Button — reset then rebuild ── */
.stButton > button {
    all: unset !important;
    display: block !important;
    width: 100% !important;
    text-align: center !important;
    padding: 14px 16px !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #ffffff !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    letter-spacing: 0.3px !important;
    white-space: nowrap !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    gap: 0 !important;
    padding-bottom: 0 !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 12px 20px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    margin-bottom: -1px !important;
    transition: color 0.15s !important;
}
button[data-baseweb="tab"]:hover { color: #94a3b8 !important; background: rgba(255,255,255,0.03) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #a5b4fc !important;
    border-bottom-color: #6366f1 !important;
    background: rgba(99,102,241,0.07) !important;
}
[data-testid="stTabsContent"] { padding-top: 24px !important; }

/* ── Chart container ── */
[data-testid="stPlotlyChart"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── Expander ── */
details {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}
details summary {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 14px 18px !important;
}
details summary:hover { color: #cbd5e1 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e2050; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

/* ── Animations ── */
@keyframes pulse-dot { 0%,100%{opacity:1; transform:scale(1)} 50%{opacity:.4; transform:scale(0.85)} }
@keyframes slide-up { from{opacity:0; transform:translateY(12px)} to{opacity:1; transform:translateY(0)} }
.slide-up { animation: slide-up 0.35s ease forwards; }
</style>
""", unsafe_allow_html=True)

# ── Domain logic ──────────────────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - (100 / (1 + gain / loss))

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"]     = df["price"].pct_change()
    df["ma5"]        = df["price"].rolling(5).mean()
    df["ma20"]       = df["price"].rolling(20).mean()
    df["volatility"] = df["return"].rolling(10).std()
    df["rsi"]        = compute_rsi(df["price"])
    return df.dropna()

FEATURES = ["ma5", "ma20", "volatility", "rsi"]

class MarketModel:
    def __init__(self, n: int = 200):
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror", n_estimators=n,
            learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42,
        )
        self._names: list[str] = []

    def train(self, X, y):
        self._names = list(X.columns)
        self.model.fit(X, y)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def rmse(self, X, y) -> float:
        return compute_rmse(y, self.predict(X))

    def importance(self) -> dict:
        return dict(zip(self._names, self.model.feature_importances_))

@st.cache_data(ttl=300)
def fetch(ticker: str, period: str):
    t = yf.Ticker(ticker)
    return t.history(period=period), t.info

# ── UI helpers ────────────────────────────────────────────────────────────────────
CARD_COLORS = {
    "price":  ("#6366f1", "99,102,241"),
    "high":   ("#10b981", "16,185,129"),
    "low":    ("#f43f5e", "244,63,94"),
    "cap":    ("#0ea5e9", "14,165,233"),
    "rmse":   ("#a855f7", "168,85,247"),
}

def stat_card(label: str, value: str, delta: str = "",
              positive: bool = True, key: str = "price") -> str:
    hex_c, rgb = CARD_COLORS.get(key, CARD_COLORS["price"])
    delta_col = "#22c55e" if positive else "#f43f5e"
    arrow     = "▲" if positive else "▼"
    delta_html = (
        f'<div style="margin-top:8px;color:{delta_col};font-size:12px;font-weight:600;">'
        f'{arrow}&thinsp;{delta}</div>'
    ) if delta else ""
    return f"""
    <div class="slide-up" style="
        background: linear-gradient(135deg, rgba({rgb},0.1) 0%, rgba(13,13,37,0.95) 70%);
        border: 1px solid rgba({rgb},0.25);
        border-radius: 16px; padding: 18px 20px; position: relative; overflow: hidden;
    ">
        <div style="position:absolute;top:0;left:0;width:100%;height:2px;
            background:linear-gradient(90deg,{hex_c},transparent);"></div>
        <div style="color:rgba({rgb},0.75);font-size:10px;font-weight:700;
            letter-spacing:1.2px;text-transform:uppercase;">{label}</div>
        <div style="color:#f1f5f9;font-size:22px;font-weight:800;margin-top:10px;
            font-variant-numeric:tabular-nums;line-height:1.15;">{value}</div>
        {delta_html}
    </div>"""

def section_title(title: str, sub: str = "") -> None:
    sub_html = f'<p style="color:#64748b;font-size:13px;margin:4px 0 0;">{sub}</p>' if sub else ""
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:18px;">
        <div style="width:3px;min-height:46px;border-radius:2px;flex-shrink:0;margin-top:2px;
            background:linear-gradient(180deg,#6366f1,#a855f7);"></div>
        <div>
            <h3 style="margin:0;color:#e2e8f0;font-size:17px;font-weight:700;
                letter-spacing:-0.3px;">{title}</h3>
            {sub_html}
        </div>
    </div>""", unsafe_allow_html=True)

# ── Plotly base theme ─────────────────────────────────────────────────────────────
def chart_layout(**extra):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d0d25",
        font=dict(color="#64748b", family="Inter, sans-serif", size=11),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)", zeroline=False,
            tickfont=dict(color="#475569", size=10),
            linecolor="rgba(255,255,255,0.06)", showline=True, showgrid=True,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)", zeroline=False,
            tickfont=dict(color="#475569", size=10),
            linecolor="rgba(255,255,255,0.06)", showline=True, showgrid=True,
        ),
        legend=dict(
            bgcolor="rgba(13,13,37,0.9)", bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1, font=dict(color="#94a3b8", size=11),
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(13,13,40,0.97)", bordercolor="rgba(99,102,241,0.35)",
            font=dict(color="#f1f5f9", size=12, family="Inter"),
        ),
        margin=dict(l=12, r=12, t=52, b=12),
    )
    base.update(extra)
    return base

# ── Sidebar ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header with rainbow accent bar at top
    st.markdown("""
    <div style="position:relative;padding:26px 20px 22px;
        border-bottom:1px solid rgba(99,102,241,0.14);">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
            background:linear-gradient(90deg,#6366f1,#8b5cf6,#a855f7,#ec4899);"></div>
        <div style="display:flex;align-items:center;gap:12px;margin-top:4px;">
            <div style="width:40px;height:40px;border-radius:11px;flex-shrink:0;
                background:linear-gradient(135deg,#6366f1,#a855f7);
                display:flex;align-items:center;justify-content:center;font-size:19px;
                box-shadow:0 4px 16px rgba(99,102,241,0.45);">📈</div>
            <div>
                <div style="color:#f1f5f9;font-size:16px;font-weight:800;
                    letter-spacing:-0.4px;">MarketPulse</div>
                <div style="color:#475569;font-size:11px;margin-top:2px;">
                    AI Market Analysis</div>
            </div>
        </div>
    </div>
    <div style="height:12px;"></div>
    """, unsafe_allow_html=True)

    ticker_raw = st.text_input("Ticker Symbol", value="AAPL",
                               placeholder="AAPL · MSFT · TSLA · NVDA")
    ticker = ticker_raw.upper().strip()

    period_map = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    period = period_map[st.selectbox("Time Period", list(period_map.keys()), index=1)]

    n_est = st.slider("Model Estimators", 50, 500, 200, 50,
                      help="More trees = potentially better fit, but slower")

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    run = st.button("⚡ Run Analysis")

    st.markdown("""
    <div style="margin-top:20px;padding:14px 16px;
        background:rgba(99,102,241,0.07);border:1px solid rgba(99,102,241,0.16);
        border-radius:12px;">
        <div style="color:#818cf8;font-size:10px;font-weight:800;letter-spacing:1.1px;
            text-transform:uppercase;margin-bottom:8px;">Pipeline</div>
        <div style="color:#94a3b8;font-size:12px;line-height:1.7;">
            Yahoo Finance&nbsp;→&nbsp;MA5 / MA20 / RSI / Volatility&nbsp;→&nbsp;XGBoost Regressor
        </div>
    </div>
    <div style="margin-top:10px;padding:10px 14px;
        background:rgba(244,63,94,0.06);border:1px solid rgba(244,63,94,0.15);
        border-radius:10px;">
        <div style="color:#fb7185;font-size:11px;font-weight:600;">
            ⚠ Educational use only — not financial advice.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Page hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 6px;">
    <div style="display:inline-flex;align-items:center;gap:8px;
        background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.22);
        border-radius:24px;padding:6px 16px;margin-bottom:18px;">
        <span style="display:inline-block;width:7px;height:7px;border-radius:50%;
            background:#22c55e;animation:pulse-dot 2s ease-in-out infinite;"></span>
        <span style="color:#a5b4fc;font-size:11.5px;font-weight:700;
            letter-spacing:1.2px;">LIVE MARKET DATA</span>
    </div>
    <h1 style="margin:0;font-size:clamp(32px,4vw,48px);font-weight:900;
        letter-spacing:-2px;line-height:1.05;
        background:linear-gradient(135deg,#f1f5f9 0%,#a5b4fc 50%,#c084fc 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        background-clip:text;">MarketPulse AI</h1>
    <p style="color:#94a3b8;font-size:15px;margin:12px 0 0;font-weight:400;line-height:1.6;
        max-width:560px;">
        AI-powered equity analysis — price prediction, technical signals, and model insights
        all in one dashboard.
    </p>
</div>
<div style="height:1px;
    background:linear-gradient(90deg,rgba(99,102,241,0.4),rgba(168,85,247,0.3),transparent);
    margin:26px 0 30px;"></div>
""", unsafe_allow_html=True)

# ── Empty state ───────────────────────────────────────────────────────────────────
if not run:
    pills = "".join(
        f'<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.22);'
        f'border-radius:8px;padding:8px 18px;color:#a5b4fc;font-size:13px;font-weight:600;'
        f'display:inline-block;margin:5px;letter-spacing:0.3px;">{t}</span>'
        for t in ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN", "META", "BRK-B"]
    )
    st.markdown(f"""
    <div style="text-align:center;padding:80px 32px;
        border:1px solid rgba(255,255,255,0.06);border-radius:24px;
        background:rgba(255,255,255,0.012);margin-top:4px;">
        <div style="font-size:64px;line-height:1;margin-bottom:20px;">📊</div>
        <h3 style="color:#e2e8f0;margin:0 0 10px;font-size:24px;font-weight:800;
            letter-spacing:-0.5px;">Ready to Analyze</h3>
        <p style="color:#94a3b8;font-size:14px;max-width:400px;margin:0 auto 26px;
            line-height:1.7;">
            Choose a ticker in the sidebar and hit
            <strong style="color:#a5b4fc;font-weight:700;">Run Analysis</strong>
            to generate AI-powered market insights.
        </p>
        <div style="display:flex;justify-content:center;flex-wrap:wrap;
            max-width:480px;margin:0 auto;">{pills}</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Data fetch ────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker}…"):
    try:
        hist, info = fetch(ticker, period)
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
        st.stop()

if hist.empty:
    st.error(f"No data for **{ticker}**. Double-check the symbol.")
    st.stop()

# Build tz-naive copies for alignment
hist_tz = hist.copy()
hist_tz.index = pd.to_datetime(hist_tz.index).tz_localize(None)

raw = hist_tz[["Close"]].rename(columns={"Close": "price"})

with st.spinner("Training model…"):
    df        = engineer_features(raw)
    X, y      = df[FEATURES], df["price"]
    model     = MarketModel(n_est)
    model.train(X, y)
    y_pred    = model.predict(X)
    rmse      = model.rmse(X, y)

# ── Derived values ────────────────────────────────────────────────────────────────
name        = info.get("longName", ticker)
cur         = float(hist["Close"].iloc[-1])
prev        = float(info.get("previousClose", hist["Close"].iloc[-2]))
chg         = cur - prev
chg_pct     = chg / prev * 100
up          = chg >= 0
chg_col     = "#22c55e" if up else "#f43f5e"
chg_arrow   = "▲" if up else "▼"

hi  = info.get("fiftyTwoWeekHigh")
lo  = info.get("fiftyTwoWeekLow")
cap = info.get("marketCap", 0) or 0
cap_str = (f"${cap/1e12:.2f}T" if cap >= 1e12
           else f"${cap/1e9:.2f}B" if cap >= 1e9
           else f"${cap/1e6:.1f}M")
sector = info.get("sector", "")

# ── Stock header ──────────────────────────────────────────────────────────────────
sector_html = (
    f'<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.22);'
    f'border-radius:6px;padding:3px 12px;font-size:11px;color:#a5b4fc;font-weight:600;">'
    f'{sector}</span>'
) if sector else ""

st.markdown(f"""
<div class="slide-up" style="margin-bottom:26px;">
    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:6px;">
        <h2 style="margin:0;color:#f1f5f9;font-size:24px;font-weight:800;
            letter-spacing:-0.5px;">{name}</h2>
        <span style="background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.1);
            border-radius:6px;padding:3px 10px;font-size:12px;color:#94a3b8;
            font-weight:600;">{ticker}</span>
        {sector_html}
    </div>
    <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;">
        <span style="color:#f1f5f9;font-size:38px;font-weight:900;
            font-variant-numeric:tabular-nums;letter-spacing:-1px;">${cur:.2f}</span>
        <span style="color:{chg_col};font-size:16px;font-weight:700;">
            {chg_arrow}&thinsp;${abs(chg):.2f} ({abs(chg_pct):.2f}%)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stat cards ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, "Current Price", f"${cur:.2f}",
     f"${abs(chg):.2f} ({abs(chg_pct):.2f}%)", up, "price"),
    (c2, "52W High",  f"${hi:.2f}"  if hi  else "N/A", "", True,  "high"),
    (c3, "52W Low",   f"${lo:.2f}"  if lo  else "N/A", "", False, "low"),
    (c4, "Market Cap", cap_str,     "", True,  "cap"),
    (c5, "Model RMSE", f"${rmse:.4f}", "", True, "rmse"),
]
for col, label, value, delta, pos, key in cards:
    with col:
        st.markdown(stat_card(label, value, delta, pos, key), unsafe_allow_html=True)

st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────────
tab_price, tab_indicators, tab_model = st.tabs([
    "📊  Price Analysis",
    "📉  Technical Indicators",
    "🤖  Model Analysis",
])

# ━━ Tab 1: Price ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_price:
    section_title("Candlestick + AI Prediction",
                  f"OHLC price action with XGBoost overlay · {period} period")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.015,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=hist_tz.index,
        open=hist_tz["Open"], high=hist_tz["High"],
        low=hist_tz["Low"],   close=hist_tz["Close"],
        name="OHLC",
        increasing=dict(line=dict(color="#22c55e", width=1), fillcolor="#22c55e"),
        decreasing=dict(line=dict(color="#f43f5e", width=1), fillcolor="#f43f5e"),
    ), row=1, col=1)

    # Prediction band
    fig.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=list(y_pred + rmse) + list((y_pred - rmse)[::-1]),
        fill="toself", fillcolor="rgba(251,191,36,0.09)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=True, name="±RMSE Band",
    ), row=1, col=1)

    # MA lines
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ma20"], mode="lines", name="MA 20",
        line=dict(color="#a5b4fc", width=1.4),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ma5"], mode="lines", name="MA 5",
        line=dict(color="#6ee7b7", width=1.2),
    ), row=1, col=1)

    # AI prediction
    fig.add_trace(go.Scatter(
        x=df.index, y=y_pred, mode="lines", name="AI Prediction",
        line=dict(color="#fbbf24", width=2.5, dash="dash"),
    ), row=1, col=1)

    # Volume bars (green/red by candle direction)
    vol_colors = [
        "#22c55e" if c >= o else "#f43f5e"
        for c, o in zip(hist_tz["Close"], hist_tz["Open"])
    ]
    fig.add_trace(go.Bar(
        x=hist_tz.index, y=hist_tz["Volume"],
        marker=dict(color=vol_colors, opacity=0.55, line=dict(width=0)),
        showlegend=False, name="Volume",
    ), row=2, col=1)

    cl = chart_layout(height=560)
    cl["xaxis2"] = dict(**cl["xaxis"], title="")
    cl["yaxis2"] = dict(**cl["yaxis"], title="Volume")
    cl["yaxis"]["title"] = "Price (USD)"
    cl["xaxis"]["rangeslider"] = dict(visible=False)
    fig.update_layout(**cl)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", row=1, col=1)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", row=2, col=1)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)")

    st.plotly_chart(fig, use_container_width=True)

    # Stats row
    accuracy_pct = max(0.0, 100 - (rmse / df["price"].mean() * 100))
    s1, s2, s3, s4 = st.columns(4)
    for col, label, val in [
        (s1, "Data Points",     f"{len(df):,}"),
        (s2, "Price Range",     f"${df['price'].min():.2f} – ${df['price'].max():.2f}"),
        (s3, "Avg Daily Move",  f"{df['return'].abs().mean()*100:.2f}%"),
        (s4, "RMSE / Mean",     f"{rmse/df['price'].mean()*100:.2f}%"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
                border-radius:12px;padding:14px 18px;text-align:center;margin-top:4px;">
                <div style="color:#64748b;font-size:10px;font-weight:700;letter-spacing:1px;
                    text-transform:uppercase;">{label}</div>
                <div style="color:#e2e8f0;font-size:18px;font-weight:700;margin-top:7px;
                    font-variant-numeric:tabular-nums;">{val}</div>
            </div>""", unsafe_allow_html=True)

# ━━ Tab 2: Technical Indicators ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_indicators:
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        section_title("RSI (14-Day)", "Relative Strength Index — overbought > 70, oversold < 30")

        fig_rsi = go.Figure()
        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(244,63,94,0.06)", line_width=0)
        fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="rgba(34,197,94,0.06)",  line_width=0)
        fig_rsi.add_hline(y=70, line=dict(color="rgba(244,63,94,0.5)", dash="dash", width=1.2))
        fig_rsi.add_hline(y=30, line=dict(color="rgba(34,197,94,0.5)",  dash="dash", width=1.2))
        fig_rsi.add_trace(go.Scatter(
            x=df.index, y=df["rsi"], mode="lines", name="RSI",
            line=dict(color="#a5b4fc", width=2),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.07)",
        ))
        # Overbought/oversold annotations
        fig_rsi.add_annotation(x=df.index[-1], y=72, text="Overbought",
            showarrow=False, font=dict(color="#fb7185", size=10), xanchor="right")
        fig_rsi.add_annotation(x=df.index[-1], y=28, text="Oversold",
            showarrow=False, font=dict(color="#4ade80", size=10), xanchor="right")

        fig_rsi.update_layout(**chart_layout(height=320,
            margin=dict(l=12,r=12,t=20,b=12)),
            yaxis_title="RSI", yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col_r:
        section_title("Volatility", "10-day rolling std of daily returns")

        fig_vol_ind = go.Figure()
        fig_vol_ind.add_trace(go.Scatter(
            x=df.index, y=df["volatility"] * 100, mode="lines", name="Volatility",
            line=dict(color="#fbbf24", width=2),
            fill="tozeroy", fillcolor="rgba(251,191,36,0.07)",
        ))
        fig_vol_ind.update_layout(**chart_layout(height=320,
            margin=dict(l=12,r=12,t=20,b=12)),
            yaxis_title="Volatility (%)")
        st.plotly_chart(fig_vol_ind, use_container_width=True)

    # MA comparison
    section_title("Moving Averages", "MA5 vs MA20 crossovers — golden cross & death cross signals")
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=hist_tz.index, y=hist_tz["Close"], mode="lines", name="Price",
        line=dict(color="rgba(148,163,184,0.4)", width=1.2),
    ))
    fig_ma.add_trace(go.Scatter(
        x=df.index, y=df["ma5"], mode="lines", name="MA 5",
        line=dict(color="#6ee7b7", width=2),
    ))
    fig_ma.add_trace(go.Scatter(
        x=df.index, y=df["ma20"], mode="lines", name="MA 20",
        line=dict(color="#a5b4fc", width=2),
    ))
    fig_ma.update_layout(**chart_layout(height=300), yaxis_title="Price (USD)")
    st.plotly_chart(fig_ma, use_container_width=True)

# ━━ Tab 3: Model Analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_model:
    col_imp, col_scatter = st.columns(2, gap="large")

    with col_imp:
        section_title("Feature Importance", "Which signals drive the XGBoost model most")
        fi       = model.importance()
        fi_df    = (pd.DataFrame({"Feature": list(fi.keys()),
                                  "Importance": list(fi.values())})
                      .sort_values("Importance", ascending=True))
        bar_colors = ["#6366f1","#8b5cf6","#fbbf24","#22c55e"]
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker=dict(color=bar_colors[:len(fi_df)],
                        line=dict(width=0),
                        cornerradius=4),
            text=fi_df["Importance"].round(3).astype(str),
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11),
        ))
        fig_fi.update_layout(**chart_layout(height=340,
            margin=dict(l=12,r=48,t=20,b=12)),
            xaxis_title="Importance Score",
            bargap=0.3)
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_scatter:
        section_title("Actual vs. Predicted", "How closely the model tracks real prices")
        fig_scatter = go.Figure()
        # perfect-fit reference line
        mn, mx = float(y.min()), float(y.max())
        fig_scatter.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color="rgba(255,255,255,0.15)", dash="dash", width=1.5),
            showlegend=False, name="Perfect Fit",
        ))
        fig_scatter.add_trace(go.Scatter(
            x=y, y=y_pred, mode="markers", name="Predictions",
            marker=dict(
                color=y_pred - y.values,
                colorscale=[[0,"#f43f5e"],[0.5,"#6366f1"],[1,"#22c55e"]],
                size=4, opacity=0.7,
                colorbar=dict(title="Error", tickfont=dict(color="#64748b", size=10),
                              title_font=dict(color="#64748b")),
            ),
            hovertemplate="Actual: $%{x:.2f}<br>Predicted: $%{y:.2f}<extra></extra>",
        ))
        fig_scatter.update_layout(**chart_layout(height=340,
            margin=dict(l=12,r=12,t=20,b=12)),
            xaxis_title="Actual Price (USD)",
            yaxis_title="Predicted Price (USD)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Model stats row
    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
    section_title("Model Statistics")
    ms1, ms2, ms3, ms4 = st.columns(4)
    mae = float(np.mean(np.abs(y_pred - y.values)))
    r2  = float(1 - np.sum((y.values - y_pred)**2) / np.sum((y.values - y.mean())**2))
    for col, label, val, key in [
        (ms1, "RMSE",    f"${rmse:.4f}",          "rmse"),
        (ms2, "MAE",     f"${mae:.4f}",            "price"),
        (ms3, "R² Score", f"{r2:.4f}",             "cap"),
        (ms4, "Estimators", str(n_est),            "high"),
    ]:
        with col:
            st.markdown(stat_card(label, val, key=key), unsafe_allow_html=True)

# ── Raw data (below tabs) ─────────────────────────────────────────────────────────
st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
with st.expander("📋  View Raw Feature Data"):
    display = df.copy()
    display.insert(len(display.columns), "predicted", y_pred.round(4))
    display = display.iloc[::-1].round(4)
    display.index = display.index.strftime("%Y-%m-%d")
    st.dataframe(display, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="height:1px;
    background:linear-gradient(90deg,rgba(99,102,241,0.25),rgba(168,85,247,0.2),transparent);
    margin:40px 0 16px;"></div>
<div style="display:flex;justify-content:space-between;align-items:center;
    padding-bottom:28px;flex-wrap:wrap;gap:8px;">
    <span style="color:#334155;font-size:12px;">
        MarketPulse AI &mdash; For educational purposes only. Not financial advice.
    </span>
    <span style="color:#334155;font-size:12px;">Powered by XGBoost · Yahoo Finance · Streamlit</span>
</div>
""", unsafe_allow_html=True)
