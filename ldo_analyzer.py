#!/usr/bin/env python3
"""
LDO Deep Analyzer
==================
Pulls price, market, technical, and on-chain (TVL) data for Lido DAO (LDO)
from free public APIs (CoinGecko + DefiLlama, no API key required) and
builds a single self-contained HTML dashboard you can open in any browser.

Run it manually any time, or schedule it (see bottom of this file / README
notes) to run every 4 hours so you always have a fresh report waiting.

Usage:
    python3 ldo_analyzer.py

Output:
    ./docs/index.html            <- always the newest report (served by GitHub Pages)
    ./docs/archive/ldo_report_<ts>.html <- timestamped archive copy
"""

import json
import base64
import io
import statistics
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

COINGECKO_ID = "lido-dao"
OUT_DIR = Path("./docs")
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "archive").mkdir(exist_ok=True)


def fetch_coingecko_snapshot():
    url = f"https://api.coingecko.com/api/v3/coins/{COINGECKO_ID}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "true",
        "developer_data": "false",
        "sparkline": "false",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_coingecko_history(days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{COINGECKO_ID}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_defillama_tvl():
    url = "https://api.llama.fi/protocol/lido"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_fear_greed():
    r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=15)
    r.raise_for_status()
    data = r.json()
    return data["data"][0]


def sma(values, window):
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def pct_change(values, lookback):
    if len(values) <= lookback:
        return None
    return (values[-1] - values[-1 - lookback]) / values[-1 - lookback] * 100


def volatility(values, window=14):
    if len(values) < window + 1:
        return None
    rets = [
        (values[i] - values[i - 1]) / values[i - 1]
        for i in range(len(values) - window, len(values))
    ]
    return statistics.pstdev(rets) * 100


def fibonacci_levels(low, high):
    diff = high - low
    return {
        "0.0% (high)": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100% (low)": low,
    }


def render_price_chart_b64(dates, prices, sma20_series, sma50_series):
    if not HAVE_MPL:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=140)
    ax.plot(dates, prices, label="LDO/USD", color="#2563eb", linewidth=1.6)
    if sma20_series:
        ax.plot(dates[-len(sma20_series):], sma20_series, label="SMA20", color="#f59e0b", linewidth=1.2)
    if sma50_series:
        ax.plot(dates[-len(sma50_series):], sma50_series, label="SMA50", color="#ef4444", linewidth=1.2)
    ax.set_title("LDO/USD - price history")
    ax.set_ylabel("USD")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def build_report():
    snap = fetch_coingecko_snapshot()
    hist = fetch_coingecko_history(days=90)
    try:
        tvl_data = fetch_defillama_tvl()
    except Exception:
        tvl_data = None
    try:
        fng = fetch_fear_greed()
    except Exception:
        fng = None

    md = snap["market_data"]
    prices = [p[1] for p in hist["prices"]]
    volumes = [v[1] for v in hist["total_volumes"]]
    dates = [datetime.fromtimestamp(p[0] / 1000, tz=timezone.utc) for p in hist["prices"]]

    sma20 = [sma(prices[: i + 1], 20) for i in range(len(prices))]
    sma20 = [v for v in sma20 if v is not None]
    sma50 = [sma(prices[: i + 1], 50) for i in range(len(prices))]
    sma50 = [v for v in sma50 if v is not None]

    current_rsi = rsi(prices, 14)
    vol_14d = volatility(prices, 14)

    window_low = min(prices[-30:]) if len(prices) >= 30 else min(prices)
    window_high = max(prices[-30:]) if len(prices) >= 30 else max(prices)
    fib = fibonacci_levels(window_low, window_high)

    chart_b64 = render_price_chart_b64(dates, prices, sma20, sma50)

    changes = {
        "24h": md.get("price_change_percentage_24h"),
        "7d": md.get("price_change_percentage_7d"),
        "30d": md.get("price_change_percentage_30d"),
        "90d": pct_change(prices, min(89, len(prices) - 1)),
    }

    tvl_now = None
    tvl_30d_ago = None
    if tvl_data and "tvl" in tvl_data:
        series = tvl_data["tvl"]
        if series:
            tvl_now = series[-1]["totalLiquidityUSD"]
            if len(series) > 30:
                tvl_30d_ago = series[-30]["totalLiquidityUSD"]

    ctx = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "price": md["current_price"]["usd"],
        "market_cap": md["market_cap"]["usd"],
        "market_cap_rank": snap.get("market_cap_rank"),
        "volume_24h": md["total_volume"]["usd"],
        "ath": md["ath"]["usd"],
        "ath_date": md["ath_date"]["usd"][:10],
        "ath_change_pct": md["ath_change_percentage"]["usd"],
        "atl": md["atl"]["usd"],
        "atl_date": md["atl_date"]["usd"][:10],
        "circ_supply": md["circulating_supply"],
        "max_supply": md.get("max_supply"),
        "changes": changes,
        "rsi14": current_rsi,
        "vol14": vol_14d,
        "sma20_last": sma20[-1] if sma20 else None,
        "sma50_last": sma50[-1] if sma50 else None,
        "fib": fib,
        "window_low": window_low,
        "window_high": window_high,
        "tvl_now": tvl_now,
        "tvl_30d_ago": tvl_30d_ago,
        "fng": fng,
        "chart_b64": chart_b64,
    }
    return ctx


def rsi_read(v):
    if v is None:
        return "n/a"
    if v >= 70:
        return f"{v:.1f} - overbought zone"
    if v <= 30:
        return f"{v:.1f} - oversold zone"
    return f"{v:.1f} - neutral"


def trend_read(price, sma20, sma50):
    if sma20 is None or sma50 is None:
        return "Not enough history yet for SMA20/50 comparison."
    if price > sma20 > sma50:
        return "Price above both SMA20 and SMA50 - short/medium-term structure is constructive."
    if price < sma20 < sma50:
        return "Price below both SMA20 and SMA50 - short/medium-term structure is weak."
    return "Price is mixed relative to SMA20/SMA50 - no clean trend alignment."


def render_html(ctx):
    chg = ctx["changes"]

    def fmt_pct(v):
        if v is None:
            return "n/a"
        color = "#16a34a" if v >= 0 else "#dc2626"
        return f'<span style="color:{color}">{v:+.2f}%</span>'

    fib_rows = "".join(
        f"<tr><td>{k}</td><td>${v:,.4f}</td></tr>" for k, v in ctx["fib"].items()
    )

    tvl_html = "n/a"
    if ctx["tvl_now"] is not None:
        tvl_change = ""
        if ctx["tvl_30d_ago"]:
            delta = (ctx["tvl_now"] - ctx["tvl_30d_ago"]) / ctx["tvl_30d_ago"] * 100
            tvl_change = f" ({fmt_pct(delta)} vs 30d ago)"
        tvl_html = f"${ctx['tvl_now']:,.0f}{tvl_change}"

    fng_html = "n/a"
    if ctx["fng"]:
        fng_html = f"{ctx['fng']['value']} - {ctx['fng']['value_classification']}"

    chart_html = ""
    if ctx["chart_b64"]:
        chart_html = f'<img src="data:image/png;base64,{ctx["chart_b64"]}" style="width:100%;border-radius:8px;margin:16px 0;" />'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LDO Deep Analysis - {ctx['generated_at']}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 860px; margin: 0 auto; padding: 24px 16px 60px; background:#0b1220; color:#e5e7eb; }}
  h1 {{ font-size: 22px; margin-bottom:2px; }}
  .ts {{ color:#9ca3af; font-size:13px; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap:12px; margin:16px 0; }}
  .card {{ background:#141c30; border-radius:10px; padding:14px; }}
  .card .label {{ font-size:12px; color:#9ca3af; text-transform:uppercase; letter-spacing:.03em; }}
  .card .value {{ font-size:19px; font-weight:600; margin-top:4px; }}
  section {{ margin-top:28px; }}
  h2 {{ font-size:16px; border-bottom:1px solid #263149; padding-bottom:6px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; margin-top:8px; }}
  td {{ padding:6px 4px; border-bottom:1px solid #1f2937; }}
  td:last-child {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .note {{ color:#9ca3af; font-size:12.5px; margin-top:24px; line-height:1.5; }}
</style>
</head>
<body>
  <h1>LDO (Lido DAO) - Deep Analysis</h1>
  <div class="ts">Generated {ctx['generated_at']} - refresh by re-running the script</div>

  <div class="grid">
    <div class="card"><div class="label">Price</div><div class="value">${ctx['price']:.4f}</div></div>
    <div class="card"><div class="label">24h</div><div class="value">{fmt_pct(chg['24h'])}</div></div>
    <div class="card"><div class="label">7d</div><div class="value">{fmt_pct(chg['7d'])}</div></div>
    <div class="card"><div class="label">30d</div><div class="value">{fmt_pct(chg['30d'])}</div></div>
    <div class="card"><div class="label">Market Cap</div><div class="value">${ctx['market_cap']:,.0f}</div></div>
    <div class="card"><div class="label">Rank</div><div class="value">#{ctx['market_cap_rank']}</div></div>
    <div class="card"><div class="label">24h Volume</div><div class="value">${ctx['volume_24h']:,.0f}</div></div>
    <div class="card"><div class="label">Circ. Supply</div><div class="value">{ctx['circ_supply']:,.0f}</div></div>
  </div>

  {chart_html}

  <section>
    <h2>Technicals (daily, 90d lookback)</h2>
    <table>
      <tr><td>RSI (14)</td><td>{rsi_read(ctx['rsi14'])}</td></tr>
      <tr><td>SMA20</td><td>${ctx['sma20_last']:.4f}</td></tr>
      <tr><td>SMA50</td><td>${ctx['sma50_last']:.4f}</td></tr>
      <tr><td>14d volatility (stdev of daily returns)</td><td>{ctx['vol14']:.2f}%</td></tr>
      <tr><td>30d range</td><td>${ctx['window_low']:.4f} - ${ctx['window_high']:.4f}</td></tr>
    </table>
    <p style="font-size:14px;color:#cbd5e1;">{trend_read(ctx['price'], ctx['sma20_last'], ctx['sma50_last'])}</p>
  </section>

  <section>
    <h2>Fibonacci levels (based on 30d range)</h2>
    <table>{fib_rows}</table>
  </section>

  <section>
    <h2>Fundamentals / on-chain</h2>
    <table>
      <tr><td>Lido protocol TVL</td><td>{tvl_html}</td></tr>
      <tr><td>ATH</td><td>${ctx['ath']:,.4f} ({ctx['ath_date']}, {ctx['ath_change_pct']:.1f}% from ATH)</td></tr>
      <tr><td>ATL</td><td>${ctx['atl']:,.6f} ({ctx['atl_date']})</td></tr>
      <tr><td>Max supply</td><td>{ctx['max_supply']:,.0f}</td></tr>
    </table>
  </section>

  <section>
    <h2>Macro context</h2>
    <table>
      <tr><td>Crypto Fear &amp; Greed Index</td><td>{fng_html}</td></tr>
    </table>
  </section>

  <div class="note">
    Data: CoinGecko (price/market), DefiLlama (TVL), alternative.me (Fear &amp; Greed). Informational only -
    not financial advice. Verify against your exchange before trading.
  </div>
</body>
</html>"""
    return html


def main():
    ctx = build_report()
    html = render_html(ctx)

    latest_path = OUT_DIR / "index.html"
    latest_path.write_text(html, encoding="utf-8")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = OUT_DIR / "archive" / f"ldo_report_{ts}.html"
    archive_path.write_text(html, encoding="utf-8")

    print(f"Report written: {latest_path.resolve()}")
    print(f"Archived copy:  {archive_path.resolve()}")
    print(f"Price: ${ctx['price']:.4f} | 24h: {ctx['changes']['24h']:.2f}% | RSI14: {ctx['rsi14']:.1f}" if ctx['rsi14'] else "")


if __name__ == "__main__":
    main()
