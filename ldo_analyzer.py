#!/usr/bin/env python3
"""
LDO Deep Analyzer V2
=====================
Lido DAO (LDO) dashboard with:
- CoinGecko market snapshot
- Binance 1H / 4H / 1D OHLCV technicals
- EMA20/50/200, SMA20/50, RSI14, MACD, ATR, volatility
- 30/90-day support/resistance and Fibonacci levels
- DefiLlama Lido TVL + 30d change
- Lido API: stETH APR + stETH/ETH price data when available
- Fear & Greed
- Automatic Persian summary at the end of the dashboard
- Timestamped archive
- No API keys required

Run:
    python3 ldo_analyzer_v2.py

Output:
    ./docs/index.html
    ./docs/archive/ldo_report_<timestamp>.html
"""

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
BINANCE_SYMBOL = "LDOUSDT"

OUT_DIR = Path("./docs")
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "archive").mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "LDO-Deep-Analyzer/2.0",
    "Accept": "application/json",
}


# -----------------------------
# Generic HTTP
# -----------------------------
def get_json(url, params=None, timeout=20):
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


# -----------------------------
# Data sources
# -----------------------------
def fetch_coingecko_snapshot():
    return get_json(
        f"https://api.coingecko.com/api/v3/coins/{COINGECKO_ID}",
        {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "false",
            "sparkline": "false",
        },
    )


def fetch_coingecko_history(days=90):
    return get_json(
        f"https://api.coingecko.com/api/v3/coins/{COINGECKO_ID}/market_chart",
        {"vs_currency": "usd", "days": days, "interval": "daily"},
    )


def fetch_binance_klines(interval="4h", limit=1000):
    data = get_json(
        "https://api.binance.com/api/v3/klines",
        {"symbol": BINANCE_SYMBOL, "interval": interval, "limit": limit},
    )
    return [
        {
            "time": int(x[0]),
            "open": float(x[1]),
            "high": float(x[2]),
            "low": float(x[3]),
            "close": float(x[4]),
            "volume": float(x[5]),
        }
        for x in data
    ]


def fetch_defillama_lido():
    return get_json("https://api.llama.fi/protocol/lido")


def fetch_fear_greed():
    data = get_json("https://api.alternative.me/fng/?limit=1", timeout=15)
    return data["data"][0]


def fetch_lido_apr():
    try:
        data = get_json("https://eth-api.lido.fi/v1/protocol/steth/apr/last")
        return extract_first_number(
            data,
            [
                "apr",
                "value",
                "annualPercentageRate",
            ],
        )
    except Exception:
        return None


def fetch_lido_stats():
    try:
        return get_json("https://eth-api.lido.fi/v1/protocol/steth/stats")
    except Exception:
        return None


def fetch_lido_steth_price():
    try:
        data = get_json("https://eth-api.lido.fi/v1/protocol/steth/price")
        return extract_first_number(
            data,
            ["price", "value", "usd"],
        )
    except Exception:
        return None


# -----------------------------
# Math / indicators
# -----------------------------
def sma(values, window):
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def ema_series(values, window):
    if len(values) < window:
        return []
    multiplier = 2 / (window + 1)
    ema = sum(values[:window]) / window
    result = [ema]
    for price in values[window:]:
        ema = (price - ema) * multiplier + ema
        result.append(ema)
    return result


def ema_last(values, window):
    s = ema_series(values, window)
    return s[-1] if s else None


def rsi(values, period=14):
    if len(values) < period + 1:
        return None

    gains = []
    losses = []
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


def macd(values, fast=12, slow=26, signal=9):
    ef = ema_series(values, fast)
    es = ema_series(values, slow)
    if not ef or not es:
        return None, None, None

    # Align fast EMA with slow EMA.
    offset = slow - fast
    fast_aligned = ef[offset:]
    n = min(len(fast_aligned), len(es))
    macd_line = [
        fast_aligned[-n + i] - es[-n + i]
        for i in range(n)
    ]

    signal_series = ema_series(macd_line, signal)
    if not signal_series:
        return macd_line[-1], None, None

    signal_last = signal_series[-1]
    macd_last = macd_line[-1]
    return macd_last, signal_last, macd_last - signal_last


def atr(klines, period=14):
    if len(klines) < period + 1:
        return None

    trs = []
    for i in range(1, len(klines)):
        h = klines[i]["high"]
        l = klines[i]["low"]
        pc = klines[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))

    return sum(trs[-period:]) / period


def volatility(values, window=14):
    if len(values) < window + 1:
        return None
    rets = [
        (values[i] - values[i - 1]) / values[i - 1]
        for i in range(len(values) - window, len(values))
    ]
    return statistics.pstdev(rets) * 100


def pct_change(values, lookback):
    if len(values) <= lookback:
        return None
    old = values[-1 - lookback]
    if old == 0:
        return None
    return (values[-1] - old) / old * 100


def fibonacci_levels(low, high):
    diff = high - low
    return {
        "0.0% (high)": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.500 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100% (low)": low,
    }


def support_resistance(klines, lookback=90):
    k = klines[-lookback:] if len(klines) > lookback else klines
    highs = sorted([x["high"] for x in k], reverse=True)
    lows = sorted([x["low"] for x in k])

    # Simple, transparent levels rather than pretending they are exact orders.
    resistance = []
    support = []

    for x in highs:
        if not resistance or abs(x - resistance[-1]) / resistance[-1] > 0.025:
            resistance.append(x)
        if len(resistance) >= 4:
            break

    for x in lows:
        if not support or abs(x - support[-1]) / support[-1] > 0.025:
            support.append(x)
        if len(support) >= 4:
            break

    return sorted(support), sorted(resistance, reverse=True)


def extract_first_number(obj, preferred_keys):
    if isinstance(obj, dict):
        for key in preferred_keys:
            if key in obj:
                value = obj[key]
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except ValueError:
                        pass
        for value in obj.values():
            found = extract_first_number(value, preferred_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = extract_first_number(value, preferred_keys)
            if found is not None:
                return found
    return None


def extract_number_by_keys(obj, keys):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in [k.lower() for k in keys]:
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except ValueError:
                        pass
            found = extract_number_by_keys(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = extract_number_by_keys(value, keys)
            if found is not None:
                return found
    return None


# -----------------------------
# Timeframe analysis
# -----------------------------
def analyze_timeframe(klines):
    closes = [x["close"] for x in klines]
    price = closes[-1]

    ema20 = ema_last(closes, 20)
    ema50 = ema_last(closes, 50)
    ema200 = ema_last(closes, 200)
    rsi14 = rsi(closes, 14)
    macd_v, macd_sig, macd_hist = macd(closes)
    atr14 = atr(klines, 14)

    score = 50
    reasons = []

    if ema20 and price > ema20:
        score += 8
        reasons.append("قیمت بالای EMA20")
    elif ema20:
        score -= 8
        reasons.append("قیمت زیر EMA20")

    if ema50 and price > ema50:
        score += 8
        reasons.append("قیمت بالای EMA50")
    elif ema50:
        score -= 8
        reasons.append("قیمت زیر EMA50")

    if ema200:
        if price > ema200:
            score += 10
            reasons.append("قیمت بالای EMA200")
        else:
            score -= 10
            reasons.append("قیمت زیر EMA200")

    if rsi14 is not None:
        if 50 <= rsi14 <= 68:
            score += 8
            reasons.append("RSI در محدوده مثبت و بدون اشباع شدید")
        elif rsi14 > 70:
            score -= 3
            reasons.append("RSI وارد ناحیه اشباع خرید شده")
        elif rsi14 < 30:
            score += 2
            reasons.append("RSI در اشباع فروش است؛ احتمال واکنش وجود دارد")
        else:
            score += 0

    if macd_hist is not None:
        if macd_hist > 0:
            score += 8
            reasons.append("MACD Histogram مثبت")
        else:
            score -= 8
            reasons.append("MACD Histogram منفی")

    score = max(0, min(100, score))

    if score >= 65:
        state = "مثبت"
    elif score <= 35:
        state = "ضعیف"
    else:
        state = "خنثی"

    return {
        "price": price,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
        "rsi": rsi14,
        "macd": macd_v,
        "macd_signal": macd_sig,
        "macd_hist": macd_hist,
        "atr": atr14,
        "atr_pct": (atr14 / price * 100) if atr14 and price else None,
        "score": score,
        "state": state,
        "reasons": reasons,
        "change_24": pct_change(closes, 6 if len(closes) > 6 else 1),
        "change_7": pct_change(closes, 42 if len(closes) > 42 else 1),
    }


# -----------------------------
# Automatic Persian conclusion
# -----------------------------
def build_persian_summary(ctx):
    a1 = ctx["tf"]["1h"]
    a4 = ctx["tf"]["4h"]
    ad = ctx["tf"]["1d"]

    scores = [x["score"] for x in [a1, a4, ad]]
    technical_score = round(sum(scores) / len(scores))

    tvl_change = ctx.get("tvl_change_30d")
    apr = ctx.get("lido_apr")
    premium = ctx.get("steth_eth_ratio")

    fund_score = 50
    fund_reasons = []

    if tvl_change is not None:
        if tvl_change > 5:
            fund_score += 15
            fund_reasons.append("TVL طی ۳۰ روز رشد کرده")
        elif tvl_change < -5:
            fund_score -= 15
            fund_reasons.append("TVL طی ۳۰ روز کاهش داشته")
        else:
            fund_reasons.append("TVL تقریباً بدون تغییر شدید است")

    if apr is not None:
        if apr >= 2.5:
            fund_score += 10
            fund_reasons.append("APR استیکینگ در سطح قابل‌توجهی قرار دارد")
        elif apr < 2:
            fund_score -= 5
            fund_reasons.append("APR استیکینگ نسبتاً پایین است")

    if premium is not None:
        if 0.995 <= premium <= 1.005:
            fund_score += 10
            fund_reasons.append("stETH نسبت به ETH نزدیک به برابری است")
        elif premium < 0.99:
            fund_score -= 8
            fund_reasons.append("stETH نسبت به ETH با فاصله منفی معامله/قیمت‌گذاری شده")
        else:
            fund_score += 5
            fund_reasons.append("stETH نسبت به ETH بالاتر از برابری است")

    fund_score = max(0, min(100, fund_score))
    overall = round(technical_score * 0.65 + fund_score * 0.35)

    if overall >= 65:
        overall_state = "مثبت"
    elif overall <= 35:
        overall_state = "ضعیف"
    else:
        overall_state = "خنثی / نیازمند تأیید"

    direction = ""
    if a1["state"] == "مثبت" and a4["state"] == "مثبت":
        direction = "در تایم‌فریم‌های 1H و 4H، مومنتوم فعلاً هم‌جهت و مثبت است."
    elif a1["state"] == "ضعیف" and a4["state"] == "ضعیف":
        direction = "در 1H و 4H فشار نزولی هم‌جهت دیده می‌شود."
    else:
        direction = "بین 1H و 4H هم‌جهتی کامل وجود ندارد و بهتر است شکست/تأیید بعدی بررسی شود."

    risk = []
    if a4["rsi"] is not None and a4["rsi"] >= 70:
        risk.append("RSI چهار‌ساعته بالا است؛ احتمال اصلاح کوتاه‌مدت وجود دارد.")
    if a4["rsi"] is not None and a4["rsi"] <= 30:
        risk.append("RSI چهار‌ساعته در اشباع فروش است؛ واکنش صعودی ممکن است اما تأیید لازم است.")
    if a4["ema200"] and a4["price"] < a4["ema200"]:
        risk.append("قیمت 4H زیر EMA200 است؛ این سطح باید دوباره پس گرفته شود.")
    if tvl_change is not None and tvl_change < 0:
        risk.append("کاهش TVL در ۳۰ روز اخیر یک عامل منفی بنیادی است.")
    if not risk:
        risk.append("فعلاً هشدار تکنیکال/بنیادی بسیار شدیدی در داده‌های این گزارش دیده نمی‌شود.")

    supports = ctx.get("support", [])
    resistances = ctx.get("resistance", [])

    summary = []
    summary.append(
        f"وضعیت کلی LDO: {overall_state} | امتیاز ترکیبی {overall}/100 "
        f"(تکنیکال {technical_score}/100، بنیادی {fund_score}/100)."
    )
    summary.append(direction)

    if supports:
        summary.append(
            "حمایت‌های مهم تقریبی: " +
            "، ".join(f"${x:.4f}" for x in supports[:3]) + "."
        )
    if resistances:
        summary.append(
            "مقاومت‌های مهم تقریبی: " +
            "، ".join(f"${x:.4f}" for x in resistances[:3]) + "."
        )

    summary.append("نکات بنیادی: " + "؛ ".join(fund_reasons) + ".")
    summary.append("ریسک/هشدار: " + " ".join(risk))

    summary.append(
        "این جمع‌بندی به‌صورت خودکار از داده‌های همان لحظه ساخته می‌شود؛ "
        "به‌تنهایی سیگنال خرید یا فروش محسوب نمی‌شود."
    )

    return {
        "text": " ".join(summary),
        "technical_score": technical_score,
        "fundamental_score": fund_score,
        "overall": overall,
        "state": overall_state,
    }


# -----------------------------
# Chart
# -----------------------------
def render_chart(klines):
    if not HAVE_MPL or not klines:
        return None

    dates = [
        datetime.fromtimestamp(x["time"] / 1000, tz=timezone.utc)
        for x in klines[-180:]
    ]
    prices = [x["close"] for x in klines[-180:]]

    e20_full = ema_series([x["close"] for x in klines], 20)
    e50_full = ema_series([x["close"] for x in klines], 50)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    ax.plot(dates, prices, label="LDO/USDT", linewidth=1.5)

    if e20_full:
        e20 = e20_full[-len(dates):]
        ax.plot(dates, e20, label="EMA20", linewidth=1.1)

    if e50_full:
        e50 = e50_full[-len(dates):]
        ax.plot(dates, e50, label="EMA50", linewidth=1.1)

    ax.set_title("LDO/USDT - 4H")
    ax.set_ylabel("USDT")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# -----------------------------
# Main data build
# -----------------------------
def build_report():
    snap = fetch_coingecko_snapshot()
    hist = fetch_coingecko_history(90)

    try:
        d1 = fetch_binance_klines("1h", 1000)
    except Exception:
        d1 = []

    try:
        d4 = fetch_binance_klines("4h", 1000)
    except Exception:
        d4 = []

    try:
        dd = fetch_binance_klines("1d", 500)
    except Exception:
        dd = []

    try:
        tvl_data = fetch_defillama_lido()
    except Exception:
        tvl_data = None

    try:
        fng = fetch_fear_greed()
    except Exception:
        fng = None

    lido_apr = fetch_lido_apr()
    lido_stats = fetch_lido_stats()
    steth_price = fetch_lido_steth_price()

    md = snap["market_data"]
    cg_prices = [p[1] for p in hist["prices"]]

    # Fallback to CoinGecko daily data if Binance is unavailable.
    if not dd:
        dd = [
            {
                "time": int(hist["prices"][i][0]),
                "open": float(hist["prices"][i][1]),
                "high": float(hist["prices"][i][1]),
                "low": float(hist["prices"][i][1]),
                "close": float(hist["prices"][i][1]),
                "volume": float(hist["total_volumes"][i][1]),
            }
            for i in range(len(hist["prices"]))
        ]

    tf = {
        "1h": analyze_timeframe(d1) if d1 else None,
        "4h": analyze_timeframe(d4) if d4 else None,
        "1d": analyze_timeframe(dd) if dd else None,
    }

    # If 1H or 4H fails, use a neutral placeholder rather than crashing.
    for key in ["1h", "4h"]:
        if tf[key] is None:
            tf[key] = {
                "price": md["current_price"]["usd"],
                "ema20": None, "ema50": None, "ema200": None,
                "rsi": None, "macd": None, "macd_signal": None,
                "macd_hist": None, "atr": None, "atr_pct": None,
                "score": 50, "state": "داده در دسترس نیست",
                "reasons": [], "change_24": None, "change_7": None,
            }

    if tf["1d"] is None:
        tf["1d"] = tf["4h"]

    # TVL
    tvl_now = None
    tvl_30d_ago = None
    if tvl_data and "tvl" in tvl_data and tvl_data["tvl"]:
        series = tvl_data["tvl"]
        tvl_now = series[-1].get("totalLiquidityUSD")
        if len(series) >= 31:
            tvl_30d_ago = series[-31].get("totalLiquidityUSD")

    tvl_change_30d = None
    if tvl_now is not None and tvl_30d_ago:
        tvl_change_30d = (tvl_now - tvl_30d_ago) / tvl_30d_ago * 100

    # stETH/ETH ratio if Lido API gives a USD stETH price.
    eth_price = None
    try:
        eth = get_json(
            "https://api.coingecko.com/api/v3/simple/price",
            {"ids": "ethereum", "vs_currencies": "usd"},
        )
        eth_price = eth["ethereum"]["usd"]
    except Exception:
        pass

    steth_eth_ratio = None
    if steth_price and eth_price:
        steth_eth_ratio = steth_price / eth_price

    # Try to extract pooled ETH from Lido stats if schema changes.
    pooled_eth = None
    if lido_stats:
        pooled_eth = extract_number_by_keys(
            lido_stats,
            [
                "totalPooledEther",
                "totalPooledEth",
                "pooledEth",
                "totalPooledEtherAmount",
            ],
        )
        if pooled_eth and pooled_eth > 1e18:
            pooled_eth = pooled_eth / 1e18

    support, resistance = support_resistance(dd, 90)

    fib_low = min([x["low"] for x in dd[-90:]])
    fib_high = max([x["high"] for x in dd[-90:]])
    fib = fibonacci_levels(fib_low, fib_high)

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
        "changes": {
            "24h": md.get("price_change_percentage_24h"),
            "7d": md.get("price_change_percentage_7d"),
            "30d": md.get("price_change_percentage_30d"),
            "90d": pct_change(cg_prices, min(89, len(cg_prices) - 1)),
        },
        "tf": tf,
        "tvl_now": tvl_now,
        "tvl_30d_ago": tvl_30d_ago,
        "tvl_change_30d": tvl_change_30d,
        "fng": fng,
        "lido_apr": lido_apr,
        "lido_stats": lido_stats,
        "pooled_eth": pooled_eth,
        "steth_price": steth_price,
        "eth_price": eth_price,
        "steth_eth_ratio": steth_eth_ratio,
        "support": support,
        "resistance": resistance,
        "fib": fib,
        "chart_b64": render_chart(d4),
    }

    ctx["summary"] = build_persian_summary(ctx)
    return ctx


# -----------------------------
# HTML
# -----------------------------
def fmt_money(v, digits=2):
    if v is None:
        return "n/a"
    return f"${v:,.{digits}f}"


def fmt_pct(v):
    if v is None:
        return "n/a"
    return f"{v:+.2f}%"


def fmt_num(v, digits=4):
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def indicator_table(tf):
    return f"""
    <table>
      <tr><td>Price</td><td>{fmt_money(tf['price'], 4)}</td></tr>
      <tr><td>EMA20</td><td>{fmt_money(tf['ema20'], 4)}</td></tr>
      <tr><td>EMA50</td><td>{fmt_money(tf['ema50'], 4)}</td></tr>
      <tr><td>EMA200</td><td>{fmt_money(tf['ema200'], 4)}</td></tr>
      <tr><td>RSI14</td><td>{fmt_num(tf['rsi'], 1)}</td></tr>
      <tr><td>MACD</td><td>{fmt_num(tf['macd'], 5)}</td></tr>
      <tr><td>MACD Signal</td><td>{fmt_num(tf['macd_signal'], 5)}</td></tr>
      <tr><td>MACD Histogram</td><td>{fmt_num(tf['macd_hist'], 5)}</td></tr>
      <tr><td>ATR14</td><td>{fmt_money(tf['atr'], 5)} ({fmt_pct(tf['atr_pct'])})</td></tr>
      <tr><td>Score</td><td>{tf['score']}/100 — {tf['state']}</td></tr>
    </table>
    """


def render_html(ctx):
    chg = ctx["changes"]
    s = ctx["summary"]

    chart_html = ""
    if ctx["chart_b64"]:
        chart_html = (
            f'<img src="data:image/png;base64,{ctx["chart_b64"]}" '
            'style="width:100%;border-radius:10px;margin:16px 0;" />'
        )

    def list_rows(values):
        if not values:
            return "<tr><td>داده موجود نیست</td><td>n/a</td></tr>"
        return "".join(
            f"<tr><td>سطح</td><td>{fmt_money(v, 4)}</td></tr>" for v in values[:4]
        )

    fib_rows = "".join(
        f"<tr><td>{k}</td><td>{fmt_money(v, 4)}</td></tr>"
        for k, v in ctx["fib"].items()
    )

    tvl_text = fmt_money(ctx["tvl_now"], 0)
    if ctx["tvl_change_30d"] is not None:
        tvl_text += f" ({fmt_pct(ctx['tvl_change_30d'])} در ۳۰ روز)"

    fng_text = "n/a"
    if ctx["fng"]:
        fng_text = f"{ctx['fng']['value']} — {ctx['fng']['value_classification']}"

    html = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LDO Deep Analyzer V2 — {ctx['generated_at']}</title>
<style>
body {{
  font-family: Tahoma, Arial, sans-serif;
  max-width: 1050px;
  margin:0 auto;
  padding:22px 14px 60px;
  background:#0b1220;
  color:#e5e7eb;
  line-height:1.8;
}}
h1 {{font-size:23px;margin:0 0 3px}}
h2 {{font-size:17px;border-bottom:1px solid #29354d;padding-bottom:7px;margin-top:30px}}
.ts {{color:#9ca3af;font-size:12px}}
.grid {{
 display:grid;
 grid-template-columns:repeat(auto-fit,minmax(155px,1fr));
 gap:10px;margin:15px 0;
}}
.card {{
 background:#141c30;border-radius:10px;padding:13px;
}}
.label {{font-size:11px;color:#9ca3af}}
.value {{font-size:18px;font-weight:700;margin-top:2px}}
.summary {{
 background:#17243a;
 border:1px solid #31435f;
 border-radius:12px;
 padding:16px;
 margin-top:18px;
}}
.score {{
 display:grid;
 grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
 gap:10px;margin-top:12px;
}}
.scorebox {{
 background:#101a2c;
 border-radius:9px;padding:12px;text-align:center;
}}
table {{
 width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;
}}
td {{
 padding:6px 5px;border-bottom:1px solid #202b40;
}}
td:last-child {{
 text-align:left;font-variant-numeric:tabular-nums;
}}
.columns {{
 display:grid;
 grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
 gap:15px;
}}
.box {{
 background:#111a2b;border-radius:10px;padding:13px;
}}
.note {{
 color:#9ca3af;font-size:12px;margin-top:25px;
}}
</style>
</head>
<body>

<h1>LDO (Lido DAO) — Deep Analyzer V2</h1>
<div class="ts">Generated {ctx['generated_at']} — CoinGecko + Binance + DefiLlama + Lido API + Fear &amp; Greed</div>

<div class="summary">
  <h2 style="margin-top:0">🇮🇷 خلاصه وضعیت LDO</h2>
  <p>{s['text']}</p>
  <div class="score">
    <div class="scorebox"><div>وضعیت کلی</div><strong>{s['state']}</strong></div>
    <div class="scorebox"><div>امتیاز ترکیبی</div><strong>{s['overall']}/100</strong></div>
    <div class="scorebox"><div>تکنیکال</div><strong>{s['technical_score']}/100</strong></div>
    <div class="scorebox"><div>بنیادی</div><strong>{s['fundamental_score']}/100</strong></div>
  </div>
</div>

<div class="grid">
  <div class="card"><div class="label">قیمت</div><div class="value">{fmt_money(ctx['price'],4)}</div></div>
  <div class="card"><div class="label">24h</div><div class="value">{fmt_pct(chg['24h'])}</div></div>
  <div class="card"><div class="label">7d</div><div class="value">{fmt_pct(chg['7d'])}</div></div>
  <div class="card"><div class="label">30d</div><div class="value">{fmt_pct(chg['30d'])}</div></div>
  <div class="card"><div class="label">Market Cap</div><div class="value">{fmt_money(ctx['market_cap'],0)}</div></div>
  <div class="card"><div class="label">Rank</div><div class="value">#{ctx['market_cap_rank']}</div></div>
  <div class="card"><div class="label">24h Volume</div><div class="value">{fmt_money(ctx['volume_24h'],0)}</div></div>
  <div class="card"><div class="label">Circ. Supply</div><div class="value">{ctx['circ_supply']:,.0f}</div></div>
</div>

{chart_html}

<h2>📊 تکنیکال چند تایم‌فریمی</h2>
<div class="columns">
  <div class="box"><h3>1H</h3>{indicator_table(ctx['tf']['1h'])}</div>
  <div class="box"><h3>4H</h3>{indicator_table(ctx['tf']['4h'])}</div>
  <div class="box"><h3>1D</h3>{indicator_table(ctx['tf']['1d'])}</div>
</div>

<h2>📍 حمایت و مقاومت تقریبی — 90D</h2>
<div class="columns">
  <div class="box"><h3>حمایت‌ها</h3><table>{list_rows(ctx['support'])}</table></div>
  <div class="box"><h3>مقاومت‌ها</h3><table>{list_rows(ctx['resistance'])}</table></div>
</div>

<h2>📐 Fibonacci — محدوده 90 روزه</h2>
<table>{fib_rows}</table>

<h2>🏦 وضعیت بنیادی و On-chain</h2>
<table>
<tr><td>Lido TVL</td><td>{tvl_text}</td></tr>
<tr><td>Lido staking APR</td><td>{fmt_pct(ctx['lido_apr'])}</td></tr>
<tr><td>stETH price</td><td>{fmt_money(ctx['steth_price'],4)}</td></tr>
<tr><td>ETH price</td><td>{fmt_money(ctx['eth_price'],4)}</td></tr>
<tr><td>stETH / ETH</td><td>{fmt_num(ctx['steth_eth_ratio'],6)}</td></tr>
<tr><td>Estimated pooled ETH</td><td>{fmt_num(ctx['pooled_eth'],2)}</td></tr>
<tr><td>Fear &amp; Greed</td><td>{fng_text}</td></tr>
<tr><td>ATH</td><td>{fmt_money(ctx['ath'],4)} — {ctx['ath_date']} — {ctx['ath_change_pct']:.1f}% از ATH فاصله</td></tr>
<tr><td>ATL</td><td>{fmt_money(ctx['atl'],6)} — {ctx['atl_date']}</td></tr>
<tr><td>Max supply</td><td>{ctx['max_supply']:,.0f}</td></tr>
</table>

<h2>🧠 چرا این وضعیت صادر شده؟</h2>
<div class="columns">
  <div class="box">
    <h3>1H</h3>
    <ul>{"".join(f"<li>{x}</li>" for x in ctx['tf']['1h']['reasons']) or "<li>داده کافی نیست</li>"}</ul>
  </div>
  <div class="box">
    <h3>4H</h3>
    <ul>{"".join(f"<li>{x}</li>" for x in ctx['tf']['4h']['reasons']) or "<li>داده کافی نیست</li>"}</ul>
  </div>
  <div class="box">
    <h3>1D</h3>
    <ul>{"".join(f"<li>{x}</li>" for x in ctx['tf']['1d']['reasons']) or "<li>داده کافی نیست</li>"}</ul>
  </div>
</div>

<div class="note">
این داشبورد ابزار تحلیلی است، نه توصیه مالی. سطوح حمایت/مقاومت به‌صورت الگوریتمی و تقریبی محاسبه شده‌اند.
برای تصمیم معاملاتی، قیمت و حجم صرافی خودت را نیز بررسی کن. در صورت قطعی بودن APIها، داده‌های 1H/4H از Binance
و داده‌های بنیادی از DefiLlama/Lido API می‌آیند.
</div>

</body>
</html>"""
    return html


def main():
    try:
        ctx = build_report()
        html = render_html(ctx)

        latest_path = OUT_DIR / "index.html"
        latest_path.write_text(html, encoding="utf-8")

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_path = OUT_DIR / "archive" / f"ldo_report_{ts}.html"
        archive_path.write_text(html, encoding="utf-8")

        print(f"Report written: {latest_path.resolve()}")
        print(f"Archived copy:  {archive_path.resolve()}")
        print(f"Price: ${ctx['price']:.4f}")
        print(f"Overall: {ctx['summary']['state']} ({ctx['summary']['overall']}/100)")
        print("\nPERSIAN SUMMARY:")
        print(ctx["summary"]["text"])

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
