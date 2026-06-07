import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Config fra env ────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
FINNHUB_KEY      = os.getenv("FINNHUB_KEY", "")
AV_KEY           = os.getenv("AV_KEY", "")

# ── Trading parametre ─────────────────────────────────────────────────────────
SYMBOLS = {
    "NVDA":  "NVIDIA",
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "USO":   "US Oil ETF",
    "GLD":   "Guld ETF",
}

CAPITAL          = 10_000.0
RISK_PER_TRADE   = 0.02
MAX_POSITIONS    = 3
ATR_SL_MULT      = 1.5
ATR_TP_MULT      = 3.0
RSI_PERIOD       = 14
RSI_OVERSOLD     = 35
RSI_OVERBOUGHT   = 65
NEWS_MIN_SCORE   = 0.15

# ── State ─────────────────────────────────────────────────────────────────────
positions   = {}
trade_log   = []
capital     = CAPITAL
analyzer    = SentimentIntensityAnalyzer()


# ═════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ═════════════════════════════════════════════════════════════════════════════
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram ikke konfigureret")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        log.error(f"Telegram fejl: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# MARKEDSDATA via Alpha Vantage
# ═════════════════════════════════════════════════════════════════════════════
def get_price_data(symbol: str) -> pd.DataFrame | None:
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_INTRADAY"
            f"&symbol={symbol}"
            f"&interval=60min"
            f"&outputsize=full"
            f"&apikey={AV_KEY}"
        )
        resp = requests.get(url, timeout=15)
        data = resp.json()

        ts_key = "Time Series (60min)"
        if ts_key not in data:
            log.error(f"Alpha Vantage fejl for {symbol}: {data.get('Note') or data.get('Information') or data}")
            return None

        rows = []
        for dt_str, vals in data[ts_key].items():
            rows.append({
                "Datetime": pd.to_datetime(dt_str),
                "Open":     float(vals["1. open"]),
                "High":     float(vals["2. high"]),
                "Low":      float(vals["3. low"]),
                "Close":    float(vals["4. close"]),
                "Volume":   float(vals["5. volume"]),
            })

        df = pd.DataFrame(rows).sort_values("Datetime").reset_index(drop=True)
        if len(df) < 50:
            log.warning(f"For få datapunkter for {symbol}: {len(df)}")
            return None
        return df

    except Exception as e:
        log.error(f"Pris-data fejl {symbol}: {e}")
        return None


def get_current_price(symbol: str) -> float | None:
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE"
            f"&symbol={symbol}"
            f"&apikey={AV_KEY}"
        )
        resp = requests.get(url, timeout=10)
        data = resp.json()
        price = data.get("Global Quote", {}).get("05. price")
        return float(price) if price else None
    except Exception as e:
        log.error(f"Pris fejl {symbol}: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# TEKNISKE INDIKATORER
# ═════════════════════════════════════════════════════════════════════════════
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series):
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def get_signals(symbol: str) -> dict | None:
    df = get_price_data(symbol)
    if df is None:
        return None

    close = df["Close"]
    rsi   = calc_rsi(close, RSI_PERIOD)
    macd, macd_signal = calc_macd(close)
    atr   = calc_atr(df)

    last_rsi    = float(rsi.iloc[-1])
    last_macd   = float(macd.iloc[-1])
    last_signal = float(macd_signal.iloc[-1])
    last_atr    = float(atr.iloc[-1])
    last_price  = float(close.iloc[-1])

    ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    uptrend = ema50 > ema200

    return {
        "price":    last_price,
        "rsi":      last_rsi,
        "macd":     last_macd,
        "macd_sig": last_signal,
        "atr":      last_atr,
        "uptrend":  uptrend,
    }


# ═════════════════════════════════════════════════════════════════════════════
# NYHEDER & SENTIMENT
# ═════════════════════════════════════════════════════════════════════════════
def get_news_sentiment(symbol: str) -> float:
    ticker_map = {"USO": "CL", "GLD": "GC"}
    ticker = ticker_map.get(symbol, symbol)
    try:
        to_date   = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}&from={from_date}&to={to_date}&token={FINNHUB_KEY}"
        )
        resp = requests.get(url, timeout=10)
        news = resp.json()
        if not isinstance(news, list) or len(news) == 0:
            return 0.0
        scores = []
        for article in news[:10]:
            text  = f"{article.get('headline', '')}. {article.get('summary', '')}"
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        log.error(f"Nyheder fejl {symbol}: {e}")
        return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# POSITION SIZING
# ═════════════════════════════════════════════════════════════════════════════
def calc_position_size(price: float, atr: float) -> tuple[float, float, float]:
    sl_dist  = atr * ATR_SL_MULT
    risk_usd = capital * RISK_PER_TRADE
    size     = risk_usd / sl_dist
    sl_price = price - sl_dist
    tp_price = price + (sl_dist * ATR_TP_MULT)
    return round(size, 4), round(sl_price, 4), round(tp_price, 4)


# ═════════════════════════════════════════════════════════════════════════════
# TRADING LOGIK
# ═════════════════════════════════════════════════════════════════════════════
def check_entry(symbol: str, name: str):
    global capital

    if symbol in positions:
        return
    if len(positions) >= MAX_POSITIONS:
        return

    sig = get_signals(symbol)
    if sig is None:
        return

    sentiment = get_news_sentiment(symbol)
    price     = sig["price"]
    rsi       = sig["rsi"]
    macd      = sig["macd"]
    macd_sig  = sig["macd_sig"]
    uptrend   = sig["uptrend"]
    atr       = sig["atr"]

    long_signal = (
        rsi < RSI_OVERSOLD and
        macd > macd_sig and
        uptrend and
        sentiment > NEWS_MIN_SCORE
    )

    short_signal = (
        rsi > RSI_OVERBOUGHT and
        macd < macd_sig and
        not uptrend and
        sentiment < -NEWS_MIN_SCORE
    )

    if long_signal:
        size, sl, tp = calc_position_size(price, atr)
        cost = size * price
        if cost > capital:
            return

        positions[symbol] = {
            "entry": price, "sl": sl, "tp": tp,
            "size": size, "direction": "LONG", "name": name,
        }
        capital -= cost

        send_telegram(
            f"🟢 <b>PAPER LONG åbnet</b>\n"
            f"Aktie: {name} ({symbol})\n"
            f"Pris: {price:.4f}\n"
            f"SL: {sl:.4f} | TP: {tp:.4f}\n"
            f"Sentiment: positiv | RSI: {rsi:.1f}\n"
            f"Kapital: {capital:.2f} USD"
        )
        log.info(f"LONG åbnet: {symbol} @ {price}")

    elif short_signal:
        size, _, _ = calc_position_size(price, atr)
        sl = price + (atr * ATR_SL_MULT)
        tp = price - (atr * ATR_SL_MULT * ATR_TP_MULT)

        positions[symbol] = {
            "entry": price, "sl": sl, "tp": tp,
            "size": size, "direction": "SHORT", "name": name,
        }

        send_telegram(
            f"🔴 <b>PAPER SHORT åbnet</b>\n"
            f"Aktie: {name} ({symbol})\n"
            f"Pris: {price:.4f}\n"
            f"SL: {sl:.4f} | TP: {tp:.4f}\n"
            f"Sentiment: negativ | RSI: {rsi:.1f}\n"
            f"Kapital: {capital:.2f} USD"
        )
        log.info(f"SHORT åbnet: {symbol} @ {price}")


def check_exits():
    global capital

    to_close = []
    for symbol, pos in positions.items():
        current = get_current_price(symbol)
        if current is None:
            continue

        entry     = pos["entry"]
        sl        = pos["sl"]
        tp        = pos["tp"]
        size      = pos["size"]
        direction = pos["direction"]
        name      = pos["name"]

        hit_tp = (direction == "LONG" and current >= tp) or (direction == "SHORT" and current <= tp)
        hit_sl = (direction == "LONG" and current <= sl) or (direction == "SHORT" and current >= sl)

        if hit_tp or hit_sl:
            pnl    = (current - entry) * size if direction == "LONG" else (entry - current) * size
            capital += (entry * size) + pnl
            reason  = "TP" if hit_tp else "SL"
            emoji   = "✅" if hit_tp else "❌"

            trade_log.append({"symbol": symbol, "pnl": pnl, "reason": reason})

            send_telegram(
                f"{emoji} <b>PAPER trade lukket — {reason}</b>\n"
                f"Aktie: {name} ({symbol})\n"
                f"Entry: {entry:.4f} → Exit: {current:.4f}\n"
                f"PnL: {'+' if pnl >= 0 else ''}{pnl:.2f} USD\n"
                f"Kapital: {capital:.2f} USD"
            )
            log.info(f"Lukket {symbol}: {reason} PnL={pnl:.2f}")
            to_close.append(symbol)

    for s in to_close:
        del positions[s]


def send_status():
    total_pnl    = sum(t["pnl"] for t in trade_log)
    wins         = sum(1 for t in trade_log if t["pnl"] > 0)
    total_trades = len(trade_log)
    wr           = f"{wins/total_trades*100:.1f}%" if total_trades > 0 else "N/A"

    open_pos = "\n  ingen" if not positions else ""
    for sym, p in positions.items():
        open_pos += f"\n  {sym}: {p['direction']} @ {p['entry']:.4f}"

    send_telegram(
        f"📊 <b>PAPER STATUS</b>\n"
        f"Kapital: {capital:.2f} USD\n"
        f"Total PnL: {'+' if total_pnl >= 0 else ''}{total_pnl:.2f} USD\n"
        f"Trades: {total_trades} | WR: {wr}\n"
        f"Åbne positioner:{open_pos}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    log.info("🚀 Stock Trading Bot scanner...")

    try:
        if positions:
            check_exits()

        for symbol, name in SYMBOLS.items():
            check_entry(symbol, name)
            time.sleep(13)  # Alpha Vantage: max 5 req/min på gratis tier

        counter_file = "/tmp/scan_counter.txt"
        try:
            with open(counter_file, "r") as f:
                counter = int(f.read().strip())
        except Exception:
            counter = 0

        counter += 1
        if counter >= 4:
            send_status()
            counter = 0

        with open(counter_file, "w") as f:
            f.write(str(counter))

        log.info(f"Scan færdig. Positioner: {len(positions)}")

    except Exception as e:
        log.error(f"Fejl i scan: {e}")


if __name__ == "__main__":
    main()
