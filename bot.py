import os
import time
import logging
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

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

# ── Trading parametre ─────────────────────────────────────────────────────────
SYMBOLS = {
    "NVDA":  "NVIDIA",
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "CL=F":  "US Oil",
    "GC=F":  "Guld",
}

CAPITAL          = 10_000.0   # Start kapital USD
RISK_PER_TRADE   = 0.02       # 2% af kapital per trade
MAX_POSITIONS    = 3          # Max åbne positioner
ATR_SL_MULT      = 1.5        # SL = 1.5x ATR
ATR_TP_MULT      = 3.0        # TP = 3.0x ATR (R:R = 1:2)
RSI_PERIOD       = 14
RSI_OVERSOLD     = 35
RSI_OVERBOUGHT   = 65
SCAN_INTERVAL    = 900        # 15 min mellem scans
NEWS_MIN_SCORE   = 0.15       # Min sentiment score for signal

# ── State ─────────────────────────────────────────────────────────────────────
positions   = {}   # { symbol: {entry, sl, tp, size, direction, pnl} }
trade_log   = []   # Alle lukkede trades
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
# MARKEDSDATA
# ═════════════════════════════════════════════════════════════════════════════
def get_price_data(symbol: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame | None:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or len(df) < 50:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception as e:
        log.error(f"Pris-data fejl {symbol}: {e}")
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
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
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

    close = df["Close"].squeeze()
    rsi   = calc_rsi(close, RSI_PERIOD)
    macd, macd_signal = calc_macd(close)
    atr   = calc_atr(df)

    last_rsi    = float(rsi.iloc[-1])
    last_macd   = float(macd.iloc[-1])
    last_signal = float(macd_signal.iloc[-1])
    last_atr    = float(atr.iloc[-1])
    last_price  = float(close.iloc[-1])

    # Trend: EMA50 vs EMA200
    ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    uptrend = ema50 > ema200

    return {
        "price":      last_price,
        "rsi":        last_rsi,
        "macd":       last_macd,
        "macd_sig":   last_signal,
        "atr":        last_atr,
        "uptrend":    uptrend,
        "ema50":      ema50,
        "ema200":     ema200,
    }


# ═════════════════════════════════════════════════════════════════════════════
# NYHEDER & SENTIMENT
# ═════════════════════════════════════════════════════════════════════════════
def get_news_sentiment(symbol: str) -> float:
    """Returnerer sentiment score -1.0 til +1.0"""
    # Finnhub bruger ticker uden =F suffix
    ticker = symbol.replace("=F", "")
    # Map til Finnhub-venlige symboler
    ticker_map = {"CL": "USO", "GC": "GLD"}
    ticker = ticker_map.get(ticker, ticker)

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
            headline = article.get("headline", "")
            summary  = article.get("summary", "")
            text     = f"{headline}. {summary}"
            score    = analyzer.polarity_scores(text)["compound"]
            scores.append(score)

        return float(np.mean(scores)) if scores else 0.0

    except Exception as e:
        log.error(f"Nyheder fejl {symbol}: {e}")
        return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# POSITION SIZING
# ═════════════════════════════════════════════════════════════════════════════
def calc_position_size(price: float, atr: float) -> tuple[float, float, float]:
    """Returnerer (size, sl_price, tp_price) for LONG"""
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

    # ── LONG signal ───────────────────────────────────────────────────────────
    long_signal = (
        rsi < RSI_OVERSOLD and          # RSI oversolgt
        macd > macd_sig and             # MACD bullish crossover
        uptrend and                     # Over EMA200
        sentiment > NEWS_MIN_SCORE      # Positiv nyhedssentiment
    )

    # ── SHORT signal (kun stocks der understøtter det) ────────────────────────
    short_signal = (
        rsi > RSI_OVERBOUGHT and        # RSI overkøbt
        macd < macd_sig and             # MACD bearish crossover
        not uptrend and                 # Under EMA200
        sentiment < -NEWS_MIN_SCORE     # Negativ nyhedssentiment
    )

    if long_signal:
        size, sl, tp = calc_position_size(price, atr)
        cost = size * price
        if cost > capital:
            log.info(f"Ikke nok kapital til {symbol}")
            return

        positions[symbol] = {
            "entry":     price,
            "sl":        sl,
            "tp":        tp,
            "size":      size,
            "direction": "LONG",
            "name":      name,
            "opened_at": datetime.now().isoformat(),
        }
        capital -= cost

        msg = (
            f"🟢 <b>PAPER LONG åbnet</b>\n"
            f"Aktie: {name} ({symbol})\n"
            f"Pris: {price:.4f}\n"
            f"SL: {sl:.4f} | TP: {tp:.4f}\n"
            f"Størrelse: {size:.4f} enheder\n"
            f"Sentiment: {'positiv' if sentiment > 0 else 'neutral'}\n"
            f"RSI: {rsi:.1f} | MACD: {'bullish' if macd > macd_sig else 'bearish'}\n"
            f"Kapital: {capital:.2f} USD"
        )
        send_telegram(msg)
        log.info(f"LONG åbnet: {symbol} @ {price}")

    elif short_signal:
        size, sl_dist, _ = calc_position_size(price, atr)
        sl = price + (atr * ATR_SL_MULT)
        tp = price - (atr * ATR_SL_MULT * ATR_TP_MULT)

        positions[symbol] = {
            "entry":     price,
            "sl":        sl,
            "tp":        tp,
            "size":      size,
            "direction": "SHORT",
            "name":      name,
            "opened_at": datetime.now().isoformat(),
        }

        msg = (
            f"🔴 <b>PAPER SHORT åbnet</b>\n"
            f"Aktie: {name} ({symbol})\n"
            f"Pris: {price:.4f}\n"
            f"SL: {sl:.4f} | TP: {tp:.4f}\n"
            f"Størrelse: {size:.4f} enheder\n"
            f"Sentiment: negativ\n"
            f"RSI: {rsi:.1f} | MACD: {'bearish'}\n"
            f"Kapital: {capital:.2f} USD"
        )
        send_telegram(msg)
        log.info(f"SHORT åbnet: {symbol} @ {price}")


def check_exits():
    global capital

    to_close = []
    for symbol, pos in positions.items():
        try:
            ticker = yf.Ticker(symbol)
            current = ticker.fast_info.get("last_price") or ticker.fast_info.last_price
            if current is None:
                continue
            current = float(current)
        except Exception:
            continue

        entry     = pos["entry"]
        sl        = pos["sl"]
        tp        = pos["tp"]
        size      = pos["size"]
        direction = pos["direction"]
        name      = pos["name"]

        hit_tp = (direction == "LONG"  and current >= tp) or (direction == "SHORT" and current <= tp)
        hit_sl = (direction == "LONG"  and current <= sl) or (direction == "SHORT" and current >= sl)

        if hit_tp or hit_sl:
            if direction == "LONG":
                pnl = (current - entry) * size
            else:
                pnl = (entry - current) * size

            capital += (entry * size) + pnl
            reason   = "TP" if hit_tp else "SL"
            emoji    = "✅" if hit_tp else "❌"

            trade_log.append({
                "symbol":    symbol,
                "direction": direction,
                "entry":     entry,
                "exit":      current,
                "pnl":       pnl,
                "reason":    reason,
                "closed_at": datetime.now().isoformat(),
            })

            msg = (
                f"{emoji} <b>PAPER trade lukket — {reason}</b>\n"
                f"Aktie: {name} ({symbol})\n"
                f"Entry: {entry:.4f} → Exit: {current:.4f}\n"
                f"PnL: {'+' if pnl >= 0 else ''}{pnl:.2f} USD\n"
                f"Kapital: {capital:.2f} USD"
            )
            send_telegram(msg)
            log.info(f"Lukket {symbol}: {reason} PnL={pnl:.2f}")
            to_close.append(symbol)

    for s in to_close:
        del positions[s]


def send_status():
    total_pnl  = sum(t["pnl"] for t in trade_log)
    wins       = sum(1 for t in trade_log if t["pnl"] > 0)
    total_trades = len(trade_log)
    wr         = f"{wins/total_trades*100:.1f}%" if total_trades > 0 else "N/A"

    open_pos = ""
    if positions:
        for sym, p in positions.items():
            open_pos += f"\n  {sym}: {p['direction']} @ {p['entry']:.4f}"
    else:
        open_pos = "\n  ingen"

    msg = (
        f"📊 <b>PAPER STATUS</b>\n"
        f"Kapital: {capital:.2f} USD\n"
        f"Total PnL: {'+' if total_pnl >= 0 else ''}{total_pnl:.2f} USD\n"
        f"Trades: {total_trades} | WR: {wr}\n"
        f"Åbne positioner:{open_pos}"
    )
    send_telegram(msg)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════
def main():
    log.info("🚀 Stock Trading Bot starter...")
    send_telegram(
        "🤖 <b>Stock Trading Bot er online!</b>\n"
        f"Kører {len(SYMBOLS)} aktier: {', '.join(SYMBOLS.keys())}\n"
        f"Kapital: {capital:.2f} USD | Risk/trade: {RISK_PER_TRADE*100:.0f}%"
    )

    status_counter = 0

    while True:
        try:
            now = datetime.now()

            # Scan kun i markedstimer (14:30-21:00 UTC = USA markedet)
            # Commodities handler næsten 24/7 — scan altid for dem
            market_open = (now.hour >= 14 and now.minute >= 30) or now.hour in range(15, 21)

            # Check exits altid
            if positions:
                check_exits()

            # Check entries
            for symbol, name in SYMBOLS.items():
                is_commodity = symbol.endswith("=F")
                if market_open or is_commodity:
                    check_entry(symbol, name)
                    time.sleep(2)  # Rate limit

            # Send status hver 4. scan (ca. hver time)
            status_counter += 1
            if status_counter >= 4:
                send_status()
                status_counter = 0

            log.info(f"Scan færdig. Næste scan om {SCAN_INTERVAL//60} min. Positioner: {len(positions)}")
            time.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            log.info("Bot stoppet manuelt")
            send_telegram("🛑 <b>Bot stoppet</b>")
            break
        except Exception as e:
            log.error(f"Uventet fejl i main loop: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()