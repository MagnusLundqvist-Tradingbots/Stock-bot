import os
import json
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
    handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")]
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

CAPITAL          = 10_000.0
RISK_PER_TRADE   = 0.02
MAX_POSITIONS    = 3
ATR_SL_MULT      = 1.5
ATR_TP_MULT      = 3.0
RSI_PERIOD       = 14
RSI_OVERSOLD     = 35
RSI_OVERBOUGHT   = 65
NEWS_MIN_SCORE   = 0.15

STATE_FILE = "state.json"

analyzer = SentimentIntensityAnalyzer()

# ═════════════════════════════════════════════════════════════════════════════
# STATE PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════
def load_state():
    """Loader state fra JSON fil"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Kunne ikke loade state: {e}")
    return {
        "capital": CAPITAL,
        "positions": {},
        "trade_log": [],
        "scan_counter": 0,
        "last_scan": None
    }


def save_state(state):
    """Gemmer state til JSON fil"""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        log.error(f"Kunne ikke gemme state: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ═════════════════════════════════════════════════════════════════════════════
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram ikke konfigureret")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
        if resp.status_code != 200:
            log.warning(f"Telegram fejl: {resp.status_code} - {resp.text}")
    except Exception as e:
        log.error(f"Telegram fejl: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# MARKEDSDATA via yfinance (gratis, ingen rate limits)
# ═════════════════════════════════════════════════════════════════════════════
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    log.warning("yfinance ikke installeret - bruger fallback")


def get_price_data(symbol: str) -> pd.DataFrame | None:
    """Henter pris-data via yfinance (gratis, ingen API key)"""
    if not YFINANCE_AVAILABLE:
        log.error("yfinance ikke tilgængelig")
        return None

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="60d", interval="1h")

        if df is None or len(df) < 50:
            log.warning(f"For få datapunkter for {symbol}: {len(df) if df is not None else 0}")
            return None

        df = df.reset_index()

        # Standardiser kolonne navne
        rename_map = {}
        for col in df.columns:
            c = str(col).lower()
            if "open" in c:
                rename_map[col] = "Open"
            elif "high" in c:
                rename_map[col] = "High"
            elif "low" in c:
                rename_map[col] = "Low"
            elif "close" in c:
                rename_map[col] = "Close"
            elif "volume" in c:
                rename_map[col] = "Volume"
            elif "datetime" in c or "date" in c:
                rename_map[col] = "Datetime"

        df = df.rename(columns=rename_map)

        if not all(c in df.columns for c in ["Open", "High", "Low", "Close", "Volume", "Datetime"]):
            log.error(f"Manglende kolonner for {symbol}: {df.columns.tolist()}")
            return None

        return df.sort_values("Datetime").reset_index(drop=True)

    except Exception as e:
        log.error(f"Pris-data fejl {symbol}: {e}")
        return None


def get_current_price(symbol: str) -> float | None:
    """Henter nuværende pris via yfinance"""
    if not YFINANCE_AVAILABLE:
        return None
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        return float(info.last_price) if info.last_price else None
    except Exception as e:
        log.error(f"Nuværende pris fejl {symbol}: {e}")
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
    """Henter nyheder fra Finnhub og beregner sentiment"""
    if not FINNHUB_KEY:
        return 0.0

    # Map futures til underliggende for nyheder
    ticker_map = {"CL=F": "CL", "GC=F": "GC"}
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
def calc_position_size(price: float, atr: float, available_capital: float) -> tuple[float, float, float]:
    sl_dist  = atr * ATR_SL_MULT
    risk_usd = available_capital * RISK_PER_TRADE
    size     = risk_usd / sl_dist if sl_dist > 0 else 0
    sl_price = price - sl_dist
    tp_price = price + (sl_dist * ATR_TP_MULT)
    return round(size, 4), round(sl_price, 4), round(tp_price, 4)


# ═════════════════════════════════════════════════════════════════════════════
# TRADING LOGIK
# ═════════════════════════════════════════════════════════════════════════════
def check_entry(symbol: str, name: str, state: dict):
    if symbol in state["positions"]:
        return
    if len(state["positions"]) >= MAX_POSITIONS:
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

    # Beregn hvor meget kapital der er bundet i åbne positioner
    tied_capital = sum(p["size"] * p["entry"] for p in state["positions"].values())
    available = state["capital"] - tied_capital

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
        size, sl, tp = calc_position_size(price, atr, available)
        cost = size * price

        if cost > available or size <= 0:
            log.info(f"Ikke nok kapital til LONG {symbol}")
            return

        state["positions"][symbol] = {
            "entry": price, "sl": sl, "tp": tp,
            "size": size, "direction": "LONG", "name": name,
            "opened_at": datetime.now().isoformat()
        }
        state["capital"] -= cost

        send_telegram(
            f"🟢 <b>PAPER LONG åbnet</b>\n"
            f"Aktie: {name} ({symbol})\n"
            f"Pris: {price:.4f}\n"
            f"SL: {sl:.4f} | TP: {tp:.4f}\n"
            f"Sentiment: {sentiment:.2f} | RSI: {rsi:.1f}\n"
            f"Kapital: {state['capital']:.2f} USD"
        )
        log.info(f"LONG åbnet: {symbol} @ {price}")

    elif short_signal:
        size, sl, tp = calc_position_size(price, atr, available)
        # For SHORT: SL er over entry, TP er under entry
        sl_short = price + (atr * ATR_SL_MULT)
        tp_short = price - (atr * ATR_SL_MULT * ATR_TP_MULT)

        # Margin requirement (100% for sikkerhed)
        margin = size * price

        if margin > available or size <= 0:
            log.info(f"Ikke nok kapital til SHORT {symbol}")
            return

        state["positions"][symbol] = {
            "entry": price, "sl": sl_short, "tp": tp_short,
            "size": size, "direction": "SHORT", "name": name,
            "opened_at": datetime.now().isoformat()
        }
        state["capital"] -= margin  # Reserver margin

        send_telegram(
            f"🔴 <b>PAPER SHORT åbnet</b>\n"
            f"Aktie: {name} ({symbol})\n"
            f"Pris: {price:.4f}\n"
            f"SL: {sl_short:.4f} | TP: {tp_short:.4f}\n"
            f"Sentiment: {sentiment:.2f} | RSI: {rsi:.1f}\n"
            f"Kapital: {state['capital']:.2f} USD"
        )
        log.info(f"SHORT åbnet: {symbol} @ {price}")


def check_exits(state: dict):
    to_close = []

    for symbol, pos in state["positions"].items():
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
            if direction == "LONG":
                pnl = (current - entry) * size
                # Returner initial investering + PnL
                state["capital"] += (entry * size) + pnl
            else:  # SHORT
                pnl = (entry - current) * size
                # Returner margin + PnL
                state["capital"] += (entry * size) + pnl

            reason  = "TP" if hit_tp else "SL"
            emoji   = "✅" if hit_tp else "❌"

            state["trade_log"].append({
                "symbol": symbol,
                "direction": direction,
                "pnl": round(pnl, 2),
                "reason": reason,
                "entry": entry,
                "exit": current,
                "closed_at": datetime.now().isoformat()
            })

            send_telegram(
                f"{emoji} <b>PAPER trade lukket — {reason}</b>\n"
                f"Aktie: {name} ({symbol})\n"
                f"Entry: {entry:.4f} → Exit: {current:.4f}\n"
                f"PnL: {'+' if pnl >= 0 else ''}{pnl:.2f} USD\n"
                f"Kapital: {state['capital']:.2f} USD"
            )
            log.info(f"Lukket {symbol}: {reason} PnL={pnl:.2f}")
            to_close.append(symbol)

    for s in to_close:
        del state["positions"][s]


def send_status(state: dict):
    total_pnl    = sum(t["pnl"] for t in state["trade_log"])
    wins         = sum(1 for t in state["trade_log"] if t["pnl"] > 0)
    total_trades = len(state["trade_log"])
    wr           = f"{wins/total_trades*100:.1f}%" if total_trades > 0 else "N/A"

    open_pos = "\n  ingen" if not state["positions"] else ""
    for sym, p in state["positions"].items():
        open_pos += f"\n  {sym}: {p['direction']} @ {p['entry']:.4f}"

    send_telegram(
        f"📊 <b>PAPER STATUS</b>\n"
        f"Kapital: {state['capital']:.2f} USD\n"
        f"Total PnL: {'+' if total_pnl >= 0 else ''}{total_pnl:.2f} USD\n"
        f"Trades: {total_trades} | WR: {wr}\n"
        f"Åbne positioner:{open_pos}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    state = load_state()
    log.info(f"🚀 Stock Trading Bot starter... Kapital: {state['capital']:.2f}")

    try:
        # Tjek exits først
        if state["positions"]:
            check_exits(state)

        # Tjek entries
        for symbol, name in SYMBOLS.items():
            check_entry(symbol, name, state)
            time.sleep(1)  # Undgå at overbelaste yfinance

        # Send status hver 4. scan
        state["scan_counter"] = state.get("scan_counter", 0) + 1
        if state["scan_counter"] >= 4:
            send_status(state)
            state["scan_counter"] = 0

        state["last_scan"] = datetime.now().isoformat()
        save_state(state)

        log.info(f"Scan færdig. Positioner: {len(state['positions'])}")

    except Exception as e:
        log.error(f"Fejl i scan: {e}")
        save_state(state)  # Gem state selvom der var fejl


if __name__ == "__main__":
    main()
