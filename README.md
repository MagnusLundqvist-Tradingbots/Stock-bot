# 📈 Stock Trading Bot

Paper trading bot til aktier og commodities med RSI, MACD, ATR og nyhedssentiment.

## Aktiver
- NVDA (NVIDIA)
- AAPL (Apple)
- MSFT (Microsoft)
- CL=F (US Oil)
- GC=F (Guld)

## Strategi
- **RSI (14):** Entry når oversolgt (<35) eller overkøbt (>65)
- **MACD:** Bekræfter trend-retning
- **EMA 50/200:** Trend-filter
- **ATR:** Dynamisk SL og TP (R:R = 1:2)
- **Nyhedssentiment:** VADER + Finnhub som filter

## Risk Management
- 2% af kapital per trade
- Max 3 åbne positioner
- SL = 1.5x ATR | TP = 3.0x ATR

---

## 🚀 Deploy til Render.com

### 1. GitHub
```bash
git init
git add .
git commit -m "Stock bot - første version"
git remote add origin https://MagnusLundqvist-Tradingbots:DIN_TOKEN@github.com/MagnusLundqvist-Tradingbots/Stock-bot.git
git push -u origin main
```

### 2. Render.com
1. Gå til render.com og log ind
2. New → Blueprint (finder render.yaml automatisk)
3. Tilslut dit GitHub repo
4. Tilføj environment variables:
   - `TELEGRAM_TOKEN` = din bot token
   - `TELEGRAM_CHAT_ID` = dit chat ID
   - `FINNHUB_KEY` = din Finnhub API nøgle
5. Deploy!

### 3. Environment Variables
```
TELEGRAM_TOKEN=8720250424:AAE6lsyec73qzSZLH-5_Svbcjck4DCvprdo
TELEGRAM_CHAT_ID=7973320871
FINNHUB_KEY=d8iqgfpr01qtcvnt81s0d8iqgfpr01qtcvnt81sg
```

---

## 📱 Telegram beskeder
- 🟢 LONG åbnet
- 🔴 SHORT åbnet  
- ✅ Trade lukket — TP
- ❌ Trade lukket — SL
- 📊 Status (hver time)