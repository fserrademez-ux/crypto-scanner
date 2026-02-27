import ccxt
import pandas as pd
import time
import math
from datetime import datetime

# =========================
# CONFIG
# =========================
CONFIG = {
    "timeframe": "4h",
    "limit": 250,                 # candle count
    "max_pairs": 250,             # how many USDT pairs to scan
    "min_price": 0.0000001,       # ignore near-zero
    "min_quote_volume_usdt": 2_000_000,  # 24h quoteVolume filter (approx)
    "sleep_ms": 120,              # polite pacing
    "top_n": 10,                  # print top N
    "save_csv": True,
    "csv_name": "scan_results.csv",

    # Trend filters
    "ema_len": 200,
    "rsi_len": 14,
    "atr_len": 14,

    # Breakout rules
    "pivot_lookback": 60,         # to find pivot high/low levels
    "break_buffer_atr": 0.15,     # breakout needs close beyond level by X*ATR
    "max_atr_pct": 8.0,           # avoid too wild coins (ATR% too high)
    "min_vol_ratio": 1.2,         # volume confirmation: last vol / SMA(vol) >= ratio

    # Compression rules (B seçeneği)
    "compression_window": 24,     # last N candles range width check
    "compression_width_pct": 3.0, # if (hi-lo)/mid*100 <= this => compression
    "near_break_distance_atr": 0.35,  # if price near key level within X*ATR => "about to break"
}

# =========================
# INDICATORS
# =========================
def ema(series: pd.Series, n: int):
    return series.ewm(span=n, adjust=False).mean()

def rsi(close: pd.Series, n: int = 14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(n).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(n).mean()
    rs = gain / (loss.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, n: int = 14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def vol_sma(vol: pd.Series, n: int = 20):
    return vol.rolling(n).mean()

# =========================
# STRUCTURE / PATTERNS
# =========================
def detect_compression(df: pd.DataFrame, window: int, width_pct: float):
    """B seçeneği: sıkışma/range daralması"""
    if len(df) < window + 5:
        return False, None, None
    sub = df.tail(window)
    hi = float(sub["high"].max())
    lo = float(sub["low"].min())
    mid = (hi + lo) / 2 if (hi + lo) != 0 else 1.0
    width = (hi - lo) / mid * 100.0
    if width <= width_pct:
        return True, "compression", {"hi": hi, "lo": lo, "width_pct": width}
    return False, None, {"hi": hi, "lo": lo, "width_pct": width}

def pivot_levels(df: pd.DataFrame, lookback: int):
    """Simple pivot: last lookback highs/lows as key levels"""
    if len(df) < lookback + 5:
        lookback = max(30, len(df) - 10)
    sub = df.tail(lookback)
    pivot_hi = float(sub["high"].max())
    pivot_lo = float(sub["low"].min())
    return pivot_hi, pivot_lo

def channel_breakout(df: pd.DataFrame, window: int = 60):
    """
    Basit kanal: son window içindeki üst/alt bandı alıp
    son kapanışın dışarı taşıp taşmadığına bakar.
    """
    if len(df) < window + 5:
        return False, None, None
    sub = df.tail(window)
    hi = float(sub["high"].max())
    lo = float(sub["low"].min())
    last = df.iloc[-1]
    close = float(last["close"])
    return True, (hi, lo), close

# =========================
# SCORING
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def score_candidate(df: pd.DataFrame, cfg: dict, meta: dict):
    """
    meta: {'type':..., 'level':..., 'distance_atr':..., 'compression':..., 'vol_ratio':..., 'atr_pct':...}
    """
    last = df.iloc[-1]
    price = float(last["close"])
    ema200 = float(last["EMA200"])
    r = float(last["RSI"])
    atrv = float(last["ATR"])
    atrp = float(last["ATR_PCT"])
    vratio = float(last["VOL_RATIO"])

    # Basic gates
    if price <= cfg["min_price"]:
        return None
    if atrp > cfg["max_atr_pct"]:
        return None

    # Trend weight: prefer above EMA200
    trend_score = 10 if price >= ema200 else 0

    # RSI zone: prefer 45-70 for longs (not too overbought)
    rsi_score = 0
    if 45 <= r <= 70:
        rsi_score = 15
    elif 40 <= r < 45:
        rsi_score = 8
    elif 70 < r <= 78:
        rsi_score = 6

    # Volume confirmation
    vol_score = 0
    if vratio >= cfg["min_vol_ratio"]:
        vol_score = 15
    elif vratio >= 1.05:
        vol_score = 6

    # Structure / breakout
    struct_score = 0
    if meta["type"] in ("pivot_breakout_up", "range_breakout_up", "channel_breakout_up"):
        struct_score = 25
    elif meta["type"] in ("near_break_up",):
        struct_score = 18
    elif meta["type"] == "compression":
        struct_score = 14

    # Distance to level (closer is better for "about to break")
    dist_score = 0
    if meta.get("distance_atr") is not None:
        # distance_atr: 0 is perfect, 0.35 is edge of our "near" threshold
        d = float(meta["distance_atr"])
        dist_score = int(clamp((1 - d / cfg["near_break_distance_atr"]) * 12, 0, 12))

    # ATR sanity: medium ATR is ok, too low = sleepy, too high = risky
    atr_quality = 0
    if 0.7 <= atrp <= 3.5:
        atr_quality = 8
    elif 0.4 <= atrp < 0.7:
        atr_quality = 5
    elif 3.5 < atrp <= 6.0:
        atr_quality = 4

    total = trend_score + rsi_score + vol_score + struct_score + dist_score + atr_quality
    return int(total)

# =========================
# EXCHANGE / DATA
# =========================
def make_exchange():
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        },
        "timeout": 20000,
    })
    return exchange

def safe_fetch_ohlcv(exchange, symbol, timeframe, limit, max_retry=4):
    last_err = None
    for i in range(max_retry):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            last_err = e
            # backoff
            time.sleep(0.8 + i * 1.2)
    raise last_err

def safe_load_markets(exchange, max_retry=4):
    last_err = None
    for i in range(max_retry):
        try:
            return exchange.load_markets()
        except Exception as e:
            last_err = e
            time.sleep(0.8 + i * 1.2)
    raise last_err

# =========================
# MAIN SCAN LOGIC
# =========================
def analyze_symbol(df: pd.DataFrame, cfg: dict):
    # indicators
    df["EMA200"] = ema(df["close"], cfg["ema_len"])
    df["RSI"] = rsi(df["close"], cfg["rsi_len"])
    df["ATR"] = atr(df, cfg["atr_len"])
    df["ATR_PCT"] = (df["ATR"] / df["close"]) * 100
    df["VOL_SMA"] = vol_sma(df["volume"], 20)
    df["VOL_RATIO"] = df["volume"] / (df["VOL_SMA"].replace(0, 1e-12))

    last = df.iloc[-1]
    price = float(last["close"])
    atrv = float(last["ATR"]) if not math.isnan(float(last["ATR"])) else 0.0

    # Key levels
    piv_hi, piv_lo = pivot_levels(df, cfg["pivot_lookback"])

    # Breakout conditions (UP only for now)
    # pivot breakout up: close > piv_hi + buffer
    buf = cfg["break_buffer_atr"] * atrv
    breakout_up = price > (piv_hi + buf)

    # Near-break: within X*ATR below pivot high
    distance_atr = None
    near_break = False
    if atrv > 0:
        dist = (piv_hi - price) / atrv
        distance_atr = dist
        near_break = (0 <= dist <= cfg["near_break_distance_atr"])

    # Channel breakout (simple)
    ch_ok, ch_levels, ch_close = channel_breakout(df, window=cfg["pivot_lookback"])
    channel_type = None
    if ch_ok and ch_levels:
        ch_hi, ch_lo = ch_levels
        if price > (ch_hi + buf):
            channel_type = "channel_breakout_up"

    # Compression (B seçeneği)
    is_comp, comp_type, comp_info = detect_compression(
        df, window=cfg["compression_window"], width_pct=cfg["compression_width_pct"]
    )

    # Decide candidate type
    cand_type = None
    level = None

    if breakout_up:
        cand_type = "pivot_breakout_up"
        level = piv_hi
    elif channel_type == "channel_breakout_up":
        cand_type = "channel_breakout_up"
        level = ch_levels[0]
    elif near_break and is_comp:
        # strongest "about to break": near level + compression
        cand_type = "near_break_up"
        level = piv_hi
    elif is_comp:
        cand_type = "compression"
        level = comp_info["hi"] if comp_info else piv_hi
    elif near_break:
        cand_type = "near_break_up"
        level = piv_hi

    if cand_type is None:
        return None

    meta = {
        "type": cand_type,
        "level": level,
        "distance_atr": distance_atr if cand_type in ("near_break_up",) else None,
        "compression": is_comp,
        "compression_width_pct": comp_info["width_pct"] if comp_info else None,
    }

    sc = score_candidate(df, cfg, meta)
    if sc is None:
        return None

    out = {
        "symbol": None,  # fill later
        "score": sc,
        "type": cand_type,
        "price": price,
        "level": level,
        "atr_pct": float(last["ATR_PCT"]),
        "rsi": float(last["RSI"]),
        "vol_ratio": float(last["VOL_RATIO"]),
        "ema200": float(last["EMA200"]),
        "compression_width_pct": meta.get("compression_width_pct"),
    }
    return out

def main():
    print("BINANCE SPOT 4H BREAKOUT / SIKIŞMA Tarama Başladı...")
    exchange = make_exchange()

    err_summary = {"rate_limit_or_network": 0, "binance_block_or_conn": 0, "other_error": 0}
    results = []

    try:
        markets = safe_load_markets(exchange)
    except Exception as e:
        print("Marketleri yüklerken hata:", str(e))
        input("Kapatmak için Enter...")
        return

    # Build USDT spot active pairs
    symbols = []
    for s, m in markets.items():
        try:
            if "/USDT" in s and m.get("spot") and m.get("active"):
                # filter leveraged tokens
                if any(x in s for x in ["UP/", "DOWN/", "BULL/", "BEAR/"]):
                    continue
                symbols.append(s)
        except Exception:
            continue

    # Optional: sort for stability
    symbols = sorted(symbols)

    # Limit count
    symbols = symbols[: CONFIG["max_pairs"]]

    total = len(symbols)
    for idx, sym in enumerate(symbols, start=1):
        print(f"Taranıyor: {sym} ({idx}/{total})")

        try:
            # quick 24h volume filter (if available)
            try:
                t = exchange.fetch_ticker(sym)
                qv = t.get("quoteVolume", None)
                if qv is not None and float(qv) < CONFIG["min_quote_volume_usdt"]:
                    time.sleep(CONFIG["sleep_ms"] / 1000)
                    continue
            except Exception:
                # if ticker fails, still try candles
                pass

            ohlcv = safe_fetch_ohlcv(exchange, sym, CONFIG["timeframe"], CONFIG["limit"])
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            if len(df) < 120:
                time.sleep(CONFIG["sleep_ms"] / 1000)
                continue

            # Ensure numeric
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            res = analyze_symbol(df, CONFIG)
            if res:
                res["symbol"] = sym
                results.append(res)

        except ccxt.RateLimitExceeded:
            err_summary["rate_limit_or_network"] += 1
            time.sleep(2.0)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout):
            err_summary["rate_limit_or_network"] += 1
            time.sleep(1.5)
        except ccxt.ExchangeError:
            err_summary["binance_block_or_conn"] += 1
            time.sleep(1.0)
        except Exception:
            err_summary["other_error"] += 1

        time.sleep(CONFIG["sleep_ms"] / 1000)

    # Sort by score desc
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    print("\n" + "=" * 60)
    print(f"EN GÜÇLÜ ADAYLAR (Top {CONFIG['top_n']}):")
    if not results:
        print("Uygun aday bulunamadı.")
    else:
        top = results[: CONFIG["top_n"]]
        for i, r in enumerate(top, start=1):
            print(
                f"{i:02d}. {r['symbol']:12}  score={r['score']:3d}  "
                f"type={r['type']:<16}  rsi={r['rsi']:.1f}  atr%={r['atr_pct']:.2f}  volR={r['vol_ratio']:.2f}"
            )

    if CONFIG["save_csv"]:
        try:
            df_out = pd.DataFrame(results)
            df_out["ts"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            df_out.to_csv(CONFIG["csv_name"], index=False)
            print(f"\nCSV kaydedildi: {CONFIG['csv_name']}")
        except Exception as e:
            print("CSV yazma hatası:", str(e))

    print(f"\nHata özeti: {err_summary}")
    print("=" * 60)
    input("Bitirdi. Kapatmak için Enter...")

if __name__ == "__main__":
    main()
