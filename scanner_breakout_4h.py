import ccxt
import pandas as pd
import numpy as np
import time
from collections import Counter

# =========================
# CONFIG (kolay ayar)
# =========================
CONFIG = {
    "timeframe": "4h",
    "limit": 260,

    # Universe
    "max_symbols": 250,
    "min_bars": 220,

    # Market filters (kalite)
    "exclude_bases": {"USDT", "USDC", "FDUSD", "TUSD", "DAI", "BUSD"},
    "min_price": 0.000001,   # mikro fiyatlı çöp elemek için
    "min_volume_usdt_approx": 0,  # spotta 24h volume için ayrı endpoint gerek; burada 0 bırakıyoruz.

    # Strategy filters
    "ema_trend": 200,
    "rsi_len": 14,
    "adx_len": 14,
    "atr_len": 14,

    # Breakout logic
    "pivot_left": 2,
    "pivot_right": 2,
    "pivot_lookback": 80,     # son kaç bar içinde pivotlardan direnç çıkaralım
    "breakout_atr_mult": 0.25, # close > R + (ATR * mult) (fake breakout azaltma)
    "min_vol_ratio": 1.3,      # son hacim / 20 SMA hacim
    "vol_sma_len": 20,

    # Extra quality filters
    "rsi_min": 48,
    "rsi_max": 72,
    "adx_min": 14,
    "atrp_min": 0.6,          # ATR% çok düşükse (ölü piyasa) kırılım zayıf olabilir
    "atrp_max": 8.0,          # ATR% çok yüksekse noise olabilir

    # Scoring weights (toplam 100 civarı)
    "w_trend": 20,
    "w_breakout": 25,
    "w_volume": 15,
    "w_momentum": 15,
    "w_adx": 15,
    "w_structure": 10,  # formasyon / yapı
}

# =========================
# Indicators
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill")

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def wilder_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = low.shift() - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sm = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_sm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_sm = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (plus_sm / atr_sm)
    minus_di = 100 * (minus_sm / atr_sm)

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

# =========================
# Pivots & structures
# =========================
def pivot_idxs(series: pd.Series, left: int, right: int, mode: str):
    vals = series.values
    idxs = []
    for i in range(left, len(vals) - right):
        L = vals[i-left:i]
        R = vals[i+1:i+1+right]
        if mode == "high":
            if vals[i] >= np.max(L) and vals[i] >= np.max(R):
                idxs.append(i)
        else:
            if vals[i] <= np.min(L) and vals[i] <= np.min(R):
                idxs.append(i)
    return idxs

def last_pivot_resistance(df: pd.DataFrame, lookback: int, left: int, right: int):
    sub = df.iloc[-lookback:].copy()
    idxs = pivot_idxs(sub["high"], left, right, "high")
    if not idxs:
        return None
    # en son pivot high
    i = idxs[-1]
    return float(sub["high"].iloc[i])

def last_pivot_support(df: pd.DataFrame, lookback: int, left: int, right: int):
    sub = df.iloc[-lookback:].copy()
    idxs = pivot_idxs(sub["low"], left, right, "low")
    if not idxs:
        return None
    i = idxs[-1]
    return float(sub["low"].iloc[i])

def detect_triangle(df: pd.DataFrame, lookback: int, left: int, right: int):
    sub = df.iloc[-lookback:].copy()
    ph = pivot_idxs(sub["high"], left, right, "high")
    pl = pivot_idxs(sub["low"], left, right, "low")
    if len(ph) < 3 or len(pl) < 3:
        return False

    highs = [float(sub["high"].iloc[i]) for i in ph[-3:]]
    lows  = [float(sub["low"].iloc[i])  for i in pl[-3:]]

    lower_highs = highs[0] > highs[1] > highs[2]
    higher_lows = lows[0] < lows[1] < lows[2]
    return bool(lower_highs and higher_lows)

def detect_range(df: pd.DataFrame, window: int = 60):
    sub = df.iloc[-window:].copy()
    hi = sub["high"].max()
    lo = sub["low"].min()
    mid = (hi + lo) / 2
    width_pct = (hi - lo) / mid * 100 if mid else 999
    # 60 bar içinde bant çok dar ise range
    return width_pct < 8.0

def detect_flag_channel(df: pd.DataFrame, impulse_bars: int = 12, cons_bars: int = 30):
    # Basit: önce güçlü yükseliş (impuls), sonra dar bant (konsolidasyon)
    if len(df) < impulse_bars + cons_bars + 5:
        return False

    imp = df.iloc[-(impulse_bars + cons_bars):-cons_bars]
    cons = df.iloc[-cons_bars:]

    imp_move = (imp["close"].iloc[-1] - imp["open"].iloc[0]) / imp["open"].iloc[0] * 100
    cons_hi = cons["high"].max()
    cons_lo = cons["low"].min()
    cons_mid = (cons_hi + cons_lo) / 2
    cons_width = (cons_hi - cons_lo) / cons_mid * 100 if cons_mid else 999

    return (imp_move > 6.0) and (cons_width < 6.0)

# =========================
# Scoring
# =========================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def score_candidate(df: pd.DataFrame, R: float, S: float, cfg=CONFIG):
    last = df.iloc[-1]
    price = float(last["close"])
    ema200 = float(last["EMA"])
    r = float(last["RSI"])
    a = float(last["ADX"])
    atrv = float(last["ATR"])
    atrp = float(last["ATR_PCT"])
    vol_ratio = float(last["VOL_RATIO"])

    # Filter gates
    if price <= cfg["min_price"]:
        return None

    if not (price > ema200):
        return None

    if not (cfg["rsi_min"] <= r <= cfg["rsi_max"]):
        return None

    if not (a >= cfg["adx_min"]):
        return None

    if not (cfg["atrp_min"] <= atrp <= cfg["atrp_max"]):
        return None

    if R is None:
        return None

    breakout_thresh = R + (atrv * cfg["breakout_atr_mult"])
    is_breakout = price > breakout_thresh

    if not is_breakout:
        return None

    if vol_ratio < cfg["min_vol_ratio"]:
        return None

    # Scores (0..weight)
    trend_score = clamp(((price - ema200) / ema200) * 100, 0, 6) / 6 * cfg["w_trend"]

    # breakout ne kadar güçlü (R üstünde ne kadar)
    breakout_strength_pct = ((price - R) / R) * 100 if R else 0
    breakout_score = clamp(breakout_strength_pct, 0, 4) / 4 * cfg["w_breakout"]

    volume_score = clamp(vol_ratio, 1.0, 3.0)
    volume_score = (volume_score - 1.0) / (3.0 - 1.0) * cfg["w_volume"]

    # momentum: RSI 60 civarı idealdir
    momentum_score = (1 - (abs(60 - r) / 20))
    momentum_score = clamp(momentum_score, 0, 1) * cfg["w_momentum"]

    adx_score = clamp(a, cfg["adx_min"], 35)
    adx_score = (adx_score - cfg["adx_min"]) / (35 - cfg["adx_min"]) * cfg["w_adx"]

    # Structure: triangle / flag / range etiketi (kırılımla birlikte)
    tri = detect_triangle(df, lookback=80, left=cfg["pivot_left"], right=cfg["pivot_right"])
    flag = detect_flag_channel(df)
    rng = detect_range(df, window=60)

    structure_score = 0
    tags = []
    if tri:
        structure_score += 0.8 * cfg["w_structure"]; tags.append("triangle_break")
    if flag:
        structure_score += 0.8 * cfg["w_structure"]; tags.append("flag/channel_break")
    if rng:
        tags.append("range/ara-alan")

    structure_score = clamp(structure_score, 0, cfg["w_structure"])

    total = trend_score + breakout_score + volume_score + momentum_score + adx_score + structure_score

    # Risk info: nearest support estimate
    dist_to_support = None
    if S is not None and S > 0:
        dist_to_support = (price - S) / price * 100

    return {
        "price": price,
        "R": R,
        "S": S,
        "EMA200": ema200,
        "RSI": r,
        "ADX": a,
        "ATR": atrv,
        "ATR%": atrp,
        "VOL_RATIO": vol_ratio,
        "score": round(total, 2),
        "tags": ",".join(tags) if tags else "breakout"
        ,
        "dist_to_S_%": None if dist_to_support is None else round(dist_to_support, 2),
        "breakout_above_R_%": round(((price - R) / R) * 100, 2) if R else None
    }

# =========================
# Main
# =========================
def main():
    print("BINANCE SPOT 4H BREAKOUT Tarama Başladı...\n")

    exchange = ccxt.binance({"enableRateLimit": True})
    markets = exchange.load_markets()

    def is_ok(symbol: str) -> bool:
        m = markets.get(symbol, {})
        if "/USDT" not in symbol:
            return False
        if not m.get("spot", False) or not m.get("active", False):
            return False
        if any(x in symbol for x in ("UP/", "DOWN/", "BULL/", "BEAR/")):
            return False
        base = symbol.split("/")[0]
        if base in CONFIG["exclude_bases"]:
            return False
        return True

    symbols = [s for s in markets.keys() if is_ok(s)]
    symbols = symbols[: CONFIG["max_symbols"]]

    results = []
    errors = Counter()

    for sym in symbols:
        print("Taranıyor:", sym)
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe=CONFIG["timeframe"], limit=CONFIG["limit"])
            if not ohlcv or len(ohlcv) < CONFIG["min_bars"]:
                errors["insufficient_data"] += 1
                continue

            df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"]).astype(float)

            # indicators
            df["EMA"] = ema(df["close"], CONFIG["ema_trend"])
            df["RSI"] = rsi(df["close"], CONFIG["rsi_len"])
            df["ADX"] = wilder_adx(df, CONFIG["adx_len"])
            df["ATR"] = atr(df, CONFIG["atr_len"])
            df["ATR_PCT"] = (df["ATR"] / df["close"]) * 100

            df["VOL_SMA"] = df["volume"].rolling(CONFIG["vol_sma_len"]).mean()
            df["VOL_RATIO"] = (df["volume"] / df["VOL_SMA"]).replace([np.inf, -np.inf], np.nan).fillna(0)

            # pivots
            R = last_pivot_resistance(df, CONFIG["pivot_lookback"], CONFIG["pivot_left"], CONFIG["pivot_right"])
            S = last_pivot_support(df, CONFIG["pivot_lookback"], CONFIG["pivot_left"], CONFIG["pivot_right"])

            cand = score_candidate(df, R, S, CONFIG)
            if cand:
                cand["coin"] = sym
                results.append(cand)

            time.sleep(0.25)

        except ccxt.BaseError:
            errors["ccxt_error"] += 1
        except Exception:
            errors["other_error"] += 1

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    print("\nEN GÜÇLÜ BREAKOUT ADAYLARI (Top 15):\n")
    for r in results[:15]:
        print(f"{r['coin']} | SCORE: {r['score']} | Price: {r['price']}")
        print(f"R: {r['R']} | S: {r['S']} | breakout%: {r['breakout_above_R_%']} | distS%: {r['dist_to_S_%']}")
        print(f"RSI: {r['RSI']} | ADX: {r['ADX']} | ATR%: {r['ATR%']} | VOLx: {r['VOL_RATIO']}")
        print(f"TAGS: {r['tags']}")
        print("-"*40)

    if results:
        best = results[0]
        print("\n✅ EN YÜKSEK SKORLU ADAY:")
        print(best)

        # CSV export
        out = pd.DataFrame(results)
        out.to_csv("breakout_results_4h.csv", index=False)
        print("\nCSV kaydedildi: breakout_results_4h.csv")

    if errors:
        print("\nHata özeti:", dict(errors))

if __name__ == "__main__":
    main()
