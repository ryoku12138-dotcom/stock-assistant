"""
Technical analysis module.
Indicators: MA(5/10/20), MACD, RSI, KDJ, Bollinger Bands.
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()
INDICATOR_CONFIG = config["indicators"]


def calc_ma(df: pd.DataFrame) -> dict:
    """Calculate MA5, MA10, MA20."""
    result = {}
    periods = INDICATOR_CONFIG["ma_periods"]
    close = df["收盘"].values if "收盘" in df.columns else df["close"].values
    for p in periods:
        if len(close) >= p:
            result[f"ma{p}"] = round(float(np.mean(close[-p:])), 2)
        else:
            result[f"ma{p}"] = None
    current_price = float(close[-1])
    result["current"] = round(current_price, 2)
    result["trend"] = _ma_trend(close)
    return result


def _ma_trend(close: np.ndarray) -> str:
    """Determine MA trend: bullish (MA5>MA10>MA20) or bearish."""
    if len(close) < 20:
        return "insufficient_data"
    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:])
    ma20 = np.mean(close[-20:])
    if ma5 > ma10 > ma20:
        return "bullish"
    elif ma5 < ma10 < ma20:
        return "bearish"
    else:
        return "consolidation"


def calc_macd(df: pd.DataFrame) -> dict:
    """Calculate MACD (12, 26, 9)."""
    cfg = INDICATOR_CONFIG["macd"]
    close = df["收盘"].values if "收盘" in df.columns else df["close"].values
    if len(close) < cfg["slow"] + cfg["signal"]:
        return {"diff": None, "dea": None, "macd": None, "signal": "insufficient_data"}

    ema_fast = _ema(close, cfg["fast"])
    ema_slow = _ema(close, cfg["slow"])
    diff = ema_fast - ema_slow
    dea = _ema(diff, cfg["signal"])
    macd_bar = 2 * (diff - dea)

    latest_diff = round(float(diff[-1]), 4)
    latest_dea = round(float(dea[-1]), 4)
    latest_macd = round(float(macd_bar[-1]), 4)

    # Signal
    if len(diff) >= 2:
        if diff[-2] < dea[-2] and diff[-1] > dea[-1]:
            sig = "golden_cross"
        elif diff[-2] > dea[-2] and diff[-1] < dea[-1]:
            sig = "death_cross"
        elif latest_diff > latest_dea:
            sig = "bullish"
        else:
            sig = "bearish"
    else:
        sig = "neutral"

    return {"diff": latest_diff, "dea": latest_dea, "macd": latest_macd, "signal": sig}


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    result = np.zeros(len(data))
    result[0] = data[0]
    multiplier = 2 / (period + 1)
    for i in range(1, len(data)):
        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def calc_rsi(df: pd.DataFrame) -> dict:
    """Calculate RSI (14)."""
    period = INDICATOR_CONFIG["rsi"]["period"]
    close = df["收盘"].values if "收盘" in df.columns else df["close"].values
    if len(close) < period + 1:
        return {"rsi": None, "signal": "insufficient_data"}

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        rsi_val = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_val = 100 - (100 / (1 + rs))

    rsi_val = round(float(rsi_val), 2)
    if rsi_val > 80:
        sig = "overbought"
    elif rsi_val < 20:
        sig = "oversold"
    else:
        sig = "neutral"

    return {"rsi": rsi_val, "signal": sig}


def calc_kdj(df: pd.DataFrame) -> dict:
    """Calculate KDJ (9, 3, 3)."""
    cfg = INDICATOR_CONFIG["kdj"]
    n = cfg["k_period"]
    high = df["最高"].values if "最高" in df.columns else df["high"].values
    low = df["最低"].values if "最低" in df.columns else df["low"].values
    close = df["收盘"].values if "收盘" in df.columns else df["close"].values

    if len(close) < n:
        return {"k": None, "d": None, "j": None, "signal": "insufficient_data"}

    k_values = np.zeros(len(close))
    d_values = np.zeros(len(close))

    for i in range(n - 1, len(close)):
        highest = np.max(high[i - n + 1:i + 1])
        lowest = np.min(low[i - n + 1:i + 1])
        if highest != lowest:
            rsv = (close[i] - lowest) / (highest - lowest) * 100
        else:
            rsv = 50.0
        if i == n - 1:
            k_values[i] = rsv  # first K = RSV
            d_values[i] = rsv  # first D = RSV
        else:
            k_values[i] = 2 / 3 * k_values[i - 1] + 1 / 3 * rsv
            d_values[i] = 2 / 3 * d_values[i - 1] + 1 / 3 * k_values[i]

    j_values = 3 * k_values - 2 * d_values

    k = round(float(k_values[-1]), 2)
    d = round(float(d_values[-1]), 2)
    j = round(float(j_values[-1]), 2)

    if k > 80 and d > 80:
        sig = "overbought"
    elif k < 20 and d < 20:
        sig = "oversold"
    elif k > d and j > k:
        sig = "golden_cross_like"
    elif k < d and j < k:
        sig = "death_cross_like"
    else:
        sig = "neutral"

    return {"k": k, "d": d, "j": j, "signal": sig}


def calc_bollinger(df: pd.DataFrame) -> dict:
    """Calculate Bollinger Bands (20, 2)."""
    period = INDICATOR_CONFIG["bollinger"]["period"]
    std_dev = INDICATOR_CONFIG["bollinger"]["std_dev"]
    close = df["收盘"].values if "收盘" in df.columns else df["close"].values

    if len(close) < period:
        return {"upper": None, "middle": None, "lower": None, "signal": "insufficient_data"}

    middle = np.mean(close[-period:])
    std = np.std(close[-period:])
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    current = close[-1]

    upper = round(float(upper), 2)
    middle = round(float(middle), 2)
    lower = round(float(lower), 2)

    if current >= upper:
        sig = "above_upper"
    elif current <= lower:
        sig = "below_lower"
    elif current > middle:
        sig = "above_middle"
    else:
        sig = "below_middle"

    return {"upper": upper, "middle": middle, "lower": lower, "current": round(float(current), 2), "signal": sig}


def analyze(df: pd.DataFrame) -> dict:
    """Run all technical indicators on a DataFrame and return summary."""
    return {
        "ma": calc_ma(df),
        "macd": calc_macd(df),
        "rsi": calc_rsi(df),
        "kdj": calc_kdj(df),
        "bollinger": calc_bollinger(df),
    }


def indicators_to_text(ind: dict) -> str:
    """Convert indicator dict to human-readable text."""
    lines = []
    ma = ind.get("ma", {})
    if ma:
        parts = [f"现价: {ma.get('current', 'N/A')}"]
        for p in [5, 10, 20]:
            v = ma.get(f"ma{p}")
            if v is not None:
                parts.append(f"MA{p}: {v}")
        parts.append(f"趋势: {ma.get('trend', 'N/A')}")
        lines.append(" | ".join(parts))

    macd = ind.get("macd", {})
    if macd:
        lines.append(
            f"MACD — DIF: {macd.get('diff')}, DEA: {macd.get('dea')}, "
            f"柱: {macd.get('macd')}, 信号: {macd.get('signal')}"
        )

    rsi = ind.get("rsi", {})
    if rsi:
        lines.append(f"RSI(14): {rsi.get('rsi')} ({rsi.get('signal')})")

    kdj = ind.get("kdj", {})
    if kdj:
        lines.append(f"KDJ — K: {kdj.get('k')}, D: {kdj.get('d')}, J: {kdj.get('j')} ({kdj.get('signal')})")

    boll = ind.get("bollinger", {})
    if boll:
        lines.append(
            f"布林带 — 上轨: {boll.get('upper')}, 中轨: {boll.get('middle')}, "
            f"下轨: {boll.get('lower')}, 信号: {boll.get('signal')}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from data_fetcher import get_kline_daily

    code = "000001"
    print(f"Testing technical analysis for {code}...")
    df = get_kline_daily(code)
    if df is not None:
        result = analyze(df)
        print(indicators_to_text(result))
    else:
        print("Failed to fetch data")
