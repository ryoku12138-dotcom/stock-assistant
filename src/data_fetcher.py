"""
AKShare data fetcher with retry logic and multi-source fallback.
Covers: K-line, real-time quotes, financials, announcements, fund flow.
"""
import os
import time
import akshare as ak
import pandas as pd
import requests
from typing import Optional
from src.config_helper import get_config

# Bypass system proxy to avoid ProxyError on Windows
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")
_orig_request = requests.Session.request


def _no_proxy_request(self, method, url, *args, **kwargs):
    kwargs.setdefault("proxies", {"http": None, "https": None})
    kwargs.setdefault("timeout", 30)
    return _orig_request(self, method, url, *args, **kwargs)


requests.Session.request = _no_proxy_request


config = get_config()
RETRY_COUNT = config["akshare"]["retry_count"]
RETRY_DELAY = config["akshare"]["retry_delay"]


def _get_market(stock_code: str) -> str:
    return "sh" if stock_code.startswith("6") else "sz"


def _to_full_code(stock_code: str) -> str:
    """Convert '000001' to 'sz000001' or '600519' to 'sh600519'."""
    return f"{_get_market(stock_code)}{stock_code}"


def _retry(func, *args, **kwargs):
    """Execute func with retry logic. Returns None on all failures."""
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"  [AKShare] attempt {attempt}/{RETRY_COUNT} failed: {e}")
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)
    print(f"  [AKShare] all {RETRY_COUNT} attempts failed, skipping.")
    return None


def _normalize_kline_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize column names to Chinese standard used by technical.py."""
    if source == "em":
        # Eastmoney: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额...
        return df
    elif source == "sina":
        # Sina: date, open, high, low, close, volume, amount...
        col_map = {
            "date": "日期", "open": "开盘", "high": "最高",
            "low": "最低", "close": "收盘", "volume": "成交量",
            "amount": "成交额"
        }
        existing = {k: v for k, v in col_map.items() if k in df.columns}
        return df.rename(columns=existing)
    elif source == "tx":
        # Tencent: date, open, close, high, low, amount
        col_map = {
            "date": "日期", "open": "开盘", "high": "最高",
            "low": "最低", "close": "收盘", "amount": "成交额"
        }
        existing = {k: v for k, v in col_map.items() if k in df.columns}
        return df.rename(columns=existing)
    return df


def get_kline_daily(stock_code: str, period: str = "daily") -> Optional[pd.DataFrame]:
    """
    Get historical daily K-line data with multi-source fallback.
    Sources: Eastmoney -> Sina -> Tencent
    """
    # 1. Eastmoney (richest data)
    df = _retry(ak.stock_zh_a_hist, symbol=stock_code, period=period, adjust="qfq")
    if df is not None and not df.empty:
        return _normalize_kline_columns(df, "em")

    # 2. Sina (good data, reliable)
    full_code = _to_full_code(stock_code)
    df = _retry(ak.stock_zh_a_daily, symbol=full_code, adjust="qfq")
    if df is not None and not df.empty:
        return _normalize_kline_columns(df, "sina")

    # 3. Tencent (basic data, most stable)
    df = _retry(ak.stock_zh_a_hist_tx, symbol=full_code)
    if df is not None and not df.empty:
        return _normalize_kline_columns(df, "tx")

    return None


def get_stock_info(stock_code: str) -> Optional[dict]:
    """Get individual stock basic info (name, market cap, PE, etc.)."""
    df = _retry(ak.stock_individual_info_em, symbol=stock_code)
    if df is not None and not df.empty:
        info = {}
        for _, row in df.iterrows():
            info[row["item"]] = row["value"]
        return info
    return None


def get_financial_data(stock_code: str) -> Optional[pd.DataFrame]:
    """Get financial abstract data (indicators as rows, dates as columns)."""
    return _retry(ak.stock_financial_abstract, symbol=stock_code)


def get_announcements(stock_code: str) -> Optional[pd.DataFrame]:
    """Get recent stock announcements."""
    return _retry(ak.stock_individual_notice_report, security=stock_code)


def get_fund_flow(stock_code: str) -> Optional[pd.DataFrame]:
    """Get individual stock fund flow data."""
    market = _get_market(stock_code)
    return _retry(ak.stock_individual_fund_flow, stock=stock_code, market=market)


def get_stock_name(stock_code: str) -> str:
    """Look up stock name from code. Falls back to code if lookup fails."""
    info = get_stock_info(stock_code)
    if info:
        for key in ["股票简称", "名称"]:
            if key in info:
                return str(info[key])
    return stock_code


def get_call_auction(stock_code: str, kline: pd.DataFrame = None) -> Optional[dict]:
    """
    Get call auction reference data using pre-market minute data.
    Returns auction volume, open price gap, and volume vs 5-day avg comparison.
    Falls back to daily K-line if minute data unavailable.
    """
    if kline is None:
        kline = get_kline_daily(stock_code)
    if kline is None or kline.empty:
        return None

    recent = kline.tail(5)
    avg_volume_5d = float(recent["成交量"].mean()) if "成交量" in recent.columns else 0

    auction_vol = None
    open_price = None
    prev_close = float(kline["收盘"].iloc[-2]) if len(kline) >= 2 and "收盘" in kline.columns else None

    # Try AKShare minute-level pre-market data (most recent trading day)
    try:
        df_min = _retry(ak.stock_zh_a_hist_pre_min_em, symbol=stock_code)
        if df_min is not None and not df_min.empty:
            time_col, vol_col, open_col = None, None, None
            for c in df_min.columns:
                cs = str(c)
                if "时间" in cs or "time" in cs.lower():
                    time_col = c
                elif "成交量" in cs or "volume" in cs.lower():
                    vol_col = c
                elif "开盘" in cs or "open" in cs.lower():
                    open_col = c

            if time_col and vol_col:
                auction_times = set(f"09:{m:02d}" for m in range(15, 26))
                mask = df_min[time_col].astype(str).str[:5].isin(auction_times)
                auction_rows = df_min[mask]
                if not auction_rows.empty:
                    auction_vol = pd.to_numeric(auction_rows[vol_col], errors="coerce").sum()
                    if open_col:
                        open_price = float(pd.to_numeric(auction_rows[open_col].iloc[-1], errors="coerce"))
    except Exception as e:
        print(f"  [CallAuction] minute fetch failed: {e}")

    # Fallback to daily K-line opening price
    if open_price is None and "开盘" in kline.columns:
        open_price = float(kline["开盘"].iloc[-1])

    result = {"open_price": open_price}

    if avg_volume_5d > 0 and auction_vol and auction_vol > 0:
        ratio = auction_vol / avg_volume_5d * 100
        if ratio > 25:
            result["volume_text"] = f"放量({ratio:.0f}%)"
        elif ratio < 8:
            result["volume_text"] = f"缩量({ratio:.0f}%)"
        else:
            result["volume_text"] = f"正常({ratio:.0f}%)"
    else:
        result["volume_text"] = "量能无数据"

    if open_price and prev_close and prev_close > 0:
        gap = (open_price - prev_close) / prev_close * 100
        if gap > 0.5:
            result["gap_text"] = f"高开{gap:+.2f}%"
        elif gap < -0.5:
            result["gap_text"] = f"低开{gap:+.2f}%"
        else:
            result["gap_text"] = "平开"
        result["gap_pct"] = round(gap, 2)
    else:
        result["gap_text"] = "无竞价数据"

    return result


if __name__ == "__main__":
    code = "000001"
    print(f"Testing data fetch for {code}...")

    name = get_stock_name(code)
    print(f"Name: {name}")

    info = get_stock_info(code)
    print(f"Info: {'OK' if info else 'FAILED'}")

    kline = get_kline_daily(code)
    if kline is not None:
        print(f"K-line: {len(kline)} rows, cols: {list(kline.columns)}")
        print(kline.tail(3))
    else:
        print("K-line: FAILED (all sources)")

    financial = get_financial_data(code)
    if financial is not None:
        print(f"Financial: {len(financial)} rows x {len(financial.columns)} cols")
    else:
        print("Financial: FAILED")

    flow = get_fund_flow(code)
    if flow is not None:
        print(f"Fund flow: {len(flow)} rows")
    else:
        print("Fund flow: FAILED")

    ann = get_announcements(code)
    if ann is not None:
        print(f"Announcements: {len(ann)} rows")
    else:
        print("Announcements: FAILED")
