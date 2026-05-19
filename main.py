"""
Stock Assistant — Main entry point.
Modes: morning (盘前预警), daily (收盘日报), weekly (周五周报)
"""
import os
import sys
import argparse
import time as time_mod
from pathlib import Path
from datetime import datetime, timedelta, timezone, tzinfo

# Beijing timezone (UTC+8)
_BJT = timezone(timedelta(hours=8))


def now_bjt() -> datetime:
    """Return current Beijing time (UTC+8)."""
    return datetime.now(_BJT)

sys.path.insert(0, str(Path(__file__).parent))

from src.config_helper import get_config
from src.data_fetcher import (
    get_kline_daily, get_financial_data,
    get_fund_flow, get_announcements,
    get_call_auction,
)
from src.technical import analyze, indicators_to_text
from src.ai_analysis import analyze_stock, check_health
from src.feishu import FeishuNotifier
import pandas as pd


def fetch_financial_text(stock_code: str) -> str:
    fin_df = get_financial_data(stock_code)
    if fin_df is None or fin_df.empty:
        return ""
    indicator_col = fin_df.columns[1] if len(fin_df.columns) > 1 else fin_df.columns[0]
    date_col = fin_df.columns[2] if len(fin_df.columns) > 2 else None
    if not date_col:
        return ""
    text = ""
    key_indicators = ["净资产收益率", "净利润率", "毛利率", "营业总收入", "归属净利润"]
    for _, row in fin_df.iterrows():
        name = str(row[indicator_col])
        if any(k in name for k in key_indicators):
            val = row[date_col]
            if pd.notna(val):
                text += f"{name}: {val}; "
    return text


def fetch_fund_flow_text(stock_code: str) -> str:
    flow_df = get_fund_flow(stock_code)
    if flow_df is None or flow_df.empty:
        return ""
    recent = flow_df.tail(5)
    flow_cols = [c for c in recent.columns if "流入" in str(c) or "流出" in str(c)]
    text = ""
    for col in flow_cols[:2]:
        try:
            total = pd.to_numeric(recent[col], errors="coerce").sum()
            text += f"近5日{col}: {total / 1e8:.2f}亿; "
        except Exception:
            pass
    return text


def fetch_announcements_text(stock_code: str) -> str:
    ann_df = get_announcements(stock_code)
    if ann_df is None or ann_df.empty:
        return ""
    recent = ann_df.head(3)
    title_col = None
    for c in recent.columns:
        if "标题" in str(c) or "title" in str(c).lower():
            title_col = c
            break
    if title_col:
        return "\n".join([f"- {t}" for t in recent[title_col].tolist()])
    return ""


def analyze_single(stock_code: str, stock_name: str, full: bool = True) -> dict:
    """Analyze one stock. Returns result dict or None on failure."""
    df = get_kline_daily(stock_code)
    if df is None or df.empty:
        return None

    indicators = analyze(df)
    indicators_str = indicators_to_text(indicators)

    result = {
        "code": stock_code,
        "name": stock_name,
        "indicators": indicators,
        "indicators_str": indicators_str,
        "kline": df,
    }

    if full:
        financial_text = fetch_financial_text(stock_code)
        fund_flow_text = fetch_fund_flow_text(stock_code)
        announcements_text = fetch_announcements_text(stock_code)

        ai_result = analyze_stock(
            stock_code, stock_name, indicators_str,
            financial_text, fund_flow_text, announcements_text
        )
        result["ai_result"] = ai_result
        result["signal"] = _extract_signal(ai_result)
        result["financial_text"] = financial_text
        result["fund_flow_text"] = fund_flow_text
        result["announcements_text"] = announcements_text

    return result


def _extract_signal(ai_text: str) -> str:
    if not ai_text:
        return "N/A"
    text_lower = ai_text.lower()
    if "买入" in text_lower or "buy" in text_lower or "增持" in text_lower:
        return "BUY"
    elif "卖出" in text_lower or "sell" in text_lower or "减持" in text_lower:
        return "SELL"
    return "HOLD"


# ======================== Modes ========================

# Signal translation map (English -> Chinese)
_SIGNAL_CN = {
    "bullish": "多头",
    "bearish": "空头",
    "consolidation": "震荡整理",
    "golden_cross": "金叉",
    "death_cross": "死叉",
    "golden_cross_like": "类金叉",
    "death_cross_like": "类死叉",
    "neutral": "中性",
    "overbought": "超买",
    "oversold": "超卖",
    "above_upper": "触及上轨",
    "below_lower": "触及下轨",
    "above_middle": "中轨上方",
    "below_middle": "中轨下方",
    "insufficient_data": "数据不足",
}


def _cn(v: str) -> str:
    """Translate signal value to Chinese, pass through if unknown."""
    return _SIGNAL_CN.get(v, v)


def _key_signal(ma: dict, macd: dict, rsi: dict, kdj: dict) -> str:
    """Produce a one-line summary of the most noteworthy signal."""
    # Priority: cross signals > extreme RSI > trend alignment
    if macd.get("signal") == "golden_cross":
        return "MACD金叉，短期看涨动能增强，关注能否站稳均线"
    if macd.get("signal") == "death_cross":
        return "MACD死叉，短期回调风险增加，注意防守"
    if rsi.get("signal") == "oversold":
        return f"RSI超卖（{rsi.get('rsi')}），短期或有技术性反弹机会"
    if rsi.get("signal") == "overbought":
        return f"RSI超买（{rsi.get('rsi')}），追高风险较大，等待回调"
    if kdj.get("signal") == "golden_cross_like":
        return "KDJ金叉形态，短线偏多但需确认"
    if kdj.get("signal") == "death_cross_like":
        return "KDJ死叉形态，短线偏空注意风险"
    if kdj.get("signal") == "overbought":
        return f"KDJ超买（K:{kdj.get('k')} D:{kdj.get('d')}），短期过热"
    if kdj.get("signal") == "oversold":
        return f"KDJ超卖（K:{kdj.get('k')} D:{kdj.get('d')}），存在超跌反弹动能"
    if ma.get("trend") == "bullish":
        return "均线多头排列，上升趋势延续中"
    if ma.get("trend") == "bearish":
        return "均线空头排列，下行趋势未改，等待企稳"
    if ma.get("trend") == "consolidation":
        return "均线缠绕，处于震荡整理区间，方向不明"
    return "暂无明确信号，建议观望等待方向选择"


def run_morning():
    """Morning pre-market alert: indicators + call auction, no AI."""
    config = get_config()
    watchlist = config["watchlist"]
    notifier = FeishuNotifier()
    now = now_bjt()

    blocks = [f"盘前预警 | {now.strftime('%Y-%m-%d %H:%M')}\n"]

    for i, entry in enumerate(watchlist, 1):
        code = entry["code"] if isinstance(entry, dict) else entry
        name = entry.get("name", code) if isinstance(entry, dict) else code
        print(f"  [{i}/{len(watchlist)}] Morning: {name}({code})...")

        result = analyze_single(code, name, full=False)
        if result is None:
            blocks.append(f"{name}({code}) 数据获取失败\n")
            continue

        ind = result["indicators"]
        ma = ind.get("ma", {})
        macd = ind.get("macd", {})
        rsi = ind.get("rsi", {})
        kdj = ind.get("kdj", {})

        current_price = ma.get("current", "N/A")
        rsi_val = rsi.get("rsi", "N/A")
        rsi_str = f"{rsi_val}({_cn(rsi.get('signal', ''))})" if rsi_val != "N/A" else "N/A"

        block = (
            f"{name}({code}) 现价{current_price}元\n"
            f"均线：{_cn(ma.get('trend', 'N/A'))}"
            f" | MACD：{_cn(macd.get('signal', 'N/A'))}"
            f" | RSI：{rsi_str}\n"
            f"信号：{_key_signal(ma, macd, rsi, kdj)}\n"
        )

        # Call auction data
        auction = get_call_auction(code, kline=result["kline"])
        if auction:
            block += f"竞价参考：{auction.get('volume_text', '')} | {auction.get('gap_text', '')}\n"

        blocks.append(block)
        time_mod.sleep(2)

    content = "\n".join(blocks)
    content += "\n以上为盘前技术指标与竞价参考，完整分析请等待收盘日报。"
    notifier.send_text(title="盘前预警", content=content)
    print("Morning alert sent.")


def run_daily():
    """Close-of-day full analysis with AI."""
    config = get_config()
    watchlist = config["watchlist"]
    notifier = FeishuNotifier()
    now = now_bjt()

    print(f"Daily report — {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Watchlist: {len(watchlist)} stocks")

    if not check_health():
        print("[ERROR] DeepSeek API unavailable, exiting.")
        sys.exit(1)

    notifier.send_text(
        title="收盘日报 开始分析",
        content=f"时间: {now.strftime('%Y-%m-%d %H:%M')}\n自选股: {len(watchlist)} 只"
    )

    success_count = 0
    for i, entry in enumerate(watchlist, 1):
        code = entry["code"] if isinstance(entry, dict) else entry
        name = entry.get("name", code) if isinstance(entry, dict) else code
        try:
            print(f"\n[{i}/{len(watchlist)}] {name}({code})...")
            result = analyze_single(code, name, full=True)
            if result is None:
                print(f"  [Skip] No K-line data")
                continue
            notifier.send_signal(
                stock_code=code, stock_name=name,
                signal=result.get("signal", "N/A"),
                ai_summary=result.get("ai_result", "N/A")
            )
            success_count += 1
            print(f"  Signal: {result.get('signal', 'N/A')} | Sent")
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
        if i < len(watchlist):
            time_mod.sleep(3)

    summary = (
        f"日报完成: {now.strftime('%Y-%m-%d %H:%M')}\n"
        f"总数: {len(watchlist)} | 成功: {success_count} | 失败: {len(watchlist) - success_count}"
    )
    notifier.send_text(title="收盘日报 完成", content=summary)
    print(f"\nDone. Success: {success_count}/{len(watchlist)}")


def run_weekly():
    """Weekly summary: aggregate all stocks, single AI-powered weekly report."""
    config = get_config()
    watchlist = config["watchlist"]
    notifier = FeishuNotifier()
    now = now_bjt()

    print(f"Weekly report — {now.strftime('%Y-%m-%d %H:%M')}")

    if not check_health():
        print("[ERROR] DeepSeek API unavailable, exiting.")
        sys.exit(1)

    notifier.send_text(
        title="周报 开始生成",
        content=f"时间: {now.strftime('%Y-%m-%d %H:%M')}\n正在汇总本周数据分析..."
    )

    # Collect all stock summaries
    summaries = []
    for entry in watchlist:
        code = entry["code"] if isinstance(entry, dict) else entry
        name = entry.get("name", code) if isinstance(entry, dict) else code
        print(f"  Weekly: {name}({code})...")
        try:
            result = analyze_single(code, name, full=False)
            if result:
                ind = result["indicators"]
                ma = ind.get("ma", {})
                change_pct = ""
                if ma.get("current") and ma.get("ma5"):
                    week_change = (ma["current"] - ma["ma5"]) / ma["ma5"] * 100
                    change_pct = f"周涨跌: {week_change:+.2f}%"
                summaries.append(
                    f"{name}({code}): 现价{ma.get('current', 'N/A')}, "
                    f"MA趋势{ma.get('trend', 'N/A')}, "
                    f"MACD{ind.get('macd', {}).get('signal', 'N/A')}, "
                    f"RSI({ind.get('rsi', {}).get('rsi', 'N/A')})"
                    f"{', ' + change_pct if change_pct else ''}"
                )
        except Exception as e:
            summaries.append(f"{name}({code}): 获取失败 ({e})")
        time_mod.sleep(2)

    # Single AI-powered weekly summary
    stock_list = "\n".join(summaries)
    weekly_prompt = f"""你是A股投资分析专家。请根据以下自选股本周技术指标汇总，生成一份周报。

【自选股周度汇总】
{stock_list}

请按以下格式输出：
1. **本周市场概况**：整体判断（偏多/偏空/震荡），2-3句话
2. **个股亮点**：挑出1-3只表现最好或信号最积极的股票，简述理由
3. **风险关注**：挑出1-2只需要警惕的股票，简述风险
4. **下周展望**：简要操作建议

注意：以上分析仅供参考，不构成投资建议。"""

    print("  Generating weekly AI summary...")
    weekly_ai = analyze_stock(
        stock_code="WEEKLY", stock_name="自选股组合",
        indicators_text=stock_list,
        financial_text="", fund_flow_text="", announcements_text=""
    )

    # Build a custom prompt for weekly (override the default)
    from src.ai_analysis import _get_client, DS_CONFIG
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=DS_CONFIG["model"],
            messages=[{"role": "user", "content": weekly_prompt}],
            max_tokens=DS_CONFIG["max_tokens"],
            temperature=DS_CONFIG["temperature"],
        )
        weekly_report = response.choices[0].message.content
    except Exception as e:
        weekly_report = weekly_ai or f"AI分析失败: {e}"

    notifier.send_text(
        title="周报",
        content=f"**自选股周报 | {now.strftime('%Y-%m-%d')}**\n\n{weekly_report}"
    )
    notifier.send_text(
        title="周报 完成",
        content=f"时间: {now.strftime('%Y-%m-%d %H:%M')}\n已生成周报并推送"
    )
    print("Weekly report sent.")


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(description="Stock Assistant")
    parser.add_argument(
        "--mode", choices=["morning", "daily", "weekly"],
        default="daily",
        help="Analysis mode (default: daily)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Stock Assistant — {args.mode.upper()} MODE")
    print(f"  Time: {now_bjt().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if args.mode == "morning":
        run_morning()
    elif args.mode == "daily":
        run_daily()
    elif args.mode == "weekly":
        run_weekly()


if __name__ == "__main__":
    main()
