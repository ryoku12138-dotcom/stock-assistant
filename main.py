"""
Stock Assistant — Main entry point.
Modes: morning (盘前预警), daily (收盘日报), weekly (周五周报)
"""
import os
import sys
import json
import re
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
    get_call_auction, is_trading_day,
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

# ── Infrastructure ────────────────────────────────────────────

def _is_test_mode() -> bool:
    return os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"

def _test_prefix() -> str:
    return "[测试] " if _is_test_mode() else ""

def _load_prev_signals() -> dict:
    """Load the most recent signal file within last 10 days."""
    bjt = now_bjt()
    for offset in range(1, 11):
        d = bjt - timedelta(days=offset)
        path = Path(f"logs/signals_{d.strftime('%Y-%m-%d')}.json")
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

def _save_signals(signals: dict):
    path = Path(f"logs/signals_{now_bjt().strftime('%Y-%m-%d')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(signals, f, ensure_ascii=False, indent=2)

def _log_push(stock_code: str, stock_name: str, signal: str, price=None,
              entry_price=None, stop_loss=None, target_price=None):
    entry = {
        "datetime": now_bjt().strftime("%Y-%m-%d %H:%M:%S"),
        "stock_code": stock_code, "stock_name": stock_name,
        "signal": signal, "price": price,
        "entry_price": entry_price, "stop_loss": stop_loss,
        "target_price": target_price, "test_mode": _is_test_mode(),
    }
    path = Path(f"logs/push_log_{now_bjt().strftime('%Y-%m-%d')}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

def _parse_ai_conclusion(ai_text: str) -> dict:
    m = re.search(
        r'结论[：:]\s*(买入|卖出|观望)\s*\|\s*评分\s*(\d+)\s*分[^|]*\|\s*置信度[：:]\s*(高|中|低)',
        ai_text
    )
    if m:
        return {"direction": m.group(1), "score": int(m.group(2)), "confidence": m.group(3)}
    return {}

def _signal_continuity(stock_code: str, direction: str) -> str:
    if direction in ("N/A", "观望", ""):
        return "watch"
    prev = _load_prev_signals()
    prev_d = prev.get(stock_code, {}).get("direction", "")
    return "confirmed" if prev_d == direction else "watch"

def _continuity_emoji(continuity: str, direction: str) -> str:
    if direction in ("买入", "BUY"):
        return "🟢" if continuity == "confirmed" else "🟡"
    elif direction in ("卖出", "SELL"):
        return "🔴" if continuity == "confirmed" else "🟡"
    return "⚪"

def _direction_cn(d: str) -> str:
    m = {"BUY": "买入", "SELL": "卖出", "HOLD": "观望"}
    return m.get(d, d)

def _morning_score(ma, macd, rsi, kdj, auction) -> int:
    s = 0
    t = ma.get("trend", "")
    if t == "bullish": s += 2
    elif t == "consolidation": s += 1
    elif t == "bearish": s -= 2
    ms = macd.get("signal", "")
    if ms == "golden_cross": s += 3
    elif ms == "bullish": s += 1
    elif ms == "bearish": s -= 1
    elif ms == "death_cross": s -= 3
    rs = rsi.get("signal", "")
    if rs == "oversold": s += 2
    elif rs == "overbought": s -= 2
    ks = kdj.get("signal", "")
    if ks in ("golden_cross_like", "oversold"): s += 1
    elif ks in ("death_cross_like", "overbought"): s -= 1
    if auction:
        g = auction.get("gap_pct", 0) or 0
        vt = auction.get("volume_text", "")
        if g > 0.5 and "放量" in vt: s += 2
        elif g < -0.5 and "放量" in vt: s -= 2
        elif g > 0.5: s += 1
        elif g < -0.5: s -= 1
    return s

def _morning_level(score: int) -> tuple:
    if score >= 5: return "🟢", "偏多"
    elif score >= 3: return "🟢", "谨慎偏多"
    elif score >= -2: return "🟡", "观望"
    elif score >= -4: return "🔴", "谨慎偏空"
    return "🔴", "偏空"


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
    """Morning pre-market alert: two-layer structure with call auction."""
    config = get_config()
    watchlist = config["watchlist"]
    notifier = FeishuNotifier()
    now = now_bjt()
    date_str = now.strftime("%Y-%m-%d")
    prev_signals = _load_prev_signals()

    title = f"{_test_prefix()}{date_str} 盘前预警（基于集合竞价数据）"
    blocks = [f"{title}\n"]

    today_signals = {}
    for i, entry in enumerate(watchlist, 1):
        code = entry["code"] if isinstance(entry, dict) else entry
        name = entry.get("name", code) if isinstance(entry, dict) else code
        print(f"  [{i}/{len(watchlist)}] Morning: {name}({code})...")

        try:
            result = analyze_single(code, name, full=False)
        except Exception as e:
            blocks.append(f"{name}({code}) 数据获取失败\n")
            notifier.send_text(title=f"{_test_prefix()}错误通知", content=f"盘前预警 {name}({code}) 数据获取失败: {e}")
            continue

        if result is None:
            blocks.append(f"{name}({code}) 数据获取失败\n")
            continue

        ind = result["indicators"]
        ma = ind.get("ma", {})
        macd = ind.get("macd", {})
        rsi = ind.get("rsi", {})
        kdj = ind.get("kdj", {})

        price = ma.get("current", None)
        auction = get_call_auction(code, kline=result["kline"])
        score = _morning_score(ma, macd, rsi, kdj, auction)
        emoji, level = _morning_level(score)

        # Continuity check
        direction = "买入" if score >= 3 else ("卖出" if score <= -4 else "观望")
        cont = _signal_continuity(code, direction)
        cont_emoji = _continuity_emoji(cont, direction)

        today_signals[code] = {"direction": direction, "score": score, "price": price}

        # ═══ Layer 1: Decision ═══
        rsi_val = rsi.get("rsi", "N/A")
        rsi_str = f"{rsi_val}({_cn(rsi.get('signal', ''))})" if rsi_val != "N/A" else "N/A"
        price_str = f"{price}元" if price else "N/A"

        # Entry/stop from auction & MA
        auction_gap = auction.get("gap_text", "") if auction else ""
        entry_hint = ""
        if score >= 3:
            entry_hint = f"竞价{auction_gap}，可小仓位试多" if auction_gap else "均线多头，回调至MA10可考虑"
            stop = f"跌破MA20（约{ma.get('ma20', 'N/A')}元）止损"
        elif score <= -4:
            entry_hint = "反弹乏力，不建议入场"
            stop = f"若持仓，反弹至MA5（约{ma.get('ma5', 'N/A')}元）考虑减仓"
        else:
            entry_hint = "信号不明确，继续观望"
            stop = "暂无明确止损位"

        layer1 = (
            f"{name}({code}) 现价{price_str}\n"
            f"{cont_emoji} {direction} | 评分{score}分 | {'连续确认' if cont == 'confirmed' else '首日关注'}\n"
            f"📌 今日操作参考（基于集合竞价数据）：{entry_hint}\n"
            f"止损参考：{stop}\n"
            f"建议仓位：{'轻仓' if 3 <= score <= 4 else ('标准仓' if score >= 5 else '观望')}"
        )

        # ═══ Layer 2: Technical Detail ═══
        details = []
        details.append(f"均线：{_cn(ma.get('trend', 'N/A'))}"
                       f"（简单说：{'短中长期趋势向上，整体偏强' if ma.get('trend') == 'bullish' else '短期趋势转弱，注意风险' if ma.get('trend') == 'bearish' else '方向不明确，等待选择方向'}）")
        details.append(f"MACD：{_cn(macd.get('signal', 'N/A'))}"
                       f"（简单说：{'上涨动能增强' if macd.get('signal') in ('golden_cross', 'bullish') else '下跌风险增加' if macd.get('signal') in ('death_cross', 'bearish') else '方向待确认'}）")
        details.append(f"RSI：{rsi_str}"
                       f"（简单说：{'超卖，可能反弹' if rsi.get('signal') == 'oversold' else '超买，追高风险' if rsi.get('signal') == 'overbought' else '处于正常区间'}）")
        if kdj.get('signal') not in ('neutral', 'insufficient_data', None):
            details.append(f"KDJ：{_cn(kdj.get('signal', ''))} K:{kdj.get('k','')} D:{kdj.get('d','')}"
                           f"（简单说：{'短线偏多' if kdj.get('signal') in ('golden_cross_like', 'oversold') else '短线偏空' if kdj.get('signal') in ('death_cross_like', 'overbought') else '中性'}）")
        if auction:
            details.append(f"竞价：{auction.get('volume_text', '')} | {auction.get('gap_text', '')}"
                           f"（简单说：{'早盘资金积极，竞价偏多' if auction.get('gap_pct', 0) and auction.get('gap_pct', 0) > 0 and '放量' in auction.get('volume_text', '') else '竞价阶段多空平衡' if auction.get('gap_text', '') == '平开' else '早盘偏谨慎，注意防守'}）")

        block = layer1 + "\n\n📊 技术细节\n" + "\n".join(f"- {d}" for d in details)
        blocks.append(block)
        time_mod.sleep(2)

    _save_signals(today_signals)
    content = "\n\n".join(blocks)
    content += "\n\n以上为盘前技术指标与竞价参考，完整分析请等待收盘日报。"
    notifier.send_text(title=title, content=content)
    print("Morning alert sent.")


def run_daily():
    """Close-of-day full analysis with AI, signal continuity, and logging."""
    config = get_config()
    watchlist = config["watchlist"]
    notifier = FeishuNotifier()
    now = now_bjt()
    date_str = now.strftime("%Y-%m-%d")
    title = f"{_test_prefix()}{date_str} 收盘日报"

    print(f"Daily report — {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Watchlist: {len(watchlist)} stocks")

    # Health check
    if not check_health():
        msg = "DeepSeek API 不可用，日报生成失败"
        print(f"[ERROR] {msg}")
        notifier.send_text(title=f"{_test_prefix()}错误通知", content=f"{date_str} {msg}")
        sys.exit(1)

    notifier.send_text(
        title=f"{title} 开始分析",
        content=f"时间: {now.strftime('%Y-%m-%d %H:%M')}\n自选股: {len(watchlist)} 只"
    )

    today_signals = {}
    success_count = 0
    failed_stocks = []

    for i, entry in enumerate(watchlist, 1):
        code = entry["code"] if isinstance(entry, dict) else entry
        name = entry.get("name", code) if isinstance(entry, dict) else code
        try:
            print(f"\n[{i}/{len(watchlist)}] {name}({code})...")
            result = analyze_single(code, name, full=True)
            if result is None:
                print(f"  [Skip] No K-line data")
                failed_stocks.append(f"{name}({code}): 无K线数据")
                continue

            ai_text = result.get("ai_result", "")
            parsed = _parse_ai_conclusion(ai_text)
            direction = parsed.get("direction", "")
            score = parsed.get("score", 0)
            cont = _signal_continuity(code, direction)

            # Determine display signal with continuity
            if cont == "confirmed" and direction in ("买入", "卖出"):
                display_signal = direction
                cont_label = "强信号（连续确认）"
            elif direction in ("买入", "卖出"):
                display_signal = "关注偏多" if direction == "买入" else "关注偏空"
                cont_label = "关注信号（首日出现，待明日确认）"
            else:
                display_signal = "观望"
                cont_label = "观望"

            today_signals[code] = {
                "direction": direction,
                "score": score,
                "confidence": parsed.get("confidence", ""),
                "price": result["indicators"].get("ma", {}).get("current"),
            }

            # Send to Feishu
            notifier.send_signal(
                stock_code=code, stock_name=name,
                signal=display_signal,
                ai_summary=ai_text,
                continuity=cont_label,
            )

            # Log push
            price = result["indicators"].get("ma", {}).get("current")
            _log_push(stock_code=code, stock_name=name,
                      signal=display_signal, price=price)

            success_count += 1
            print(f"  Signal: {display_signal} ({cont_label}) | Sent")
        except Exception as e:
            print(f"  [ERROR] {name}({code}): {e}")
            failed_stocks.append(f"{name}({code}): {e}")
            notifier.send_text(
                title=f"{_test_prefix()}错误通知",
                content=f"{date_str} 收盘日报 {name}({code}) 分析失败: {e}"
            )
            continue
        if i < len(watchlist):
            time_mod.sleep(3)

    _save_signals(today_signals)

    # Summary
    summary = (
        f"日报完成: {now.strftime('%Y-%m-%d %H:%M')}\n"
        f"总数: {len(watchlist)} | 成功: {success_count} | 失败: {len(watchlist) - success_count}"
    )
    if failed_stocks:
        summary += "\n失败明细:\n" + "\n".join(failed_stocks)
    notifier.send_text(title=f"{title} 完成", content=summary)
    print(f"\nDone. Success: {success_count}/{len(watchlist)}")


def run_weekly():
    """Weekly summary: aggregate all stocks, single AI-powered weekly report."""
    config = get_config()
    watchlist = config["watchlist"]
    notifier = FeishuNotifier()
    now = now_bjt()
    date_str = now.strftime("%Y-%m-%d")
    # Week range: Monday to Friday of current week
    mon = now - timedelta(days=now.weekday())
    fri = mon + timedelta(days=4)
    week_range = f"{mon.strftime('%Y-%m-%d')} ~ {fri.strftime('%Y-%m-%d')}"
    title = f"{_test_prefix()}{week_range} 周报"

    print(f"Weekly report — {now.strftime('%Y-%m-%d %H:%M')}")

    if not check_health():
        msg = "DeepSeek API 不可用，周报生成失败"
        print(f"[ERROR] {msg}")
        notifier.send_text(title=f"{_test_prefix()}错误通知", content=f"{date_str} {msg}")
        sys.exit(1)

    notifier.send_text(
        title=f"{title} 开始生成",
        content=f"时间: {now.strftime('%Y-%m-%d %H:%M')}\n正在汇总本周数据分析..."
    )

    # Collect all stock summaries
    summaries = []
    failed = []
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
                    f"MA趋势{_cn(ma.get('trend', 'N/A'))}, "
                    f"MACD{_cn(ind.get('macd', {}).get('signal', 'N/A'))}, "
                    f"RSI({ind.get('rsi', {}).get('rsi', 'N/A')})"
                    f"{', ' + change_pct if change_pct else ''}"
                )
            else:
                failed.append(f"{name}({code}): 无数据")
        except Exception as e:
            failed.append(f"{name}({code}): {e}")
        time_mod.sleep(2)

    if failed:
        notifier.send_text(
            title=f"{_test_prefix()}错误通知",
            content=f"周报部分股票数据获取失败:\n" + "\n".join(failed)
        )

    # Single AI-powered weekly summary
    stock_list = "\n".join(summaries) if summaries else "暂无数据"
    weekly_prompt = f"""你是A股投资分析专家。请根据以下自选股本周技术指标汇总，生成一份周报。不要使用###等Markdown标题。

【自选股周度汇总】
{stock_list}

请按以下格式输出：
📊 本周市场概况
整体判断（偏多/偏空/震荡），2-3句话说明本周整体走势

⭐ 个股亮点
挑出1-3只表现最好或信号最积极的股票，简述理由

⚠️ 风险关注
挑出1-2只需要警惕的股票，简述风险

📅 下周展望
简要操作建议

注意：以上分析仅供参考，不构成投资建议。"""

    print("  Generating weekly AI summary...")
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
        weekly_report = f"AI分析失败: {e}"
        notifier.send_text(
            title=f"{_test_prefix()}错误通知",
            content=f"周报AI生成失败: {e}"
        )

    notifier.send_text(title=title, content=weekly_report)
    notifier.send_text(
        title=f"{title} 完成",
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
    test_note = " [TEST MODE]" if _is_test_mode() else ""
    print(f"  Test Mode: {_is_test_mode()}{test_note}")
    print("=" * 60)

    # Skip non-trading days for automatic triggers (allow manual test override)
    if not is_trading_day():
        msg = "今天非A股交易日，跳过推送"
        print(msg)
        if not _is_test_mode():
            return
        print("  [Test mode] 强制运行（忽略交易日检查）")

    if args.mode == "morning":
        run_morning()
    elif args.mode == "daily":
        run_daily()
    elif args.mode == "weekly":
        run_weekly()


if __name__ == "__main__":
    main()
