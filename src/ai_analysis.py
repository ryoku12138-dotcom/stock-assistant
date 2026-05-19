"""
AI analysis module using DeepSeek V3 API.
Generates buy/sell/hold recommendations based on technical indicators and fundamentals.
"""
from openai import OpenAI
from src.config_helper import get_config


DS_CONFIG = get_config()["deepseek"]

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=DS_CONFIG["api_key"],
            base_url=DS_CONFIG["base_url"],
        )
    return _client


def build_prompt(stock_code: str, stock_name: str, indicators_text: str,
                 financial_text: str = "", fund_flow_text: str = "",
                 announcements_text: str = "") -> str:
    prompt = f"""你是一位专业的A股技术分析专家。请根据以下信息，对股票 {stock_name}({stock_code}) 进行全面分析。

【技术指标】
{indicators_text}

【财务数据】
{financial_text if financial_text else "暂无财务数据"}

【资金流向】
{fund_flow_text if fund_flow_text else "暂无资金流向数据"}

【近期公告】
{announcements_text if announcements_text else "暂无公告数据"}

请严格按照以下两层结构输出分析结果。不要使用 ### 等Markdown标题符号（飞书不支持渲染）。每只股票只输出下面这一个模板，不要额外开场白或结尾语。

══════ 第一层：决策信号 ══════

{stock_name}({stock_code})
🔴/🟢/🟡 结论：买入/观望/卖出 | 评分X分（满分10分）| 置信度：高/中/低
📌 明日操作建议（基于今日收盘数据）：一句话说清楚明天该怎么做
入场条件：突破XX元可买入 / 跌破XX元减仓
止损位：XX元（理由：为什么设在这里）
目标价：XX元（理由：技术阻力位或估值目标）
建议仓位：轻仓/标准仓/观望

══════ 第二层：技术支撑 ══════

📊 核心信号
（只列出显著信号，每条必须包含专业术语和括号大白话解释。格式：）
- 信号描述（简单说：用通俗语言解释这个信号对股价意味着什么）

📰 近期事件
（如无相关内容则写"近期无重大事件"）
- 事件描述（影响：利好/利空，对股价的潜在影响）

⚠️ 风险评估
- 最大回撤预估：若跌破XX元，下一支撑在XX元，最大回撤约X%
- （其他风险因素，每条带括号解释）

🌐 大盘与板块
- 今日大盘：市场情绪（偏多/偏空/震荡）（简单说：当前市场环境如何）
- 资金面：主力资金态度（简单说：大资金在进还是出）
- 操作环境评估：当前环境对操作是友好/中性/不利

注意：以上分析仅供参考，不构成投资建议。每条技术信号后面必须用（简单说：...）解释其含义。"""
    return prompt


def analyze_stock(stock_code: str, stock_name: str, indicators_text: str,
                  financial_text: str = "", fund_flow_text: str = "",
                  announcements_text: str = "") -> str:
    """Send analysis request to DeepSeek and return response."""
    client = _get_client()
    prompt = build_prompt(stock_code, stock_name, indicators_text,
                          financial_text, fund_flow_text, announcements_text)

    try:
        response = client.chat.completions.create(
            model=DS_CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=DS_CONFIG["max_tokens"],
            temperature=DS_CONFIG["temperature"],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [DeepSeek] API call failed: {e}")
        return f"AI分析失败: {e}"


def check_health() -> bool:
    """Quick health check for DeepSeek API connectivity."""
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=DS_CONFIG["model"],
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10,
            timeout=15,
        )
        content = response.choices[0].message.content
        return bool(content)
    except Exception as e:
        print(f"  [DeepSeek] Health check failed: {e}")
        return False


if __name__ == "__main__":
    print("Checking DeepSeek API health...")
    healthy = check_health()
    print(f"API health: {'OK' if healthy else 'FAILED'}")

    if healthy:
        text = analyze_stock(
            stock_code="000001",
            stock_name="平安银行",
            indicators_text="MA趋势: bullish\nRSI: 45 (neutral)\nMACD: golden_cross\nKDJ: 55 (neutral)",
        )
        print("\n--- AI Analysis ---")
        print(text)
