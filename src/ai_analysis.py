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
    prompt = f"""你是一位专业的A股技术分析专家。请根据以下信息，对股票 {stock_name}({stock_code}) 进行分析，给出买入/持有/卖出的建议。

【技术指标】
{indicators_text}

【财务数据】
{financial_text if financial_text else "暂无财务数据"}

【资金流向】
{fund_flow_text if fund_flow_text else "暂无资金流向数据"}

【近期公告】
{announcements_text if announcements_text else "暂无公告数据"}

请按以下格式回答：
1. **综合评分**（满分10分）
2. **操作建议**：买入 / 持有 / 卖出
3. **置信度**：高 / 中 / 低
4. **分析摘要**：用3-5句话说明核心逻辑
5. **风险提示**：列出1-2个需要注意的风险

注意：以上分析仅供参考，不构成投资建议。"""
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
