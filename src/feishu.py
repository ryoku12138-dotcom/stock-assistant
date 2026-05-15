"""
Feishu webhook bot notification module.
Supports plain text and rich card messages.
"""
import requests
from src.config_helper import get_config


class FeishuNotifier:
    def __init__(self):
        self.webhook_url = get_config()["feishu"]["webhook_url"]

    def send_text(self, title: str, content: str) -> bool:
        """Send a rich text card message."""
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": "blue"
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {"tag": "lark_md", "content": content}
                    }
                ]
            }
        }
        try:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            result = r.json()
            if result.get("code") == 0:
                return True
            else:
                print(f"Feishu send failed: {result}")
                return False
        except Exception as e:
            print(f"Feishu request error: {e}")
            return False

    def send_signal(self, stock_code: str, stock_name: str, signal: str,
                    indicators: dict, ai_summary: str) -> bool:
        """Send a formatted buy/sell signal card."""
        color = "red" if "buy" in signal.lower() or "买入" in signal else "green"
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"{stock_name}({stock_code}) {signal}"
                    },
                    "template": color
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**技术指标：**\n{self._format_indicators(indicators)}\n\n**AI分析：**\n{ai_summary}"
                        }
                    },
                    {"tag": "hr"},
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": "以上分析仅供参考，不构成投资建议"
                            }
                        ]
                    }
                ]
            }
        }
        try:
            r = requests.post(self.webhook_url, json=payload, timeout=10)
            result = r.json()
            return result.get("code") == 0
        except Exception as e:
            print(f"Feishu request error: {e}")
            return False

    def _format_indicators(self, indicators: dict) -> str:
        lines = []
        if "ma" in indicators:
            lines.append(f"- MA: {indicators['ma']}")
        if "macd" in indicators:
            lines.append(f"- MACD: {indicators['macd']}")
        if "rsi" in indicators:
            lines.append(f"- RSI: {indicators['rsi']}")
        if "kdj" in indicators:
            lines.append(f"- KDJ: {indicators['kdj']}")
        if "bollinger" in indicators:
            lines.append(f"- 布林带: {indicators['bollinger']}")
        return "\n".join(lines)


if __name__ == "__main__":
    notifier = FeishuNotifier()
    success = notifier.send_text(
        title="Stock Assistant 启动",
        content="系统已就绪，开始监控自选股..."
    )
    print(f"Test notification: {'sent' if success else 'failed'}")
