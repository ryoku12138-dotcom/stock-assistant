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
                    ai_summary: str) -> bool:
        """Send AI analysis card. The AI output is displayed directly."""
        color = "red" if signal == "BUY" else ("green" if signal == "SELL" else "blue")
        # Extract score from AI output for header
        header_title = f"{stock_name}({stock_code})"
        if signal == "BUY":
            header_title += " | 买入信号"
        elif signal == "SELL":
            header_title += " | 卖出信号"
        else:
            header_title += " | 观望"

        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": header_title
                    },
                    "template": color
                },
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": ai_summary
                        }
                    },
                    {"tag": "hr"},
                    {
                        "tag": "note",
                        "elements": [
                            {
                                "tag": "plain_text",
                                "content": "以上分析仅供参考，不构成投资建议 | AI生成"
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


if __name__ == "__main__":
    notifier = FeishuNotifier()
    success = notifier.send_text(
        title="Stock Assistant 启动",
        content="系统已就绪，开始监控自选股..."
    )
    print(f"Test notification: {'sent' if success else 'failed'}")
