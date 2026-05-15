"""
Config loader with multi-layer override for secrets.
Priority: env var > config.local.yaml > config.yaml
"""
import os
import yaml
from pathlib import Path

_config = None


def _merge_config(base: dict, override: dict) -> dict:
    """Shallow merge override into base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key].update(value)
        else:
            base[key] = value
    return base


def _is_placeholder(val: str) -> bool:
    """Check if value is a placeholder like ${VAR_NAME}."""
    return isinstance(val, str) and val.startswith("${") and val.endswith("}")


def get_config() -> dict:
    global _config
    if _config is not None:
        return _config

    root = Path(__file__).parent.parent

    # 1. Load base config (committed, may have placeholders)
    with open(root / "config.yaml", "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f)

    # 2. Load local override (gitignored, has real secrets for dev)
    local_path = root / "config.local.yaml"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            local_config = yaml.safe_load(f)
        if local_config:
            _merge_config(_config, local_config)

    # 3. Override with environment variables (for GitHub Actions)
    if os.environ.get("FEISHU_WEBHOOK_URL"):
        val = os.environ["FEISHU_WEBHOOK_URL"]
        if not _is_placeholder(val):
            _config["feishu"]["webhook_url"] = val

    if os.environ.get("DEEPSEEK_API_KEY"):
        val = os.environ["DEEPSEEK_API_KEY"]
        if not _is_placeholder(val):
            _config["deepseek"]["api_key"] = val
    if os.environ.get("DEEPSEEK_BASE_URL"):
        _config["deepseek"]["base_url"] = os.environ["DEEPSEEK_BASE_URL"]

    if os.environ.get("STOCK_WATCHLIST"):
        codes = [c.strip() for c in os.environ["STOCK_WATCHLIST"].split(",") if c.strip()]
        watchlist = []
        for entry in codes:
            if ":" in entry:
                code, name = entry.split(":", 1)
                watchlist.append({"code": code.strip(), "name": name.strip()})
            else:
                watchlist.append({"code": entry.strip(), "name": entry.strip()})
        _config["watchlist"] = watchlist

    return _config
