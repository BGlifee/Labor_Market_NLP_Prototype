import tomli
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"

def load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomli.load(f)