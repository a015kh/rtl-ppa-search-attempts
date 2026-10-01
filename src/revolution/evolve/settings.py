"""Explicit locations for the migrated EvolVE workflows."""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'resources'
CONFIG_FILE = os.environ.get('REVOLUTION_EVOLVE_CONFIG')
CONFIG = json.loads(Path(CONFIG_FILE).read_text(encoding='utf-8')) if CONFIG_FILE else {}

def path(name, default=None):
    value = CONFIG.get('paths', {}).get(name, default)
    if value is None:
        return ROOT / '.unconfigured' / name
    result = Path(value).expanduser()
    base = Path(CONFIG_FILE).resolve().parent if CONFIG_FILE else ROOT
    return result.resolve() if result.is_absolute() else (base / result).resolve()

def tool(name):
    value = CONFIG.get('tools', {}).get(name)
    if value:
        found = shutil.which(value)
        if found:
            return Path(found).resolve()
        result = Path(value).expanduser()
        base = Path(CONFIG_FILE).resolve().parent if CONFIG_FILE else ROOT
        return result.resolve() if result.is_absolute() else (base / result).resolve()
    return ROOT / '.unconfigured' / name

INFERENCE = CONFIG.get('inference', {})
TOOL_SELECTION = CONFIG.get('tool_selection', ['tool', '1', '4', '6'])
