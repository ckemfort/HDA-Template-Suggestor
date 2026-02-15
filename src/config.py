from pathlib import Path

# Default path (you can override via CLI --templates)
DEFAULT_TEMPLATES_PATH = Path("templates") / "templates.json"

# Prototype settings
TOP_K = 2
MIN_CONFIDENCE = 0.10  # below this, we still return results but mark "low confidence"
END_MARKER = "END"     # multiline paste terminator in CLI mode