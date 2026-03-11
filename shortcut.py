import sys
from pathlib import Path

ROOT = Path(__file__).parent

sys.argv = [
    "cli_embed_ver",
    "--templates", str(ROOT / "templates/templates.json"),
    "--model-path", str(ROOT / "scripts/models/all-MiniLM-L6-v2"),
]

from src.cli_embed_ver import main
main()