import subprocess
import sys

cmd = [
    sys.executable, "-m", "src.cli_embed_ver",
    "--templates", "templates/templates.json",
    "--model-path", "models/all-MiniLM-L6-v2",
]

subprocess.run(cmd, check=False)