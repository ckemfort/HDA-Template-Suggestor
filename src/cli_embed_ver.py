import argparse
import json
import sys
import time
from pathlib import Path

from .config import END_MARKER
from .embeddings import (
    load_templates_from_json,
    load_local_model,
    build_template_index,
    suggest,
)

DIVIDER = "--------------------"

def _read_email_from_stdin() -> str:
    if not sys.stdin.isatty():
        return sys.stdin.read()

    print(f"Paste the incoming email. Type {END_MARKER} on its own line to finish.\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == END_MARKER:
            break
        lines.append(line)
    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser(description="Email Template Suggestor (Embeddings, offline)")
    p.add_argument("--templates", default="templates/templates.json", help="Path to templates.json")
    p.add_argument("--model-path", default="models/all-MiniLM-L6-v2", help="Local path to model folder")
    p.add_argument("--top-k", type=int, default=3, help="How many matches to show (default: 2)")
    p.add_argument("--min-confidence", type=float, default=0.30)
    args = p.parse_args()

    data = json.loads(Path(args.templates).read_text(encoding="utf-8"))
    templates = load_templates_from_json(data, lang="en")

    model = load_local_model(args.model_path)
    template_emb = build_template_index(model, templates)

    email_text = _read_email_from_stdin().strip()
    if not email_text:
        print("No email text provided.")
        sys.exit(1)

    t0 = time.perf_counter()
    results = suggest(
        email_text=email_text,
        model=model,
        template_embeddings=template_emb,
        templates=templates,
        top_k=max(1, args.top_k),
        min_confidence=args.min_confidence,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # 1) Print top-k summaries (usually 2)
    k = min(len(results), max(1, args.top_k))
    print("\nTop matches:\n")
    for i in range(k):
        r = results[i]
        print(f"{i+1}) {r['title']} ({r['id']})")
        print(f"   {r['rationale']}")
        #if r["placeholders"]:
            #print(f"   Placeholders: {', '.join(r['placeholders'])}")
        #else:
            #print("   Placeholders: (none)")
        #print()

    # 2) Divider
    print(DIVIDER)

    # 3) Print best template body (#1)
    best = results[0]
    print("\nBest template (copy/paste):\n")
    print(best["body"] or "(empty template body)")
    print(f"\n{DIVIDER}\n")

    print(f"Query latency: {dt_ms:.1f} ms")

if __name__ == "__main__":
    main()

