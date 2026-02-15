import argparse
import sys
import time
from pathlib import Path

from src.config import DEFAULT_TEMPLATES_PATH, TOP_K, MIN_CONFIDENCE, END_MARKER
from src.test.ranking import load_templates, build_ranker, suggest


def _read_email_from_stdin() -> str:
    """
    If piped input exists, read it all.
    If interactive, read multiline until END marker.
    """
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
    parser = argparse.ArgumentParser(description="Offline Email Template Suggestor (CLI)")
    parser.add_argument("--templates", type=str, default=str(DEFAULT_TEMPLATES_PATH),
                        help="Path to templates.json (default: templates/templates.json)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of suggestions to return")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE,
                        help="Below this similarity score, mark result as low confidence")
    parser.add_argument("--show-body", action="store_true", help="Print the full template body")
    args = parser.parse_args()

    templates_path = Path(args.templates)
    templates = load_templates(templates_path, lang="en")
    vectorizer, X = build_ranker(templates)

    email_text = _read_email_from_stdin().strip()
    if not email_text:
        print("No email text provided.")
        sys.exit(1)

    t0 = time.perf_counter()
    results = suggest(
        email_text=email_text,
        templates=templates,
        vectorizer=vectorizer,
        X=X,
        top_k=args.top_k,
        min_confidence=args.min_confidence,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print("\nTop matches:\n")
    for r in results:
        print(f"- {r['title']} ({r['id']})")
        print(f"  {r['rationale']}")
        if r["placeholders"]:
            print(f"  Placeholders: {', '.join(r['placeholders'])}")
        else:
            print("  Placeholders: (none)")
        if args.show_body:
            print("  --- Template body ---")
            print(r["body"] or "(empty)")
            print("  ---------------------")
        print()

    print(f"Query latency: {dt_ms:.1f} ms")


if __name__ == "__main__":
    main()
