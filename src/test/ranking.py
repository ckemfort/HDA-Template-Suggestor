import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import normalize, extract_placeholders, safe_get_lang, safe_get_keywords


@dataclass
class TemplateItem:
    id: str
    title: str
    body: str
    keywords: List[str]
    placeholders: List[str]
    doc_norm: str


def _coerce_template_list(data):
    """Support either a list or a dict wrapper like {'templates': [...]}."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("templates", "items", "data"):
            if k in data and isinstance(data[k], list):
                return data[k]
    raise ValueError("Unsupported templates.json structure (expected list or dict with a list field).")


def load_templates(json_path: Path, lang: str = "en") -> List[TemplateItem]:
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    items = _coerce_template_list(raw)

    templates: List[TemplateItem] = []
    for t in items:
        tid = str(t.get("id", "")).strip() or str(t.get("template_id", "")).strip()
        title = safe_get_lang(t, "title", lang=lang, default="").strip()
        body = safe_get_lang(t, "body", lang=lang, default="").strip()
        desc = safe_get_lang(t, "description", lang=lang, default="").strip()
        kws = safe_get_keywords(t, "keywords", lang=lang)

        # Build a single document to match against
        doc = " ".join([title, desc, " ".join(kws), body])
        templates.append(
            TemplateItem(
                id=tid,
                title=title or tid or "Untitled",
                body=body,
                keywords=kws,
                placeholders=extract_placeholders(body),
                doc_norm=normalize(doc),
            )
        )

    if not templates:
        raise ValueError("No templates found in templates.json.")
    return templates


def build_ranker(templates: List[TemplateItem]) -> Tuple[TfidfVectorizer, object]:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform([t.doc_norm for t in templates])
    return vectorizer, X


def _keyword_hits(email_norm: str, keywords: List[str], limit: int = 6) -> List[str]:
    hits = []
    for kw in keywords:
        k = normalize(kw)
        if k and k in email_norm:
            hits.append(kw)
    return hits[:limit]


def suggest(
    email_text: str,
    templates: List[TemplateItem],
    vectorizer: TfidfVectorizer,
    X,
    top_k: int = 2,
    min_confidence: float = 0.10,
):
    email_norm = normalize(email_text)
    v = vectorizer.transform([email_norm])
    sims = cosine_similarity(v, X).ravel()

    # rank high -> low
    idxs = sims.argsort()[::-1][:max(top_k, 1)]

    results = []
    for i in idxs:
        t = templates[i]
        score = float(sims[i])
        hits = _keyword_hits(email_norm, t.keywords)
        rationale = f"Matched because similarity={score:.3f}"
        if hits:
            rationale += f" and keyword hits: {', '.join(hits)}"
        if score < min_confidence:
            rationale += " (low confidence)"

        results.append(
            {
                "id": t.id,
                "title": t.title,
                "score": score,
                "rationale": rationale,
                "placeholders": t.placeholders,
                "body": t.body,
            }
        )

    return results
