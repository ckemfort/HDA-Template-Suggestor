import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from .preprocessing import normalize, extract_placeholders, safe_get_lang, safe_get_keywords


@dataclass
class TemplateItem:
    id: str
    title: str
    body: str
    keywords: List[str]
    #placeholders: List[str]
    doc_text: str  # raw (not normalized) text used for embedding


def load_templates_from_json(data: Any, lang: str = "en") -> List[TemplateItem]:
    # Support either list or dict wrapper
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("templates") or data.get("items") or data.get("data")
        if not isinstance(items, list):
            raise ValueError("Unsupported templates.json structure.")
    else:
        raise ValueError("Unsupported templates.json structure.")

    out: List[TemplateItem] = []
    for t in items:
        tid = str(t.get("id", "")).strip() or str(t.get("template_id", "")).strip()
        title = safe_get_lang(t, "title", lang=lang, default="").strip()
        body = safe_get_lang(t, "body", lang=lang, default="").strip()
        desc = safe_get_lang(t, "description", lang=lang, default="").strip()
        kws = safe_get_keywords(t, "keywords", lang=lang)

        # Embedding "doc": include title + desc + keywords + body
        doc = " ".join([title, desc, " ".join(kws), body]).strip()

        out.append(
            TemplateItem(
                id=tid,
                title=title or tid or "Untitled",
                body=body,
                keywords=kws,
                #placeholders=extract_placeholders(body),
                doc_text=doc,
            )
        )

    if not out:
        raise ValueError("No templates found.")
    return out


def _keyword_hits(email_norm: str, keywords: List[str], limit: int = 6) -> List[str]:
    hits = []
    for kw in keywords:
        k = normalize(kw)
        if k and k in email_norm:
            hits.append(kw)
    return hits[:limit]


def load_local_model(model_path: str) -> SentenceTransformer:
    # Force offline behavior if you want (optional)
    # If the model isn't present locally, this will fail (which is good for "offline-only").
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    return SentenceTransformer(model_path)


def build_template_index(model: SentenceTransformer, templates: List[TemplateItem]):
    # Compute embeddings for templates once
    template_texts = [t.doc_text for t in templates]
    emb = model.encode(template_texts, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def suggest(
    email_text: str,
    model: SentenceTransformer,
    template_embeddings: np.ndarray,
    templates: List[TemplateItem],
    top_k: int = 2,
    min_confidence: float = 0.30,  # embeddings often have different score ranges
) -> List[Dict[str, Any]]:
    email_norm = normalize(email_text)
    email_emb = model.encode([email_text], normalize_embeddings=True)
    email_emb = np.asarray(email_emb, dtype=np.float32)  # shape (1, d)

    # cosine similarity with normalized vectors == dot product
    scores = (template_embeddings @ email_emb[0]).astype(np.float32)  # shape (N,)

    idxs = scores.argsort()[::-1][:max(top_k, 1)]
    results = []
    for i in idxs:
        t = templates[i]
        score = float(scores[i])
        hits = _keyword_hits(email_norm, t.keywords)
        rationale = f"Semantic similarity={score:.3f}"
        if hits:
            rationale += f" , keyword hits: {', '.join(hits)}"
        if score < min_confidence:
            rationale += " (low confidence)"

        results.append(
            {
                "id": t.id,
                "title": t.title,
                "score": score,
                "rationale": rationale,
                #"placeholders": t.placeholders,
                "body": t.body,
            }
        )
    return results
