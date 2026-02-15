import re

_URL_RE = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b\S+@\S+\b")
_WS_RE = re.compile(r"\s+")
_PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")

def normalize(text: str) -> str:
    """Light normalization for matching."""
    if not text:
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text

def extract_placeholders(body: str):
    """Extract placeholders like {name}, {date} from template body."""
    if not body:
        return []
    return sorted(set(_PLACEHOLDER_RE.findall(body)))

def safe_get_lang(obj, key: str, lang: str = "en", default=""):
    """Get obj[key][lang] safely if present."""
    val = obj.get(key, {})
    if isinstance(val, dict):
        return val.get(lang, default) or default
    return default

def safe_get_keywords(obj, key: str = "keywords", lang: str = "en"):
    """Get keywords list safely."""
    val = obj.get(key, {})
    if isinstance(val, dict):
        kws = val.get(lang, []) or []
        return [str(x) for x in kws if str(x).strip()]
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    return []
