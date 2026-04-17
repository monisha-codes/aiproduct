import re
import spacy

nlp = spacy.load("en_core_web_sm")


def _overlaps(existing_spans, start, end):
    for s, e in existing_spans:
        if not (end <= s or start >= e):
            return True
    return False


def _looks_like_legal_reference(text: str) -> bool:
    t = text.lower().strip()

    legal_reference_patterns = [
        r"\bsection\s+\d+",
        r"\bsec\.?\s+\d+",
        r"\btitle\s+[ivx0-9]+",
        r"\b\d+\s+u\.?s\.?c\.?",
        r"\b\d+\s+c\.?f\.?r\.?",
        r"\b[a-z]+\s+v\.?\s+[a-z]+",   # case names like Brown v. Board
        r"§\s*\d+",
    ]

    return any(re.search(p, t, flags=re.IGNORECASE) for p in legal_reference_patterns)


def _is_common_legal_term(text: str) -> bool:
    return text.lower().strip() in {
        "court", "plaintiff", "defendant", "state", "section",
        "act", "rule", "title", "code", "law", "legal", "contract",
        "breach", "privacy", "compliance", "regulation", "statute",
        "damages", "employment", "liability", "claim", "case"
    }

def _looks_like_name_tokens(parts: list[str]) -> bool:
    """
    Conservative name check:
    - allow 2 or 3 token names only
    - reject phrases containing legal words / stopwords
    - reject obviously typo-heavy legal phrases
    """
    if len(parts) < 2 or len(parts) > 3:
        return False

    banned_tokens = {
        "of", "for", "with", "under", "regarding", "about",
        "law", "legal", "contract", "breach", "privacy", "rule", "rules",
        "section", "title", "code", "act", "court", "case", "statute",
        "compliance", "liability", "damages", "rights", "duties",
        "gdpr", "hipaa", "eeoc", "usc", "cfr", "tax", "employment"
    }

    for p in parts:
        p_low = p.lower().strip()

        # weak token
        if len(p_low) < 2:
            return False

        # legal / connector word inside phrase
        if p_low in banned_tokens:
            return False

        # token should be alphabetic name-like
        if not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", p):
            return False

    return True 


def _is_likely_person_entity(ent_text: str) -> bool:
    """
    Conservative PERSON masking:
    - mask only likely personal names
    - avoid masking typo-heavy legal phrases like:
      'brech of contrct law'
    """
    cleaned = ent_text.strip()

    if len(cleaned) < 3:
        return False

    if _looks_like_legal_reference(cleaned):
        return False

    if _is_common_legal_term(cleaned):
        return False

    # remove punctuation and split
    parts = [p for p in re.split(r"\s+", re.sub(r"[^A-Za-z\s\-']", " ", cleaned)) if p]

    # only allow true name-like tokens
    if not _looks_like_name_tokens(parts):
        return False

    return True


def mask_pii(text: str):
    try:
        if not text:
            return text, False

        pii_flag = False
        masked = text

        # -------------------------------
        # 🔹 Step 1: Regex masking
        # -------------------------------
        patterns = {
            "PHONE": r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b",
            "EMAIL": r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "ZIP": r"\b\d{5}(?:-\d{4})?\b",
        }

        for label, pattern in patterns.items():
            new_masked, count = re.subn(pattern, f"[{label}]", masked)
            if count > 0:
                pii_flag = True
                masked = new_masked

        # -------------------------------
        # 🔹 Step 2: spaCy NER
        # Use title-cased copy only for better lowercase-name detection
        # Example: "john doe" -> detected as PERSON
        # Positions remain same because title() does not change string length
        # -------------------------------
        detection_text = masked.title()
        doc = nlp(detection_text)

        spans_to_mask = []
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            ent_text = masked[ent.start_char:ent.end_char].strip()

            if not _is_likely_person_entity(ent_text):
                continue

            start, end = ent.start_char, ent.end_char

            # avoid overlapping spans
            if not _overlaps(spans_to_mask, start, end):
                spans_to_mask.append((start, end))

        # Apply in reverse order to preserve offsets
        for start, end in sorted(spans_to_mask, reverse=True):
            masked = masked[:start] + "[NAME]" + masked[end:]
            pii_flag = True

        return masked, pii_flag

    except Exception:
        return text, False