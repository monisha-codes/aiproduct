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


def mask_pii(text: str):
    try:
        if not text:
            return text, False

        pii_flag = False
        masked = text
        protected_spans = []

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
        # 🔹 Step 2: spaCy NER on original masked text
        # Conservative PERSON masking
        # -------------------------------
        doc = nlp(masked)

        spans_to_mask = []
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            ent_text = ent.text.strip()

            # Skip very short or weak entities
            if len(ent_text) < 3:
                continue

            # Skip legal references / case names
            if _looks_like_legal_reference(ent_text):
                continue

            # Skip common legal words that may be misclassified
            if ent_text.lower() in {
                "court", "plaintiff", "defendant", "state", "section",
                "act", "rule", "title", "code"
            }:
                continue

            start, end = ent.start_char, ent.end_char

            if not _overlaps(spans_to_mask, start, end):
                spans_to_mask.append((start, end))

        # Apply in reverse order to preserve offsets
        for start, end in sorted(spans_to_mask, reverse=True):
            masked = masked[:start] + "[NAME]" + masked[end:]
            pii_flag = True

        return masked, pii_flag

    except Exception:
        return text, False