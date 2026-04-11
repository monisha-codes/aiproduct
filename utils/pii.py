import re
import spacy

nlp = spacy.load("en_core_web_sm")


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
            "PHONE": r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "EMAIL": r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        }

        for label, pattern in patterns.items():
            masked, count = re.subn(pattern, f"[{label}]", masked)
            if count > 0:
                pii_flag = True

        # -------------------------------
        # 🔹 Step 2: spaCy NER (case-insensitive fix 🔥)
        # -------------------------------
        # Use title case copy ONLY for detection
        doc = nlp(masked.title())

        spans = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                spans.append((ent.start_char, ent.end_char))

        # Apply masking on ORIGINAL text positions
        for start, end in reversed(spans):
            masked = masked[:start] + "[NAME]" + masked[end:]
            pii_flag = True

        return masked, pii_flag

    except Exception:
        return text, False