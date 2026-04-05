import re

def mask_pii(text):
    patterns = {
        "PHONE": r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "EMAIL": r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    masked = text
    found = False

    for label, pattern in patterns.items():
        matches = re.findall(pattern, masked)
        for m in matches:
            masked = masked.replace(m, f"[{label}]")
            found = True

    return masked, found