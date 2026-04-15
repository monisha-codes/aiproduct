import re
import json
import os

ABBREVIATION_MAP = {}

STORE_PATH = "abbr_store.json"


# -------------------------------
# 🔹 Normalize helpers
# -------------------------------
def normalize_abbr(abbr: str) -> str:
    if not abbr:
        return ""
    return abbr.strip().upper().replace(".", "")


def normalize_full(full: str) -> str:
    if not full:
        return ""
    return " ".join(full.strip().split())


# -------------------------------
# 🔹 Load / Save
# -------------------------------
def load_abbreviations():
    global ABBREVIATION_MAP

    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, "r") as f:
                data = json.load(f)

                # normalize keys on load
                ABBREVIATION_MAP = {
                    normalize_abbr(k): normalize_full(v)
                    for k, v in data.items()
                }

        except Exception:
            ABBREVIATION_MAP = {}


def save_abbreviations():
    try:
        with open(STORE_PATH, "w") as f:
            json.dump(ABBREVIATION_MAP, f, indent=2)
    except Exception:
        pass


# -------------------------------
# 🔹 Extract abbreviations (IMPROVED)
# -------------------------------
def extract_abbreviations(text: str):
    """
    Detect patterns like:
    - Health Insurance Portability and Accountability Act (HIPAA)
    - HIPAA (Health Insurance Portability and Accountability Act)
    """

    if not text:
        return

    patterns = [
        # FULL (ABBR)
        r'([A-Za-z][A-Za-z\s]{5,})\s*\(([A-Z]{2,10})\)',
        # ABBR (FULL)
        r'\(([A-Za-z][A-Za-z\s]{5,})\)\s*([A-Z]{2,10})',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)

        for part1, part2 in matches:
            # detect which is full vs abbr
            if part1.isupper():
                abbr, full = part1, part2
            elif part2.isupper():
                abbr, full = part2, part1
            else:
                continue

            abbr_clean = normalize_abbr(abbr)
            full_clean = normalize_full(full)

            # only meaningful legal phrases
            if len(full_clean.split()) >= 2:
                ABBREVIATION_MAP[abbr_clean] = full_clean

    save_abbreviations()


# -------------------------------
# 🔹 Get map
# -------------------------------
def get_abbreviation_map():
    return ABBREVIATION_MAP


# -------------------------------
# 🔹 NEW: Lookup helper (IMPORTANT)
# -------------------------------
def resolve_abbreviation(token: str):
    """
    Case-insensitive lookup.
    Used by preprocessing / validation.
    """
    if not token:
        return None

    key = normalize_abbr(token)
    return ABBREVIATION_MAP.get(key)