import re
import json
import os

ABBREVIATION_MAP = {}

STORE_PATH = "abbr_store.json"


def load_abbreviations():
    global ABBREVIATION_MAP
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, "r") as f:
                ABBREVIATION_MAP = json.load(f)
        except Exception:
            ABBREVIATION_MAP = {}


def save_abbreviations():
    with open(STORE_PATH, "w") as f:
        json.dump(ABBREVIATION_MAP, f)


def extract_abbreviations(text: str):
    pattern = r'([A-Za-z][A-Za-z\s]{5,})\s*\(([A-Z]{2,10})\)'
    matches = re.findall(pattern, text)

    for full, abbr in matches:
        full_clean = full.strip()
        abbr_clean = abbr.strip()

        if len(full_clean.split()) >= 2:
            ABBREVIATION_MAP[abbr_clean] = full_clean

    save_abbreviations()


def get_abbreviation_map():
    return ABBREVIATION_MAP