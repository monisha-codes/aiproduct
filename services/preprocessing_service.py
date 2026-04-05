import re

ABBREVIATIONS = {
    "ADA": "Americans with Disabilities Act",
    "USC": "United States Code"
}

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def expand_abbreviations(text):
    mapping = {}
    for k, v in ABBREVIATIONS.items():
        if k in text:
            text = text.replace(k, v)
            mapping[k] = v
    return text, mapping


def preprocess_query(data: dict):
    try:
        cleaned = clean_text(data["query"])
        expanded, abbr = expand_abbreviations(cleaned)

        return {
            "original_query": data["query"],
            "cleaned_query": cleaned,
            "expanded_query": expanded,
            "abbreviations": abbr
        }

    except Exception as e:
        return {"error": "preprocessing_failed", "details": str(e)}