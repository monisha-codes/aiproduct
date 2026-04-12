import re
from utils.legal_abbreviation import smart_expand_abbreviations
from utils.abbreviation_store import extract_abbreviations

ABBREVIATIONS = {
    "ADA": "Americans with Disabilities Act",
    "USC": "United States Code",
    "CFR": "Code of Federal Regulations",
    "HIPAA": "Health Insurance Portability and Accountability Act",
    "FLSA": "Fair Labor Standards Act",
    "EEOC": "Equal Employment Opportunity Commission",
    "FMLA": "Family and Medical Leave Act",
    "OSHA": "Occupational Safety and Health Act",
    "IRS": "Internal Revenue Service",
    "SEC": "Securities and Exchange Commission",
    "EPA": "Environmental Protection Agency",
    "FERPA": "Family Educational Rights and Privacy Act",
    "COPPA": "Children’s Online Privacy Protection Act",
    "SOX": "Sarbanes-Oxley Act",
    "GDPR": "General Data Protection Regulation",
    "ERISA": "Employee Retirement Income Security Act",
    "RICO": "Racketeer Influenced and Corrupt Organizations Act",
    "DMCA": "Digital Millennium Copyright Act"
}

# -------------------------------
# 🔹 Step 1: Clean Text
# -------------------------------
def clean_text(text):
    try:
        if not text:
            return text

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove escape chars
        text = text.replace("\n", " ").replace("\t", " ").replace("\\", "")

        # Normalize quotes
        text = text.replace("“", '"').replace("”", '"').replace("’", "'")

        # ✅ PRESERVE [], (), ?, . , -
        text = re.sub(r"[^\w\s\?\.\,\(\)\-\[\]]", "", text)

        # Normalize punctuation
        text = re.sub(r"\?+", "?", text)
        text = re.sub(r"\.+", ".", text)

        # Fix spacing
        text = re.sub(r"\s+([?,.])", r"\1", text)

        return text.strip()

    except Exception:
        return text

        # -------------------------------
        # 🔹 Step 7: Final trim
        # -------------------------------
        return text.strip()

    except Exception:
        return text


# -------------------------------
# 🔹 Step 2: Expand Abbreviations (IMPROVED)
# -------------------------------
def expand_abbreviations(text):
    mapping = {}

    for k, v in ABBREVIATIONS.items():
        pattern = re.compile(rf"\b{k}\b", re.IGNORECASE)

        if re.search(pattern, text):
            # ✅ Replace with Full Form + (ABBR)
            text = re.sub(pattern, f"{v} ({k})", text)
            mapping[k] = v

    return text, mapping


# -------------------------------
# 🔹 Step 3: Query Restructuring (HIGH-LEVEL)
# -------------------------------
def restructure_query(text):
    try:
        if not text:
            return text

        text = text.strip()
        text_lower = text.lower()

        # Capitalize safely
        text = text[0].upper() + text[1:] if text else text

        # Normalize legal references
        text = re.sub(r"\bsection\s+(\d+)", r"Section \1", text, flags=re.IGNORECASE)
        text = re.sub(r"\barticle\s+(\d+)", r"Article \1", text, flags=re.IGNORECASE)

        # -------------------------------
        # 🔹 Step 1: If already meaningful → keep as-is
        # -------------------------------
        valid_starts = (
            "what", "how", "why", "when", "can", "does", "is", "was", "were", "where", "shall", "will", "help", "provide", "use"
            "are", "explain", "describe", "define", "list", "show", "find", "get", "give", "display", "check", "view"
        )

        if text_lower.startswith(valid_starts):
            return text

        # -------------------------------
        # 🔹 Step 2: Context-aware restructuring
        # -------------------------------

        # Legal references
        if re.search(r"\b(section|article|act|code)\b", text_lower):
            return f"Explain the legal provisions related to {text}"

        # Case-related queries
        if re.search(r"\b(case|judgment|ruling)\b", text_lower):
            return f"Provide details about the case {text}"

        # Rights / obligations queries
        if re.search(r"\b(rights|duties|obligations|liability)\b", text_lower):
            return f"Explain the legal aspects of {text}"

        # Short conceptual queries
        if len(text.split()) <= 4:
            return f"Explain the legal concept of {text}"

        # -------------------------------
        # 🔹 Step 3: Clean fallback
        # -------------------------------
        return f"Provide a clear explanation of {text}"

    except Exception as e:
        print("Restructure error:", e)
        return text


# existing imports...

# -------------------------------
# 🔹 NEW FUNCTION (ADD HERE)
# -------------------------------
def format_restructured_query(query: str):
    try:
        if not query:
            return query

        q = query.strip()

        # Capitalize first letter
        q = q[0].upper() + q[1:] if len(q) > 1 else q.upper()

        # Add punctuation
        if not q.endswith(("?", ".", "!")):
            question_words = ["what", "why", "how", "when", "where", "who", "can", "is", "are", "do", "does"]

            if any(q.lower().startswith(w) for w in question_words):
                q += "?"
            else:
                q += "."

        return q

    except Exception:
        return query
# -------------------------------
# 🔹 Step 4: Main Preprocessing
# -------------------------------
def preprocess_query(data: dict):
    try:
        cleaned = clean_text(data["query"])

        # ✅ STEP 1: Learn abbreviations FIRST
        extract_abbreviations(cleaned)

        # ✅ STEP 2: Smart expansion (auto + embedding)
        expanded_query, smart_abbr = smart_expand_abbreviations(cleaned)

        # ✅ STEP 3: Existing abbreviation expansion
        expanded, abbr = expand_abbreviations(expanded_query)

        # ✅ Merge both
        abbr.update(smart_abbr)

        # ✅ STEP 4: Restructure
        restructured = restructure_query(expanded)

        # ✅ Format punctuation
        restructured = format_restructured_query(restructured)

        return {
            "original_query": data["query"],
            "cleaned_query": cleaned,
            "expanded_query": expanded,
            "restructured_query": restructured,
            "abbreviations": abbr
        }

    except Exception as e:
        return {"error": "preprocessing_failed", "details": str(e)}