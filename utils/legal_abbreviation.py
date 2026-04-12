from sentence_transformers import SentenceTransformer, util
from utils.abbreviation_store import get_abbreviation_map

# -------------------------------
# 🔹 Load embedding model (lazy)
# -------------------------------
embed_model = None

def get_embedder():
    global embed_model

    if embed_model is None:
        print("🚀 Loading SentenceTransformer model...")

        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        print("✅ SentenceTransformer loaded")

    return embed_model


# -------------------------------
# 🔹 Legal Ontology (expandable)
# -------------------------------
LEGAL_TERMS = {
    "Internal Revenue Code": ["irc"],
    "Internal Revenue Service": ["irs"],
    "Securities and Exchange Commission": ["sec"],
    "Uniform Commercial Code": ["ucc"],
    "Code of Federal Regulations": ["cfr"],
    "Fair Debt Collection Practices Act": ["fdcpa"],
}


# -------------------------------
# 🔹 Legal context keywords (NEW ✅)
# -------------------------------
LEGAL_CONTEXT_WORDS = [
    "law", "act", "section", "regulation", "code",
    "compliance", "policy", "statute", "tax", "court"
]


# -------------------------------
# 🔹 Flatten ontology
# -------------------------------
def build_term_index():
    term_map = {}
    for full, abbrs in LEGAL_TERMS.items():
        for a in abbrs:
            term_map[a.lower()] = full
    return term_map


TERM_INDEX = build_term_index()


# -------------------------------
# 🔹 Smart Detection (UPDATED 🔥)
# -------------------------------
def smart_expand_abbreviations(query: str):
    try:
        words = query.split()
        expanded = {}
        model = get_embedder()

        query_lower = query.lower()

        for word in words:
            w = word.lower()

            # 🔹 Step 1: Direct ontology match
            if w in TERM_INDEX:
                expanded[word] = TERM_INDEX[w]
                continue

            # 🔹 Step 2: Detect possible abbreviation
            if word.isupper() and 2 <= len(word) <= 6:

                # ✅ NEW: Ensure legal context exists
                if not any(k in query_lower for k in LEGAL_CONTEXT_WORDS):
                    continue  # skip random abbreviations like XYZ

                # Compare with ontology using embeddings
                word_vec = model.encode(word, convert_to_tensor=True)

                best_match = None
                best_score = 0

                for full_term in LEGAL_TERMS.keys():
                    term_vec = model.encode(full_term, convert_to_tensor=True)
                    score = util.cos_sim(word_vec, term_vec).item()

                    if score > best_score:
                        best_score = score
                        best_match = full_term

                # 🔥 Threshold (unchanged behavior)
                if best_score > 0.5:
                    expanded[word] = best_match
                else:
                    # 🔹 Smart fallback
                    expanded[word] = f"{word} (possible legal abbreviation)"

        # Replace in query
        for k, v in expanded.items():
            query = query.replace(k, v)

        return query, expanded

    except Exception:
        return query, {}