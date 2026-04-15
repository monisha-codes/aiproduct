from sentence_transformers import SentenceTransformer, util
import torch
import re

# from utils.abbreviation_store import get_abbreviation_map   # keep if you use it later

# -------------------------------
# 🔹 Load embedding model (lazy)
# -------------------------------
embed_model = None
term_embeddings = None


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
    "Code of Federal Regulations": ["cfr", "c.f.r"],
    "Fair Debt Collection Practices Act": ["fdcpa"],
    "United States Code": ["usc", "u.s.c"],
    "Health Insurance Portability and Accountability Act": ["hipaa"],
    "Americans with Disabilities Act": ["ada"],
    "Fair Labor Standards Act": ["flsa"],
    "Family and Medical Leave Act": ["fmla"],
    "Equal Employment Opportunity Commission": ["eeoc"],
    "Occupational Safety and Health Act": ["osha"],
    "Employee Retirement Income Security Act": ["erisa"],
    "Racketeer Influenced and Corrupt Organizations Act": ["rico"],
    "Digital Millennium Copyright Act": ["dmca"],
    "Federal Trade Commission": ["ftc"],
    "Department of Justice": ["doj"],
    "Supreme Court of the United States": ["scotus"],
    "Children's Online Privacy Protection Act": ["coppa"],
    "Family Educational Rights and Privacy Act": ["ferpa"],
    "Sarbanes-Oxley Act": ["sox"],
}


# -------------------------------
# 🔹 Legal context keywords
# -------------------------------
LEGAL_CONTEXT_WORDS = [
    "law", "act", "section", "sec", "regulation", "rule", "rules",
    "code", "compliance", "policy", "statute", "tax", "court",
    "legal", "privacy", "rights", "liability", "contract",
    "usc", "u.s.c", "cfr", "c.f.r", "title", "article", "§"
]


# -------------------------------
# 🔹 Flatten ontology
# -------------------------------
def build_term_index():
    term_map = {}
    for full, abbrs in LEGAL_TERMS.items():
        for a in abbrs:
            normalized = normalize_token(a)
            term_map[normalized] = full
    return term_map


def normalize_token(token: str) -> str:
    if not token:
        return ""
    token = token.strip().lower()
    token = token.replace(".", "")
    token = token.replace(",", "")
    token = token.replace("(", "").replace(")", "")
    return token


TERM_INDEX = build_term_index()


# -------------------------------
# 🔹 Precompute ontology embeddings
# -------------------------------
def get_term_embeddings():
    global term_embeddings

    if term_embeddings is None:
        model = get_embedder()
        full_terms = list(LEGAL_TERMS.keys())
        embeddings = model.encode(full_terms, convert_to_tensor=True)
        term_embeddings = {
            "terms": full_terms,
            "vectors": embeddings
        }

    return term_embeddings


# -------------------------------
# 🔹 Token helpers
# -------------------------------
def extract_candidate_tokens(query: str):
    """
    Extract likely abbreviation candidates without requiring strict UPPERCASE.
    Supports:
    - HIPAA
    - Hipaa
    - hipaa
    - U.S.C.
    - CFR
    """
    if not query:
        return []

    raw_tokens = re.findall(r"\b[a-zA-Z][a-zA-Z\.\-]{1,12}\b", query)
    candidates = []

    for token in raw_tokens:
        norm = normalize_token(token)

        if len(norm) < 2 or len(norm) > 12:
            continue

        # likely abbreviation-style token
        if (
            token.isupper()
            or token.istitle()
            or norm in TERM_INDEX
            or re.fullmatch(r"[a-z]{2,8}", norm) is not None
        ):
            candidates.append(token)

    return list(dict.fromkeys(candidates))


def has_legal_context(query: str) -> bool:
    q = query.lower()
    if any(k in q for k in LEGAL_CONTEXT_WORDS):
        return True

    # citation-style context
    citation_patterns = [
        r"\bsection\s+\d+",
        r"\bsec\.?\s+\d+",
        r"\btitle\s+\w+",
        r"\b\d+\s+u\.?s\.?c\.?",
        r"\b\d+\s+c\.?f\.?r\.?",
        r"§\s*\d+",
    ]
    return any(re.search(p, q, flags=re.IGNORECASE) for p in citation_patterns)


# -------------------------------
# 🔹 Smart Detection (REFINED)
# -------------------------------
def smart_expand_abbreviations(query: str):
    try:
        if not query:
            return query, {}

        expanded = {}
        model = get_embedder()
        embeddings_data = get_term_embeddings()

        has_context = has_legal_context(query)
        candidates = extract_candidate_tokens(query)

        for token in candidates:
            normalized = normalize_token(token)

            # -------------------------------
            # 🔹 Step 1: Direct ontology match
            # -------------------------------
            if normalized in TERM_INDEX:
                expanded[token] = TERM_INDEX[normalized]
                continue

            # -------------------------------
            # 🔹 Step 2: Skip weak unknowns if no legal context
            # -------------------------------
            if not has_context:
                continue

            # Ignore obvious common words
            if normalized in {
                "what", "when", "where", "which", "explain", "rule",
                "case", "law", "court", "legal", "rights", "duties"
            }:
                continue

            # -------------------------------
            # 🔹 Step 3: Embedding similarity for unknown legal abbreviations
            # -------------------------------
            token_vec = model.encode(token, convert_to_tensor=True)

            scores = util.cos_sim(token_vec, embeddings_data["vectors"])[0]
            best_idx = int(torch.argmax(scores).item())
            best_score = float(scores[best_idx].item())
            best_match = embeddings_data["terms"][best_idx]

            # Higher threshold = safer expansion
            if best_score >= 0.58:
                expanded[token] = best_match
            elif token.isupper() or token.istitle():
                # preserve probable legal abbreviation without discarding it
                expanded[token] = f"{token} (possible legal abbreviation)"

        # -------------------------------
        # 🔹 Step 4: Replace safely
        # replace longer tokens first
        # -------------------------------
        for original in sorted(expanded.keys(), key=len, reverse=True):
            replacement = expanded[original]
            pattern = re.compile(rf"\b{re.escape(original)}\b")
            query = re.sub(pattern, f"{replacement} ({original})" if replacement != f"{original} (possible legal abbreviation)" else replacement, query)

        return query, expanded

    except Exception:
        return query, {}