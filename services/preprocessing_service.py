import re
import requests
from typing import Dict, List, Tuple, Any
from spellchecker import SpellChecker
from rapidfuzz import process, fuzz

# Optional: keep your existing helpers
# from utils.legal_abbreviation import smart_expand_abbreviations
# from utils.abbreviation_store import extract_abbreviations

ABBREVIATIONS = {
    "ADA": "Americans with Disabilities Act",
    "USC": "United States Code",
    "U.S.C.": "United States Code",
    "CFR": "Code of Federal Regulations",
    "C.F.R.": "Code of Federal Regulations",
    "HIPAA": "Health Insurance Portability and Accountability Act",
    "FLSA": "Fair Labor Standards Act",
    "EEOC": "Equal Employment Opportunity Commission",
    "FMLA": "Family and Medical Leave Act",
    "OSHA": "Occupational Safety and Health Act",
    "IRS": "Internal Revenue Service",
    "SEC": "Securities and Exchange Commission",
    "EPA": "Environmental Protection Agency",
    "FERPA": "Family Educational Rights and Privacy Act",
    "COPPA": "Children's Online Privacy Protection Act",
    "SOX": "Sarbanes-Oxley Act",
    "ERISA": "Employee Retirement Income Security Act",
    "RICO": "Racketeer Influenced and Corrupt Organizations Act",
    "DMCA": "Digital Millennium Copyright Act",
    "SCOTUS": "Supreme Court of the United States",
    "DOJ": "Department of Justice",
    "FTC": "Federal Trade Commission",
    "NLRB": "National Labor Relations Board",
}

QUESTION_STARTERS = {
    "what", "why", "how", "when", "where", "who", "whom", "which",
    "can", "could", "does", "do", "did", "is", "are", "was", "were",
    "shall", "will", "would", "may", "should", "explain", "describe",
    "define", "compare", "show", "find", "list", "summarize"
}

LEGAL_HINT_TERMS = {
    "section", "sec", "article", "title", "chapter", "usc", "u.s.c",
    "cfr", "c.f.r", "act", "regulation", "statute", "case", "holding",
    "judgment", "ruling", "plaintiff", "defendant", "liability", "damages",
    "breach", "contract", "privacy", "compliance", "federal", "state"
}

spell = SpellChecker()
rewriter = None

# whitelist important legal words (VERY IMPORTANT)
LEGAL_WHITELIST = {
    "hipaa", "usc", "cfr", "fdcpa", "ada", "eeoc", "ferpa",
    "coppa", "sox", "dmca", "ftc", "doj", "scotus",
    "flsa", "fmla", "erisa", "rico", "irs", "sec", "epa",
    "employment", "liability", "breach", "contract", "privacy",
    "compliance", "regulation", "statute", "damages", "plaintiff",
    "defendant", "court", "federal", "state", "implications", "check",
    "legal", "advice", "regarding", "implication", "meaning"
}

PLACEHOLDER_PATTERN = r"\[(NAME|PHONE|EMAIL|SSN|ZIP)\]"

def correct_abbreviation_typos_in_preprocessing(query: str):
    """
    Correct likely legal abbreviation typos before generic spell correction.
    Example:
    hipa -> HIPAA
    uscc -> USC
    flasaa -> FLSA
    """
    try:
        if not query:
            return query, {}

        tokens = query.split()
        corrected_tokens = []
        corrections = {}

        known_abbr_map = {
            abbr.lower().replace(".", ""): abbr
            for abbr in ABBREVIATIONS.keys()
        }
        known_abbr_keys = list(known_abbr_map.keys())

        for token in tokens:
            raw = token
            clean = re.sub(r"[^a-zA-Z\.]", "", token).lower().replace(".", "")

            # skip empty / very short / obvious question words
            if not clean or len(clean) < 3 or clean in {"what", "why", "how", "when", "who", "is", "are"}:
                corrected_tokens.append(raw)
                continue

            # exact abbreviation already known
            if clean in known_abbr_map:
                corrected_tokens.append(known_abbr_map[clean])
                continue

            # fuzzy abbreviation match
            match = process.extractOne(clean, known_abbr_keys, scorer=fuzz.ratio)
            if match:
                candidate, score, _ = match
                if score >= 80:
                    fixed = known_abbr_map[candidate]
                    corrected_tokens.append(fixed)
                    corrections[raw] = fixed
                    continue

            corrected_tokens.append(raw)

        return " ".join(corrected_tokens), corrections

    except Exception:
        return query, {}
    
def get_rewriter():
    """
    Optional local LLM fallback via Ollama.
    Lazy-loaded so your existing process is not affected unless needed.
    """
    global rewriter

    if rewriter is None:
        try:
            rewriter = {
                "base_url": "http://localhost:11434/api/generate",
                "model": "qwen2.5:3b"
            }
        except Exception:
            rewriter = False

    return rewriter


def protect_placeholders(text: str):
    protected = {}
    counter = 0

    def replacer(match):
        nonlocal counter
        token = f"__PH_{counter}__"
        protected[token] = match.group(0).upper()
        counter += 1
        return token

    new_text = re.sub(PLACEHOLDER_PATTERN, replacer, text, flags=re.IGNORECASE)
    return new_text, protected


def restore_placeholders(text: str, protected: dict):
    for token, original in protected.items():
        text = text.replace(token, original)
    return text


def repair_common_legal_phrases(text: str):
    """
    Lightweight phrase-order cleanup without changing your process.
    """
    try:
        if not text:
            return text

        q = text

        replacements = {
            r"\badvice legal\b": "legal advice",
            r"\brule hipaa\b": "HIPAA rule",
            r"\blaw employment\b": "employment law",
            r"\bcontract breach\b": "breach of contract",
            r"\bprivacy data\b": "data privacy",
            r"\bssn check\b": "SSN check",
            r"\bcheck ssn\b": "SSN check",
            r"\bimplications legal\b": "legal implications",
        }

        for pattern, replacement in replacements.items():
            q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)

        return q

    except Exception:
        return text


def correct_spelling(query: str):
    try:
        if not query:
            return query

        protected_query, protected_map = protect_placeholders(query)

        # 🔥 NEW: correct legal abbreviation typos first
        protected_query, abbr_corrections = correct_abbreviation_typos_in_preprocessing(protected_query)

        words = protected_query.split()
        corrected_words = []

        for word in words:
            # preserve placeholders exactly
            if word.startswith("__PH_") and word.endswith("__"):
                corrected_words.append(word)
                continue

            clean = re.sub(r"[^a-zA-Z_]", "", word).lower()

            # skip empty / numbers / short tokens
            if not clean or len(clean) <= 2:
                corrected_words.append(word)
                continue

            # skip known legal terms
            if clean in LEGAL_WHITELIST:
                corrected_words.append(word)
                continue

            # skip known abbreviations and abbreviation corrections
            normalized_abbr_keys = {k.lower().replace(".", "") for k in ABBREVIATIONS.keys()}
            if clean.replace(".", "") in normalized_abbr_keys:
                corrected_words.append(word)
                continue

            if word in abbr_corrections.values():
                corrected_words.append(word)
                continue

            # preserve uppercase abbreviations
            if word.isupper():
                corrected_words.append(word)
                continue

            # 🔥 IMPORTANT: do not turn short legal-like tokens into common words
            if 3 <= len(clean) <= 6:
                match = process.extractOne(clean, list(normalized_abbr_keys), scorer=fuzz.ratio)
                if match and match[1] >= 75:
                    corrected_words.append(word)
                    continue

            correction = spell.correction(clean)

            if correction:
                if word.istitle():
                    correction = correction.title()
                corrected_words.append(correction)
            else:
                corrected_words.append(word)

        corrected = " ".join(corrected_words)
        corrected = restore_placeholders(corrected, protected_map)

        return corrected

    except Exception:
        return query


def clean_text(text: str) -> str:
    """
    Safer cleaning:
    - normalize whitespace
    - preserve legal punctuation
    - do not over-strip symbols like §, :, /, &
    - normalize smart quotes and dashes
    """
    if not text:
        return text

    text = str(text)

    replacements = {
        "\n": " ",
        "\t": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Preserve common legal symbols and references
    text = re.sub(r"[^\w\s\?\.\,\(\)\-\[\]§:/&']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize repeated punctuation
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+([?,.:;])", r"\1", text)

    return text


def extract_unknown_abbreviations(text: str) -> List[str]:
    """
    Detect likely abbreviations even if not present in ABBREVIATIONS.
    Keeps them for downstream logic instead of ignoring them.
    Ignores protected PII placeholders.
    """
    if not text:
        return []

    matches = re.findall(r"\b[A-Z][A-Z\.\-&]{1,10}\b", text)
    candidates = []
    ignored = {"NAME", "PHONE", "EMAIL", "SSN", "ZIP"}

    normalized_known = {k.replace(".", "").upper() for k in ABBREVIATIONS.keys()}

    for m in matches:
        normalized = m.strip(".").upper()
        if normalized in ignored:
            continue
        if normalized not in normalized_known and len(normalized) >= 2:
            candidates.append(normalized)

    return sorted(set(candidates))


def expand_abbreviations(text: str) -> Tuple[str, Dict[str, str], List[str]]:
    """
    Expand known abbreviations.
    Preserve unknown abbreviations separately instead of dropping them.
    """
    mapping: Dict[str, str] = {}
    expanded = text

    for short, full in ABBREVIATIONS.items():
        pattern = re.compile(rf"\b{re.escape(short)}\b", re.IGNORECASE)
        if re.search(pattern, expanded):
            expanded = re.sub(pattern, f"{full} ({short})", expanded)
            mapping[short] = full

    unknown = extract_unknown_abbreviations(expanded)
    return expanded, mapping, unknown


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Lightweight legal entity extraction for retrieval support.
    """
    entities = {
        "sections": [],
        "acts": [],
        "citations": [],
        "courts": [],
    }

    section_patterns = [
        r"\bSection\s+\d+[A-Za-z0-9\-\(\)]*",
        r"\bSec\.?\s+\d+[A-Za-z0-9\-\(\)]*",
        r"§\s*\d+[A-Za-z0-9\-\(\)]*",
        r"\bTitle\s+\d+\b",
    ]
    act_pattern = r"\b[A-Z][A-Za-z&,\- ]+(?:Act|Code|Regulation|Rule|Rules)\b"
    citation_patterns = [
        r"\b\d+\s+U\.?S\.?C\.?\s+§?\s*\d+[A-Za-z0-9\-\(\)]*",
        r"\b\d+\s+C\.?F\.?R\.?\s+§?\s*\d+[A-Za-z0-9\-\(\)]*",
    ]
    court_pattern = r"\b(Supreme Court|SCOTUS|district court|circuit court|court of appeals)\b"

    for pattern in section_patterns:
        entities["sections"].extend(re.findall(pattern, text, flags=re.IGNORECASE))

    entities["acts"].extend(re.findall(act_pattern, text))
    for pattern in citation_patterns:
        entities["citations"].extend(re.findall(pattern, text, flags=re.IGNORECASE))
    entities["courts"].extend(re.findall(court_pattern, text, flags=re.IGNORECASE))

    for key in entities:
        entities[key] = sorted(set(entities[key]))

    return entities


def infer_legal_context(text: str, entities: Dict[str, List[str]], unknown_abbr: List[str]) -> Dict[str, Any]:
    """
    Helps prevent weak restructuring.
    If the query includes legal hints, preserve that signal.
    """
    text_lower = text.lower()
    hint_count = sum(1 for term in LEGAL_HINT_TERMS if term in text_lower)

    has_entities = any(len(v) > 0 for v in entities.values())
    likely_legal = hint_count >= 1 or has_entities or len(unknown_abbr) > 0

    return {
        "likely_legal": likely_legal,
        "hint_count": hint_count,
    }


def should_use_llm_fallback(query: str) -> bool:
    """
    Use LLM only for short/medium awkward legal queries.
    Existing process stays the same; this is only a fallback.
    """
    try:
        if not query:
            return False

        q = query.lower().strip()

        if any(q.startswith(starter) for starter in QUESTION_STARTERS):
            return False

        if len(q.split()) > 15:
            return False

        has_legal_signal = any(term in q for term in [
            "law", "legal", "act", "rule", "section", "title",
            "usc", "cfr", "contract", "privacy", "compliance",
            "liability", "breach", "employment", "hipaa", "ada",
            "ssn", "[name]", "[phone]", "[ssn]", "implications", "advice"
        ])

        awkward_patterns = [
            r"\badvice legal\b",
            r"\bssn\b.*\bcheck\b",
            r"\bcheck\b$",
            r"\bfor\s+\[name\]\s+ssn\b",
            r"\bimplications\b.*\bcheck\b",
        ]

        awkward = any(re.search(p, q, flags=re.IGNORECASE) for p in awkward_patterns)

        return has_legal_signal and awkward

    except Exception:
        return False


def llm_rewrite_query(query: str) -> str:
    """
    Optional Ollama fallback to normalize awkward legal queries.
    Safe mode:
    - skip PII-heavy queries
    - do not allow placeholder-token corruption
    - only accept clearly usable rewrites
    """
    try:
        model = get_rewriter()
        if not model:
            return query

        # skip LLM rewrite for PII-heavy inputs
        placeholder_count = len(re.findall(r"\[(?:NAME|EMAIL|PHONE|SSN|ZIP)\]", query, flags=re.IGNORECASE))
        if placeholder_count >= 2:
            return query

        protected_query, protected_map = protect_placeholders(query)

        prompt = (
            "Rewrite this into a short, natural legal search query.\n"
            "Rules:\n"
            "- Keep the legal meaning.\n"
            "- Do not invent facts.\n"
            "- Do not answer the question.\n"
            "- Do not include labels like 'Query:' or 'Rewritten query:'.\n"
            "- Return only the rewritten query.\n\n"
            f"Query: {protected_query}"
        )

        response = requests.post(
            model["base_url"],
            json={
                "model": model["model"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            rewritten = data.get("response", "").strip()

            if rewritten:
                rewritten = re.sub(
                    r'^\s*(rewritten query:|query:)\s*',
                    '',
                    rewritten,
                    flags=re.IGNORECASE
                ).strip()

                rewritten = restore_placeholders(rewritten, protected_map)

                # reject broken placeholder outputs like PH_0 / PH_1 / PH_2
                if re.search(r'\bPH_\d+\b', rewritten, flags=re.IGNORECASE):
                    return query

                # reject outputs that are too short or obviously worse
                if len(rewritten.split()) < 2:
                    return query

                return rewritten

        return query

    except Exception:
        return query

def filter_irrelevant_placeholders(text: str) -> str:
    """
    Keep only legally meaningful placeholders.
    Remove noise placeholders like NAME, EMAIL, PHONE.
    """

    # placeholders to REMOVE (noise)
    remove_tags = ["NAME", "EMAIL", "PHONE", "ZIP"]

    # placeholders to KEEP (legally meaningful)
    keep_tags = ["SSN", "PHI", "DOB"]

    def replacer(match):
        tag = match.group(1).upper()
        if tag in keep_tags:
            return match.group(0)  # keep
        return ""  # remove

    cleaned = re.sub(r'\[(\w+)\]', replacer, text)

    # clean spacing
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(" ,.-")

    return cleaned


def restructure_query(text: str, entities: dict, unknown_abbr: list):
    """
    Conservative restructuring:
    - preserve meaningful queries
    - improve only when clearly helpful
    - avoid collapsing PII-masked queries
    - use LLM fallback only for safe non-PII awkward cases
    """

    try:
        if not text:
            return text

        q = text.strip()
        q = re.sub(r'^\s*query\s*:\s*', '', q, flags=re.IGNORECASE).strip()
        q_lower = q.lower()

        meaningful_starts = (
            "what", "how", "why", "when", "can", "does", "is", "are",
            "explain", "describe", "define", "compare", "list", "show",
            "find", "give", "provide", "summarize"
        )

        def count_placeholders(value: str) -> int:
            return len(re.findall(r"\[(?:NAME|EMAIL|PHONE|SSN|ZIP)\]", value, flags=re.IGNORECASE))

        def remove_filler_only(value: str) -> str:
            cleaned = value

            # remove polite filler only
            cleaned = re.sub(
                r'\b(please|kindly|assist me|i need help|need help)\b',
                '',
                cleaned,
                flags=re.IGNORECASE
            )

            # normalize whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned).strip(" ,.-")
            return cleaned

        def get_intent_prefix(q_low: str) -> str:
            if any(k in q_low for k in ["help", "assist", "guide", "support"]):
                return "Provide legal guidance on"

            if any(k in q_low for k in ["what is", "define", "meaning of"]):
                return "Explain the legal meaning of"

            if any(k in q_low for k in ["penalty", "penalties", "fine", "punishment", "violation", "violations"]):
                return "Explain the legal consequences of"

            if any(k in q_low for k in ["how to", "process", "steps", "procedure", "file", "filing"]):
                return "Explain the legal procedure for"

            if any(k in q_low for k in ["rights", "duties", "obligations", "responsibilities"]):
                return "Explain the legal rights and obligations in"

            if any(k in q_low for k in ["contract", "agreement", "breach", "damages", "liability"]):
                return "Explain the legal aspects of"

            if any(k in q_low for k in ["rule", "rules", "compliance", "regulation", "policy", "tax"]):
                return "Provide legal guidance on"

            return "Explain the legal aspects of"

        def should_use_restructure_llm(original_q: str, candidate_q: str) -> bool:
            oq = (original_q or "").lower().strip()
            cq = (candidate_q or "").lower().strip()

            # never use LLM when placeholders dominate
            if count_placeholders(original_q) >= 2:
                return False

            awkward_patterns = [
                r"^explain help\b",
                r"^what is help\b",
                r"\bwith with\b",
                r"\bfor for\b",
                r"\bon with\b",
            ]

            if any(re.search(p, cq, flags=re.IGNORECASE) for p in awkward_patterns):
                return True

            # short awkward fragments only
            if len(cq.split()) <= 6 and not cq.startswith(meaningful_starts):
                return True

            # do not rewrite if candidate already looks cleaner than original
            if len(cq) >= max(8, len(oq) * 0.6):
                return False

            return False

        # 1. Preserve already meaningful queries
        if q_lower.startswith(meaningful_starts):
            return q

        has_entities = any(len(v) > 0 for v in entities.values())

        has_legal_terms = any(k in q_lower for k in [
            "law", "act", "rule", "section", "title",
            "usc", "cfr", "contract", "privacy",
            "compliance", "liability", "breach", "advice",
            "tax", "penalty", "violation", "rights", "duties",
            "filing", "employment"
        ])

        placeholder_count = count_placeholders(q)
        cleaned_q = remove_filler_only(q)
        cleaned_q = filter_irrelevant_placeholders(cleaned_q)
        cleaned_q_lower = cleaned_q.lower()

        # 2. If query is PII-heavy, do not aggressively rewrite it.
        # Keep expanded query if it is already clearer than any restructure.
        if placeholder_count >= 2:
            # remove only a leading "help" if present, but preserve the rest
            pii_safe = re.sub(r'^\s*help\b', '', cleaned_q, flags=re.IGNORECASE).strip(" ,.-")
            pii_safe = re.sub(r'\s+', ' ', pii_safe).strip()

            # if removing help makes it too awkward, keep original expanded text
            if not pii_safe or len(pii_safe.split()) < 4:
                return q

            # for mixed personal + legal topic queries, prefer the topic-focused suffix when obvious
            m = re.search(r'\b(with|regarding|about)\b\s+(.+)$', pii_safe, flags=re.IGNORECASE)
            if m:
                topic = m.group(2).strip(" ,.-")
                if topic and len(topic.split()) >= 2:
                    prefix = get_intent_prefix(topic.lower())
                    return f"{prefix} {topic}"

            return q

        # 3. Short queries
        if len(cleaned_q.split()) <= 5:
            if unknown_abbr:
                candidate = f"Explain the legal meaning of {cleaned_q}"
            elif "section" in cleaned_q_lower or "usc" in cleaned_q_lower or "cfr" in cleaned_q_lower:
                candidate = f"What does {cleaned_q} refer to in US law"
            elif has_entities or has_legal_terms:
                prefix = get_intent_prefix(cleaned_q_lower)
                candidate = f"{prefix} {cleaned_q}"
            else:
                candidate = f"What is {cleaned_q}"

            if should_use_restructure_llm(q, candidate):
                try:
                    llm_candidate = llm_rewrite_query(cleaned_q)
                    if llm_candidate and len(llm_candidate.strip()) >= len(candidate.strip()) * 0.7:
                        return llm_candidate
                except Exception:
                    pass

            return candidate

        # 4. Medium queries
        if len(cleaned_q.split()) <= 12:
            if has_entities:
                candidate = f"Explain the legal meaning of {cleaned_q}"
            elif has_legal_terms:
                prefix = get_intent_prefix(cleaned_q_lower)
                candidate = f"{prefix} {cleaned_q}"
            else:
                candidate = cleaned_q

            if should_use_restructure_llm(q, candidate):
                try:
                    llm_candidate = llm_rewrite_query(cleaned_q)
                    if llm_candidate and len(llm_candidate.strip()) >= len(candidate.strip()) * 0.7:
                        return llm_candidate
                except Exception:
                    pass

            return candidate

        # 5. Long queries → keep as-is
        return cleaned_q

    except Exception:
        return text

def format_restructured_query(query: str):
    try:
        if not query:
            return query

        q = query.strip()

        # remove duplicate punctuation
        q = re.sub(r'[?.!]+$', '', q)

        # capitalize properly
        q = q[0].upper() + q[1:] if len(q) > 1 else q.upper()

        question_words = (
            "what", "why", "how", "when", "where", "who",
            "can", "is", "are", "do", "does", "should"
        )

        if any(q.lower().startswith(w) for w in question_words):
            q += "?"
        else:
            q += "."

        return q

    except Exception:
        return query


def preprocess_query(data: dict) -> dict:
    """
    Backward-compatible high-level preprocessing.
    Keeps your current response shape while adding more signals.
    """
    try:
        raw_query = data.get("query", "")

        cleaned = clean_text(raw_query)

        # spelling correction (same process, safer internals)
        cleaned = correct_spelling(cleaned)

        # lightweight phrase repair (same process, inserted before expansion)
        cleaned = repair_common_legal_phrases(cleaned)

        # optional LLM fallback (same flow, only for awkward legal queries)
        if should_use_llm_fallback(cleaned):
            cleaned = llm_rewrite_query(cleaned)

        # Optional: keep your existing learning hook
        # extract_abbreviations(cleaned)

        expanded_text, abbr_map, unknown_abbr = expand_abbreviations(cleaned)

        entities = extract_entities(expanded_text)

        restructured = restructure_query(expanded_text, entities, unknown_abbr)
        restructured = format_restructured_query(restructured)

        return {
            "original_query": raw_query,
            "cleaned_query": cleaned,
            "expanded_query": expanded_text,
            "restructured_query": restructured,
            "abbreviations": abbr_map,
            "unknown_abbreviations": unknown_abbr,
            "entities": entities,
        }

    except Exception as e:
        return {
            "error": "preprocessing_failed",
            "details": str(e),
        }