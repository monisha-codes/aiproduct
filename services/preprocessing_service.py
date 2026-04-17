import re
import requests
from typing import Dict, List, Tuple, Any
from spellchecker import SpellChecker
from rapidfuzz import process, fuzz

ABBREVIATIONS = {
    "ADA": "Americans with Disabilities Act",
    "USC": "United States Code",
    "U.S.C.": "United States Code",
    "CFR": "Code of Federal Regulations",
    "C.F.R.": "Code of Federal Regulations",
    "HIPAA": "Health Insurance Portability and Accountability Act",
    "GDPR": "General Data Protection Regulation",
    "CCPA": "California Consumer Privacy Act",
    "FDCPA": "Fair Debt Collection Practices Act",
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
    "breach", "contract", "privacy", "compliance", "federal", "state",
    "gdpr", "hipaa", "eeoc", "penalty", "violation", "employment", "rights"
}

spell = SpellChecker()
rewriter = None

LEGAL_WHITELIST = {
    "hipaa", "usc", "cfr", "fdcpa", "ada", "eeoc", "ferpa", "gdpr", "ccpa",
    "coppa", "sox", "dmca", "ftc", "doj", "scotus",
    "flsa", "fmla", "erisa", "rico", "irs", "sec", "epa",
    "employment", "liability", "breach", "contract", "privacy",
    "compliance", "regulation", "statute", "damages", "plaintiff",
    "defendant", "court", "federal", "state", "implications", "check",
    "legal", "advice", "regarding", "implication", "meaning", "ssn",
    "penalty", "penalties", "violation", "violations", "rights", "duties",
    "law", "rules", "rule", "section", "title", "discrimination", "retaliation"
}

LEGAL_CANONICAL_TERMS = {
    "legal", "law", "contract", "breach", "privacy", "regulation", "compliance",
    "statute", "liability", "damages", "employment", "discrimination",
    "retaliation", "penalty", "penalties", "violation", "violations",
    "rights", "duties", "rules", "rule", "section", "title", "tax",
    "gdpr", "hipaa", "eeoc", "usc", "cfr", "agreement", "consumer"
}

PLACEHOLDER_PATTERN = r"\[(NAME|PHONE|EMAIL|SSN|ZIP)\]"


def protect_legal_tokens(text: str):
    """
    Protect legal abbreviations and placeholders before spell correction.
    This prevents tokens like 'ssn' -> 'son' and 'gdpr' -> 'gear'.
    """
    protected = {}
    words = text.split()

    legal_tokens = {k.replace(".", "").upper() for k in ABBREVIATIONS.keys()}
    legal_tokens.update({"SSN", "PHI", "PII"})

    new_words = []
    for i, w in enumerate(words):
        normalized = re.sub(r"[^A-Za-z\[\]\.]", "", w).upper().replace(".", "")

        if normalized in legal_tokens or (w.startswith("[") and w.endswith("]")):
            key = f"__PROT_{i}__"
            protected[key] = w
            new_words.append(key)
        else:
            new_words.append(w)

    return " ".join(new_words), protected


def restore_legal_tokens(text: str, protected: dict):
    for k, v in protected.items():
        text = text.replace(k, v)
    return text


def correct_abbreviation_typos_in_preprocessing(query: str):
    """
    Correct likely legal abbreviation typos before generic spell correction.
    Example:
    hipa -> HIPAA
    gdpr -> GDPR
    uscc -> USC
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

            if not clean or len(clean) < 3 or clean in {"what", "why", "how", "when", "who", "is", "are"}:
                corrected_tokens.append(raw)
                continue

            if clean in known_abbr_map:
                corrected_tokens.append(known_abbr_map[clean])
                continue

            match = process.extractOne(clean, known_abbr_keys, scorer=fuzz.ratio)
            if match:
                candidate, score, _ = match
                if score >= 84:
                    fixed = known_abbr_map[candidate]
                    corrected_tokens.append(fixed)
                    corrections[raw] = fixed
                    continue

            corrected_tokens.append(raw)

        return " ".join(corrected_tokens), corrections

    except Exception:
        return query, {}


def correct_legal_term_typos(query: str):
    """
    Correct typo-heavy legal words before generic spell correction.
    Examples:
    brech -> breach
    contrct -> contract
    """
    try:
        if not query:
            return query, {}

        tokens = query.split()
        corrected_tokens = []
        corrections = {}

        candidates = list(LEGAL_CANONICAL_TERMS)

        for token in tokens:
            raw = token

            if raw.startswith("__PH_") or raw.startswith("__PROT_"):
                corrected_tokens.append(raw)
                continue

            clean = re.sub(r"[^a-zA-Z]", "", raw).lower()

            if not clean or len(clean) < 4:
                corrected_tokens.append(raw)
                continue

            if clean in LEGAL_WHITELIST or clean in LEGAL_CANONICAL_TERMS:
                corrected_tokens.append(raw)
                continue

            match = process.extractOne(clean, candidates, scorer=fuzz.ratio)
            if match:
                candidate, score, _ = match

                # conservative threshold
                if score >= 86:
                    # avoid turning likely names/random words into legal terms
                    if clean[0] == candidate[0] or len(clean) <= 6:
                        corrected_tokens.append(candidate)
                        corrections[raw] = candidate
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
            r"\bmeaning of meaning of\b": "meaning of",
            r"\bcontact law\b": "contract law",
            r"\bgear rules\b": "GDPR rules",
        }

        for pattern, replacement in replacements.items():
            q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)

        q = re.sub(r"\bof of\b", "of", q, flags=re.IGNORECASE)
        q = re.sub(r"\bwith with\b", "with", q, flags=re.IGNORECASE)
        q = re.sub(r"\s+", " ", q).strip()

        return q

    except Exception:
        return text


def correct_spelling(query: str):
    try:
        if not query:
            return query

        # Step 1: protect placeholders like [SSN]
        protected_query, protected_map = protect_placeholders(query)

        # Step 2: correct legal abbreviation typos first
        protected_query, abbr_corrections = correct_abbreviation_typos_in_preprocessing(protected_query)

        # Step 3: protect legal tokens like ssn, usc, gdpr, hipaa before spell correction
        protected_query, legal_token_map = protect_legal_tokens(protected_query)

        # Step 4: correct typo-heavy legal words before generic spellchecker
        protected_query, legal_term_corrections = correct_legal_term_typos(protected_query)

        words = protected_query.split()
        corrected_words = []

        normalized_abbr_keys = {k.lower().replace(".", "") for k in ABBREVIATIONS.keys()}

        for word in words:
            if (word.startswith("__PH_") and word.endswith("__")) or (word.startswith("__PROT_") and word.endswith("__")):
                corrected_words.append(word)
                continue

            clean = re.sub(r"[^a-zA-Z_]", "", word).lower()

            if not clean or len(clean) <= 2:
                corrected_words.append(word)
                continue

            if clean in LEGAL_WHITELIST:
                corrected_words.append(word)
                continue

            if clean.replace(".", "") in normalized_abbr_keys:
                corrected_words.append(word)
                continue

            if word in abbr_corrections.values() or word in legal_term_corrections.values():
                corrected_words.append(word)
                continue

            if word.isupper():
                corrected_words.append(word)
                continue

            if 3 <= len(clean) <= 6:
                match = process.extractOne(clean, list(normalized_abbr_keys), scorer=fuzz.ratio)
                if match and match[1] >= 78:
                    corrected_words.append(word)
                    continue

            correction = spell.correction(clean)

            # reject harmful generic corrections for legal-like words
            if correction in {"gear", "contact", "son"} and clean in {"gdpr", "contrct", "ssn"}:
                corrected_words.append(word)
                continue

            if correction:
                if word.istitle():
                    correction = correction.title()
                corrected_words.append(correction)
            else:
                corrected_words.append(word)

        corrected = " ".join(corrected_words)

        # Step 5: restore legal tokens
        corrected = restore_legal_tokens(corrected, legal_token_map)

        # Step 6: restore placeholders
        corrected = restore_placeholders(corrected, protected_map)

        corrected = repair_common_legal_phrases(corrected)

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

    text = re.sub(r"[^\w\s\?\.\,\(\)\-\[\]§:/&']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
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

def normalize_legal_citation_shortcuts(text: str) -> str:
    """
    Convert citation-style shortcuts before generic abbreviation expansion.
    Example:
    - sec 1983 usc -> section 1983 USC
    - sec. 101 cfr -> section 101 CFR
    """
    if not text:
        return text

    q = text

    # sec 1983 -> section 1983
    q = re.sub(r'\bsec\.?\s+(\d+[A-Za-z0-9\-\(\)]*)\b', r'section \1', q, flags=re.IGNORECASE)

    return q

def expand_abbreviations(text: str) -> Tuple[str, Dict[str, str], List[str]]:
    """
    Expand known abbreviations.
    Preserve unknown abbreviations separately instead of dropping them.
    """
    mapping: Dict[str, str] = {}
    expanded = text

    # prefer longer keys first (e.g. U.S.C. before USC)
    for short, full in sorted(ABBREVIATIONS.items(), key=lambda x: len(x[0]), reverse=True):
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
    Give LLM higher priority for awkward, typo-heavy, or fragment-style legal queries.
    """
    try:
        if not query:
            return False

        q = query.lower().strip()

        if len(q.split()) > 20:
            return False

        placeholder_count = len(re.findall(r"\[(?:NAME|EMAIL|PHONE|SSN|ZIP)\]", q, flags=re.IGNORECASE))
        if placeholder_count >= 2:
            return False

        has_legal_signal = any(term in q for term in [
            "law", "legal", "act", "rule", "rules", "section", "title",
            "usc", "cfr", "contract", "privacy", "compliance",
            "liability", "breach", "employment", "hipaa", "ada",
            "gdpr", "eeoc", "ssn", "implications", "advice", "rights", "tax"
        ])

        awkward_patterns = [
            r"\badvice legal\b",
            r"\bmeaning of\b",
            r"\bhelp me with\b",
            r"\bcheck\b$",
            r"\bimplications\b.*\bcheck\b",
            r"\bbrech\b",
            r"\bcontrct\b",
        ]

        awkward = any(re.search(p, q, flags=re.IGNORECASE) for p in awkward_patterns)

        # prefer LLM for short fragments that are not natural questions
        if has_legal_signal and (awkward or not any(q.startswith(starter) for starter in QUESTION_STARTERS)):
            return True

        return False

    except Exception:
        return False


def llm_rewrite_query(query: str) -> str:
    """
    Optional Ollama rewrite.
    Preserves placeholders and legal meaning.
    """
    try:
        model = get_rewriter()
        if not model:
            return query

        placeholder_count = len(re.findall(r"\[(?:NAME|EMAIL|PHONE|SSN|ZIP)\]", query, flags=re.IGNORECASE))
        if placeholder_count >= 2:
            return query

        protected_query, protected_map = protect_placeholders(query)

        prompt = (
            "Rewrite this into a short, natural, meaningful legal search query.\n"
            "Rules:\n"
            "- Keep the legal meaning.\n"
            "- Fix obvious spelling mistakes in legal terms.\n"
            "- Preserve placeholders exactly.\n"
            "- Preserve legal citations and abbreviations correctly.\n"
            "- Do not expand abbreviations incorrectly based only on ambiguity.\n"
            "- Example: 'sec 1983 usc' should mean section 1983 of the U.S. Code, not Securities and Exchange Commission.\n"
            "- Do not invent facts.\n"
            "- Do not answer the question.\n"
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
            timeout=6
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

                if re.search(r'\bPH_\d+\b', rewritten, flags=re.IGNORECASE):
                    return query

                if len(rewritten.split()) < 2:
                    return query

                return repair_common_legal_phrases(rewritten)

        return query

    except Exception:
        return query


def filter_irrelevant_placeholders(text: str) -> str:
    """
    Keep only legally meaningful placeholders.
    Remove noise placeholders like NAME, EMAIL, PHONE.
    """
    keep_tags = ["SSN", "PHI", "DOB"]

    def replacer(match):
        tag = match.group(1).upper()
        if tag in keep_tags:
            return match.group(0)
        return ""

    cleaned = re.sub(r'\[(\w+)\]', replacer, text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(" ,.-")
    return cleaned


def normalize_rewrite_subject(text: str) -> str:
    """
    Remove duplicated lead phrases that cause meaningless framing.
    """
    q = text.strip()

    q = re.sub(r'^\s*meaning of\s+', '', q, flags=re.IGNORECASE)
    q = re.sub(r'^\s*legal meaning of\s+', '', q, flags=re.IGNORECASE)
    q = re.sub(r'^\s*explain\s+', '', q, flags=re.IGNORECASE)
    q = re.sub(r'^\s*define\s+', '', q, flags=re.IGNORECASE)
    q = re.sub(r'^\s*help me with\s+', '', q, flags=re.IGNORECASE)
    q = re.sub(r'^\s*help with\s+', '', q, flags=re.IGNORECASE)

    q = re.sub(r'\s+', ' ', q).strip(" ,.-")
    return q

def strip_trailing_instruction_words(text: str) -> str:
    """
    Remove trailing instruction words that users append at the end,
    like:
    - sec 1983 usc explain
    - hipaa define
    - gdpr meaning
    """
    if not text:
        return text

    q = text.strip()

    q = re.sub(
        r'\b(explain|define|description|describe|meaning|summarize|summary)\b\s*$',
        '',
        q,
        flags=re.IGNORECASE
    )

    q = re.sub(r'\s+', ' ', q).strip(" ,.-")
    return q

def needs_legal_prefix(query: str) -> bool:
    """
    Return True for short/medium legal noun-phrase queries that are not already
    natural questions or explicit instructions.
    """
    if not query:
        return False

    q = query.strip().lower()

    meaningful_starts = (
        "what", "how", "why", "when", "can", "does", "is", "are",
        "explain", "describe", "define", "compare", "list", "show",
        "find", "give", "provide", "summarize"
    )

    if q.startswith(meaningful_starts):
        return False

    legal_noun_terms = [
        "law", "legal", "contract", "breach", "privacy", "liability",
        "damages", "violation", "penalty", "penalties", "rights",
        "duties", "employment", "tax", "compliance", "regulation",
        "gdpr", "hipaa", "eeoc", "usc", "cfr", "section", "title",
        "agreement", "consumer", "discrimination", "retaliation"
    ]

    return any(term in q for term in legal_noun_terms)

def should_use_llm_as_primary_rewriter(query: str) -> bool:
    """
    Use LLM only for difficult legal fragments.
    This reduces latency while keeping quality.
    """
    try:
        if not query:
            return False

        q = query.strip().lower()

        if is_meaningless_query(q):
            return False

        placeholder_count = len(re.findall(r"\[(?:NAME|EMAIL|PHONE|SSN|ZIP)\]", q, flags=re.IGNORECASE))
        if placeholder_count >= 2:
            return False

        meaningful_starts = (
            "what", "how", "why", "when", "can", "does", "is", "are",
            "explain", "describe", "define", "compare", "list", "show",
            "find", "give", "provide", "summarize"
        )
        if q.startswith(meaningful_starts):
            return False

        token_count = len(q.split())

        typo_like_patterns = [
            r"\bbrech\b",
            r"\bcontrct\b",
            r"\bcontrat\b",
            r"\bprvacy\b",
            r"\bcmpliance\b",
            r"\bhipa\b",
            r"\buscc\b",
            r"\bcfrr\b",
        ]

        ambiguous_patterns = [
            r"\bsection\s+\d+\s+usc\b",
            r"\bsection\s+\d+\s+cfr\b",
            r"\b\d+\s+usc\b",
            r"\b\d+\s+cfr\b",
        ]

        typo_like = any(re.search(p, q, flags=re.IGNORECASE) for p in typo_like_patterns)
        ambiguous = any(re.search(p, q, flags=re.IGNORECASE) for p in ambiguous_patterns)

        legal_terms = [
            "law", "legal", "contract", "breach", "privacy", "tax",
            "rule", "rules", "gdpr", "hipaa", "eeoc", "usc", "cfr",
            "section", "title", "violation", "penalty", "employment"
        ]
        legal_like = any(term in q for term in legal_terms)

        # only use LLM for short difficult fragments
        if token_count <= 8 and legal_like and (typo_like or ambiguous):
            return True

        return False

    except Exception:
        return False

def restructure_query(text: str, entities: dict, unknown_abbr: list):
    """
    LLM-primary restructuring:
    - preserve already meaningful natural questions
    - use LLM as primary rewriter for most short/medium legal queries
    - use rule-based prefixes only as fallback
    - avoid collapsing masked or meaningless text
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

        def get_intent_prefix(q_low: str) -> str:
            if any(k in q_low for k in ["penalty", "penalties", "fine", "punishment", "violation", "violations"]):
                return "Explain the legal consequences of"

            if any(k in q_low for k in ["how to", "process", "steps", "procedure", "file", "filing"]):
                return "Explain the legal procedure for"

            if any(k in q_low for k in ["rights", "duties", "obligations", "responsibilities"]):
                return "Explain the legal rights and obligations in"

            if any(k in q_low for k in ["rule", "rules", "compliance", "regulation", "policy", "tax"]):
                return "Explain"

            if any(k in q_low for k in ["contract", "agreement", "breach", "damages", "liability", "privacy"]):
                return "Explain the legal aspects of"

            if any(k in q_low for k in ["meaning", "define"]):
                return "Explain the legal meaning of"

            return "Explain"

        def looks_bad_candidate(candidate_q: str) -> bool:
            cq = (candidate_q or "").lower().strip()
            bad_patterns = [
                r"\bof of\b",
                r"\bmeaning of meaning\b",
                r"\bhelp me with\b",
                r"\bexplain explain\b",
                r"\bprovide legal guidance on help\b",
                r"^what is\s*\?$",
                r"^explain\s*\.$",
            ]
            return any(re.search(p, cq, flags=re.IGNORECASE) for p in bad_patterns)

        # preserve already meaningful natural questions
        if q_lower.startswith(meaningful_starts):
            return q

        placeholder_count = count_placeholders(q)

        # do not aggressively rewrite PII-heavy text
        if placeholder_count >= 2:
            return q

        cleaned_q = filter_irrelevant_placeholders(q)
        cleaned_q = normalize_rewrite_subject(cleaned_q)
        cleaned_q = strip_trailing_instruction_words(cleaned_q)
        cleaned_q = repair_common_legal_phrases(cleaned_q)
        cleaned_q_lower = cleaned_q.lower()
        subject = normalize_rewrite_subject(cleaned_q)
        subject = strip_trailing_instruction_words(subject)

        if not subject or is_meaningless_query(subject):
            return q

        has_entities = any(len(v) > 0 for v in entities.values())
        has_legal_terms = any(k in cleaned_q_lower for k in [
            "law", "act", "rule", "rules", "section", "title",
            "usc", "cfr", "contract", "privacy",
            "compliance", "liability", "breach", "advice",
            "tax", "penalty", "violation", "rights", "duties",
            "filing", "employment", "gdpr", "hipaa", "eeoc"
        ])

        # -------------------------------
        # 1. LLM PRIMARY PATH
        # -------------------------------
        if should_use_llm_as_primary_rewriter(cleaned_q):
            llm_candidate = llm_rewrite_query(cleaned_q)
            if llm_candidate and len(llm_candidate.split()) >= 2:
                llm_candidate = repair_common_legal_phrases(llm_candidate)
                llm_candidate = strip_trailing_instruction_words(llm_candidate)
                llm_candidate_lower = llm_candidate.lower().strip()
                llm_subject = normalize_rewrite_subject(llm_candidate)
                llm_subject = strip_trailing_instruction_words(llm_subject)

                if not looks_bad_candidate(llm_candidate):
                    # If LLM returns only a fragment, add prefix
                    if needs_legal_prefix(llm_subject) and not llm_candidate_lower.startswith(meaningful_starts):
                        prefix = get_intent_prefix(llm_candidate_lower)

                        if prefix.lower() == "explain":
                            prefix = "Explain the legal aspects of"

                        return f"{prefix} {llm_subject}"

                    return llm_candidate

        # -------------------------------
        # 2. RULE-BASED FALLBACK
        # -------------------------------
        if len(cleaned_q.split()) <= 6:
            if unknown_abbr:
                candidate = f"Explain the legal meaning of {subject}"

            elif "section" in cleaned_q_lower or "usc" in cleaned_q_lower or "cfr" in cleaned_q_lower:
                candidate = f"Explain the legal meaning of {subject}"

            elif needs_legal_prefix(subject):
                prefix = get_intent_prefix(cleaned_q_lower)

                if prefix.lower() == "explain":
                    prefix = "Explain the legal aspects of"

                candidate = f"{prefix} {subject}"

            elif has_entities or has_legal_terms:
                prefix = get_intent_prefix(cleaned_q_lower)
                candidate = f"{prefix} {subject}"

            else:
                candidate = f"What is {cleaned_q}"

            candidate = repair_common_legal_phrases(candidate)

            if not looks_bad_candidate(candidate):
                return candidate

            return cleaned_q

        if len(cleaned_q.split()) <= 14:
            if needs_legal_prefix(subject):
                prefix = get_intent_prefix(cleaned_q_lower)

                if prefix.lower() == "explain":
                    prefix = "Explain the legal aspects of"

                candidate = f"{prefix} {subject}"
                candidate = repair_common_legal_phrases(candidate)

                if not looks_bad_candidate(candidate):
                    return candidate

            if has_entities or has_legal_terms:
                prefix = get_intent_prefix(cleaned_q_lower)
                candidate = f"{prefix} {subject}"
                candidate = repair_common_legal_phrases(candidate)

                if not looks_bad_candidate(candidate):
                    return candidate

            return cleaned_q

        return cleaned_q

    except Exception:
        return text


def force_meaningful_legal_prefix(query: str) -> str:
    """
    Final safety layer:
    If restructure_query still returns only a legal noun phrase,
    add a meaningful legal prefix.
    """
    try:
        if not query:
            return query

        q = strip_trailing_instruction_words(query.strip())
        q_lower = q.lower()

        meaningful_starts = (
            "what", "how", "why", "when", "can", "does", "is", "are",
            "explain", "describe", "define", "compare", "list", "show",
            "find", "give", "provide", "summarize"
        )

        if q_lower.startswith(meaningful_starts):
            return q

        legal_terms = [
            "law", "legal", "contract", "breach", "privacy", "liability",
            "damages", "violation", "penalty", "penalties", "rights",
            "duties", "employment", "tax", "compliance", "regulation",
            "gdpr", "hipaa", "eeoc", "usc", "cfr", "section", "title",
            "agreement", "consumer", "discrimination", "retaliation"
        ]

        if not any(term in q_lower for term in legal_terms):
            return q

        if any(k in q_lower for k in ["section", "usc", "cfr", "title"]):
            return f"Explain the legal meaning of {q}"

        if any(k in q_lower for k in ["penalty", "penalties", "fine", "punishment", "violation", "violations"]):
            return f"Explain the legal consequences of {q}"

        if any(k in q_lower for k in ["rights", "duties", "obligations", "responsibilities"]):
            return f"Explain the legal rights and obligations in {q}"

        if any(k in q_lower for k in ["how to", "process", "steps", "procedure", "file", "filing"]):
            return f"Explain the legal procedure for {q}"

        if any(k in q_lower for k in ["rule", "rules", "compliance", "regulation", "policy", "tax"]):
            return f"Explain {q}"

        return f"Explain the legal aspects of {q}"

    except Exception:
        return query

def format_restructured_query(query: str):
    try:
        if not query:
            return query

        q = repair_common_legal_phrases(query.strip())
        q = re.sub(r'[?.!]+$', '', q)
        q = re.sub(r'\s+', ' ', q).strip()

        if not q:
            return query

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


def strip_trailing_instruction_words(text: str) -> str:
    """
    Remove trailing instruction words like:
    - explain
    - define
    - describe
    - meaning
    """
    if not text:
        return text

    q = text.strip()

    q = re.sub(
        r'\b(explain|define|describe|description|meaning|summarize|summary)\b\s*$',
        '',
        q,
        flags=re.IGNORECASE
    )

    q = re.sub(r'\s+', ' ', q).strip(" ,.-")
    return q


def preprocess_query(data: dict) -> dict:
    """
    Backward-compatible high-level preprocessing.
    Keeps current response shape unchanged.
    """
    try:
        raw_query = data.get("query", "")

        cleaned = clean_text(raw_query)

        # normalize legal citation shortcuts before abbreviation expansion
        cleaned = normalize_legal_citation_shortcuts(cleaned)

        # spelling correction
        cleaned = correct_spelling(cleaned)

        # phrase repair
        cleaned = repair_common_legal_phrases(cleaned)

        expanded_text, abbr_map, unknown_abbr = expand_abbreviations(cleaned)

        # remove trailing command words before entity extraction / restructuring
        expanded_text = strip_trailing_instruction_words(expanded_text)

        entities = extract_entities(expanded_text)

        restructured = restructure_query(expanded_text, entities, unknown_abbr)

        # remove trailing command words again in case any branch reintroduced them
        restructured = strip_trailing_instruction_words(restructured)

        restructured = force_meaningful_legal_prefix(restructured)
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