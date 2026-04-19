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
                if score >= 70:
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

        # Step 2: STRONG abbreviation correction FIRST (CRITICAL FIX)
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

            # 🚨 DO NOT spell-correct short uppercase-like tokens (likely abbreviations)
            if clean.isalpha() and len(clean) <= 5:
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
    Detect likely unknown abbreviations only.
    Avoid normal English words like TAX, DEFINE, OF, SECTION, RULES, etc.
    """
    if not text:
        return []

    candidates = set()

    ignored = {
        "NAME", "PHONE", "EMAIL", "SSN", "ZIP",
        "SECTION", "SECTIONS", "RULE", "RULES", "TITLE", "CHAPTER",
        "DEFINE", "EXPLAIN", "DESCRIBE", "SUMMARY", "SUMMARIZE",
        "LAW", "LAWS", "TAX", "OF", "AND", "OR", "THE", "A", "AN",
        "IN", "ON", "FOR", "TO", "WITH", "BY", "FROM", "AT"
    }

    ignored.update({w.upper() for w in QUESTION_STARTERS})
    ignored.update({w.upper() for w in LEGAL_HINT_TERMS})

    normalized_known = {
        re.sub(r"[^A-Z]", "", k.upper())
        for k in ABBREVIATIONS.keys()
    }

    # Match:
    # - plain uppercase abbreviations like GDPR, EEOC
    # - dotted abbreviations like U.S.C.
    matches = re.findall(r"\b(?:[A-Z]{2,6}|(?:[A-Z]\.){2,}[A-Z]?\.?)\b", text)

    for token in matches:
        normalized = re.sub(r"[^A-Z]", "", token.upper())

        if not normalized:
            continue

        if normalized in ignored:
            continue

        if normalized in normalized_known:
            continue

        if normalized.lower() in LEGAL_WHITELIST:
            continue

        if len(normalized) < 2 or len(normalized) > 6:
            continue

        candidates.add(token)

    return sorted(candidates)

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
    Avoid wrong SEC expansion in citation-style queries.
    """
    mapping: Dict[str, str] = {}
    expanded = text

    for short, full in sorted(ABBREVIATIONS.items(), key=lambda x: len(x[0]), reverse=True):
        # special handling for SEC to avoid:
        # sec 1983 usc -> Securities and Exchange Commission
        if short.upper() == "SEC":
            sec_citation_pattern = re.compile(r"\bsec\.?\s+\d+", re.IGNORECASE)
            if re.search(sec_citation_pattern, expanded):
                continue

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

def final_llm_fallback_rewrite(query: str) -> str:
    """
    Final safety net:
    if restructuring is still weak, use LLM on the raw user query.
    """
    try:
        if not query:
            return query

        rewritten = llm_rewrite_query(query)

        if rewritten and len(rewritten.split()) >= 3:
            rewritten = repair_common_legal_phrases(rewritten)
            rewritten = strip_trailing_instruction_words(rewritten)
            return rewritten.strip()

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


def is_meaningless_query(query: str) -> bool:
    if not query:
        return True

    q = query.strip().lower()

    if q in {"[name]", "[email]", "[phone]", "[ssn]", "[zip]"}:
        return True

    tokens = q.split()
    weak_tokens = {"[name]", "[email]", "[phone]", "[ssn]", "[zip]", "what", "is", "the", "a", "an"}
    if tokens and all(t in weak_tokens for t in tokens):
        return True

    return False

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
            if any(k in q_low for k in ["penalty", "penalties", "fine", "violation"]):
                return "What are the legal consequences of"

            if any(k in q_low for k in ["process", "steps", "procedure", "filing"]):
                return "What is the legal process for"

            if any(k in q_low for k in ["rights", "duties", "obligations"]):
                return "What are the legal rights related to"

            # IMPORTANT FIX
            if any(k in q_low for k in ["rule", "rules", "regulation", "compliance", "policy"]):
                return "Explain"

            if any(k in q_low for k in ["section", "usc", "cfr", "title"]):
                return "Explain"

            if any(k in q_low for k in ["law", "act"]):
                return "Explain"

            # ❌ REMOVE legal meaning default
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
                            prefix = "Explain"

                        return f"{prefix} {llm_subject}"

                    return llm_candidate

        # -------------------------------
        # 2. RULE-BASED FALLBACK
        # -------------------------------
        if len(cleaned_q.split()) <= 6:
            if unknown_abbr:
                candidate = f"Explain {subject}"

            elif "section" in cleaned_q_lower or "usc" in cleaned_q_lower or "cfr" in cleaned_q_lower:
                candidate = f"What does {subject} refer to in law"

            elif needs_legal_prefix(subject):
                prefix = get_intent_prefix(cleaned_q_lower)

                if prefix.lower() == "explain":
                    prefix = "Explain"

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
                    prefix = "Explain"

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
    If restructure_query still returns only a legal fragment,
    add a natural legal prefix without sounding robotic.
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
            "agreement", "consumer", "discrimination", "retaliation", "act"
        ]

        if not any(term in q_lower for term in legal_terms):
            return q

        # citation / section / code queries
        if any(k in q_lower for k in ["section", "usc", "cfr", "title"]):
            return f"Explain {q}"

        # law / act / regulation / rule queries
        if any(k in q_lower for k in ["law", "act", "regulation", "rule", "rules", "compliance", "policy"]):
            return f"Explain {q}"

        # rights / duties
        if any(k in q_lower for k in ["rights", "duties", "obligations", "responsibilities"]):
            return f"What are the legal rights and obligations related to {q}"

        # penalties / violations
        if any(k in q_lower for k in ["penalty", "penalties", "fine", "punishment", "violation", "violations"]):
            return f"What are the legal consequences of {q}"

        # procedure-like
        if any(k in q_lower for k in ["how to", "process", "steps", "procedure", "file", "filing"]):
            return f"What is the legal process for {q}"

        # default
        return f"Explain {q}"

    except Exception:
        return query

def format_restructured_query(query: str):
    """
    Final natural formatting:
    - sentence case, not title case for every word
    - preserve legal abbreviations like GDPR, HIPAA, EEOC, U.S.C., C.F.R.
    - keep output natural
    """
    try:
        if not query:
            return query

        q = repair_common_legal_phrases(query.strip())
        q = re.sub(r'[?.!]+$', '', q)
        q = re.sub(r'\s+', ' ', q).strip()

        if not q:
            return query

        # sentence-style lowercase first
        q = q.lower()

        # restore common legal abbreviations
        q = re.sub(r"\bgdpr\b", "GDPR", q, flags=re.IGNORECASE)
        q = re.sub(r"\bhipaa\b", "HIPAA", q, flags=re.IGNORECASE)
        q = re.sub(r"\beeoc\b", "EEOC", q, flags=re.IGNORECASE)
        q = re.sub(r"\bada\b", "ADA", q, flags=re.IGNORECASE)
        q = re.sub(r"\bccpa\b", "CCPA", q, flags=re.IGNORECASE)
        q = re.sub(r"\bfdcpa\b", "FDCPA", q, flags=re.IGNORECASE)
        q = re.sub(r"\bferpa\b", "FERPA", q, flags=re.IGNORECASE)
        q = re.sub(r"\bfmla\b", "FMLA", q, flags=re.IGNORECASE)
        q = re.sub(r"\bflsa\b", "FLSA", q, flags=re.IGNORECASE)
        q = re.sub(r"\bosha\b", "OSHA", q, flags=re.IGNORECASE)
        q = re.sub(r"\berisa\b", "ERISA", q, flags=re.IGNORECASE)
        q = re.sub(r"\brico\b", "RICO", q, flags=re.IGNORECASE)
        q = re.sub(r"\bdmca\b", "DMCA", q, flags=re.IGNORECASE)
        q = re.sub(r"\bcoppa\b", "COPPA", q, flags=re.IGNORECASE)
        q = re.sub(r"\bsox\b", "SOX", q, flags=re.IGNORECASE)
        q = re.sub(r"\bscotus\b", "SCOTUS", q, flags=re.IGNORECASE)
        q = re.sub(r"\bdoj\b", "DOJ", q, flags=re.IGNORECASE)
        q = re.sub(r"\bftc\b", "FTC", q, flags=re.IGNORECASE)
        q = re.sub(r"\birs\b", "IRS", q, flags=re.IGNORECASE)
        q = re.sub(r"\bsec\b", "SEC", q, flags=re.IGNORECASE)
        q = re.sub(r"\bepa\b", "EPA", q, flags=re.IGNORECASE)

        # restore dotted legal abbreviations
        q = re.sub(r"\bu\.?s\.?c\.?\b", "U.S.C.", q, flags=re.IGNORECASE)
        q = re.sub(r"\bc\.?f\.?r\.?\b", "C.F.R.", q, flags=re.IGNORECASE)

        # capitalize only the first character of the whole sentence
        q = q[0].upper() + q[1:]

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
    
def normalize_expanded_query_case(text: str) -> str:
    """
    Make expanded query human-readable:
    - full form in Title Case
    - abbreviation in uppercase inside brackets
    - remaining words in lowercase
    Example:
      General Data Protection Regulation (GDPR) rules
    """
    try:
        if not text:
            return text

        q = text.strip()
        q = re.sub(r"\s+", " ", q).strip()

        replacements = {
            r"general data protection regulation\s*\(gdpr\)": "General Data Protection Regulation (GDPR)",
            r"health insurance portability and accountability act\s*\(hipaa\)": "Health Insurance Portability and Accountability Act (HIPAA)",
            r"equal employment opportunity commission\s*\(eeoc\)": "Equal Employment Opportunity Commission (EEOC)",
            r"united states code\s*\(usc\)": "United States Code (USC)",
            r"code of federal regulations\s*\(cfr\)": "Code of Federal Regulations (CFR)",
            r"california consumer privacy act\s*\(ccpa\)": "California Consumer Privacy Act (CCPA)",
            r"family and medical leave act\s*\(fmla\)": "Family and Medical Leave Act (FMLA)",
            r"fair labor standards act\s*\(flsa\)": "Fair Labor Standards Act (FLSA)",
            r"occupational safety and health administration\s*\(osha\)": "Occupational Safety and Health Administration (OSHA)",
            r"americans with disabilities act\s*\(ada\)": "Americans with Disabilities Act (ADA)",
        }

        q = q.lower()
        for pattern, replacement in replacements.items():
            q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)

        # lowercase normal leftover words, but keep the already-restored title-case phrases
        q = re.sub(r"\s+", " ", q).strip()
        return q

    except Exception:
        return text


def restore_expanded_phrase_in_restructured(restructured: str, expanded_text: str, abbr_map: dict) -> str:
    """
    Restore proper casing and bracketed abbreviation form inside restructured query.
    Example:
      general data protection regulation gdpr rules
      -> General Data Protection Regulation (GDPR) rules
    """
    try:
        if not restructured:
            return restructured

        result = restructured

        for short, full in (abbr_map or {}).items():
            phrase_with_brackets = f"{full} ({short})"
            phrase_without_brackets = f"{full} {short}"

            result = re.sub(
                re.escape(phrase_without_brackets),
                phrase_with_brackets,
                result,
                flags=re.IGNORECASE
            )

            result = re.sub(
                re.escape(full.lower()),
                full,
                result,
                flags=re.IGNORECASE
            )

            result = re.sub(
                rf"\b{re.escape(short.lower())}\b",
                short,
                result,
                flags=re.IGNORECASE
            )

        result = re.sub(r"\s+", " ", result).strip()
        return result

    except Exception:
        return restructured
    

def preprocess_query(data: dict) -> dict:
    """
    Backward-compatible high-level preprocessing.
    Keeps current response shape unchanged.
    """
    try:
        raw_query = data.get("query", "")

        # 🚨 HARD STOP for meaningless placeholder-only queries
        if re.fullmatch(r"\[(NAME|EMAIL|PHONE|SSN|ZIP)\]", raw_query.strip(), flags=re.IGNORECASE):
            return {
                "original_query": raw_query,
                "cleaned_query": raw_query,
                "expanded_query": raw_query,
                "restructured_query": "",
                "abbreviations": {},
                "unknown_abbreviations": [],
                "entities": {},
            }

        cleaned = clean_text(raw_query)

        # normalize legal citation shortcuts before abbreviation expansion
        cleaned = normalize_legal_citation_shortcuts(cleaned)

        # spelling correction
        cleaned = correct_spelling(cleaned)

        # phrase repair
        cleaned = repair_common_legal_phrases(cleaned)

        expanded_text, abbr_map, unknown_abbr = expand_abbreviations(cleaned)

        # ✅ normalize expanded query style for display
        expanded_text = normalize_expanded_query_case(expanded_text)

        # remove trailing command words before entity extraction / restructuring
        expanded_text = strip_trailing_instruction_words(expanded_text)

        entities = extract_entities(expanded_text)

        restructured = restructure_query(expanded_text, entities, unknown_abbr)

        # remove trailing command words again in case any branch reintroduced them
        restructured = strip_trailing_instruction_words(restructured)

        restructured = force_meaningful_legal_prefix(restructured)

        # FINAL SAFETY NET:
        
        if (
            not restructured
            or len(restructured.split()) <= 3
            or restructured.lower().strip() in {"explain", "define", "law"}
            or restructured.lower().endswith(" explain")
            or "explain explain" in restructured.lower()
        ):
            restructured = final_llm_fallback_rewrite(raw_query)
            restructured = strip_trailing_instruction_words(restructured)
            restructured = force_meaningful_legal_prefix(restructured)

        # ✅ restore proper expanded abbreviation casing in final restructured query
        
        restructured = format_restructured_query(restructured)
        restructured = restore_expanded_phrase_in_restructured(restructured, expanded_text, abbr_map)

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


