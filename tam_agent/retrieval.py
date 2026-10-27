"""Small, dependency-free helpers used by the tools: text search and label cleaning.

Nothing in here talks to the LLM. Keeping these helpers separate means they can be
tested on their own (see tests/test_tools.py).
"""
import math
import re
from collections import Counter
from difflib import SequenceMatcher

# ---------------------------------------------------------------- module names
# tickets.csv `module` is free text. This map turns every label seen in the data
# into one of the 8 canonical module names used by accounts.csv and the KB.
# It was checked against the data: for every label, the KB articles that share the
# ticket's error code all belong to the mapped module.
CANONICAL_MODULES = ["Contracts", "Integrations", "Platform", "Sourcing",
                     "Spend Analytics", "Supplier Management", "eInvoice", "eProcurement"]

_ALIASES = {
    "ap": "eInvoice", "einv": "eInvoice", "e-invoice": "eInvoice", "invoicing": "eInvoice",
    "api": "Integrations", "erp integration": "Integrations", "connector": "Integrations",
    "clm": "Contracts", "contract mgmt": "Contracts", "contract": "Contracts",
    "eproc": "eProcurement", "e-procurement": "eProcurement", "procurement": "eProcurement",
    "rfx": "Sourcing", "esourcing": "Sourcing", "sourcing events": "Sourcing",
    "sim": "Supplier Management", "supplier mgmt": "Supplier Management",
    "supplier portal": "Supplier Management",
    "sso": "Platform", "login": "Platform",
    "analytics": "Spend Analytics", "reports": "Spend Analytics", "spend": "Spend Analytics",
}
_BY_LOWER = {m.lower(): m for m in CANONICAL_MODULES}


def canonical_module(raw: str) -> str | None:
    """'CLM' -> 'Contracts', 'e-Invoice' -> 'eInvoice'. Blank/unknown -> None (never guess)."""
    key = (raw or "").strip().lower()
    return _BY_LOWER.get(key) or _ALIASES.get(key)


# ---------------------------------------------------------------- error codes
_ERROR_CODE = re.compile(r"\bE-\d{4}\b")


def extract_error_code(*texts: str) -> str | None:
    """First error code like E-6105 found in the given texts (the CSV column is sometimes blank)."""
    for text in texts:
        m = _ERROR_CODE.search(text or "")
        if m:
            return m.group(0)
    return None


# ---------------------------------------------------------------- text search
_STOPWORDS = set("""a an and are as at be been but by can could do does for from had has have how i if in into is it
its me my no not of on or our please so that the their them then there these they this to u us was we were what when
which who will with would you your hi hello thanks thank team""".split())


def _stem(token: str) -> str:
    """Crude stemming: keep the first 6 letters so approve/approval/approvals all become 'approv'."""
    if token.startswith("e-") or token.isdigit():
        return token
    return token[:6]


def tokenize(text: str) -> list[str]:
    """Lower-case, stemmed word tokens; error codes stay whole ('e-6105') so they match exactly."""
    text = (text or "").lower()
    tokens = re.findall(r"e-\d{4}|[a-z0-9]+", text)
    return [_stem(t) for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25:
    """Classic BM25 ranking over a fixed list of documents (each a list of tokens).

    BM25 scores a document higher when it contains the query words, especially rare
    words, without over-rewarding long documents. ~30 lines, no library needed.
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(d) for d in docs]
        self.avg_len = sum(len(d) for d in docs) / max(len(docs), 1)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = []
        for tf, doc in zip(self.tf, self.docs):
            norm = self.k1 * (1 - self.b + self.b * len(doc) / (self.avg_len or 1))
            s = 0.0
            for t in set(query_tokens):
                if t in tf:
                    s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
            out.append(s)
        return out


def jaccard(a: list[str], b: list[str]) -> float:
    """Word overlap between two texts, 0..1 (used to compare ticket descriptions)."""
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


# ---------------------------------------------------------------- company names
_LEGAL_SUFFIXES = {"inc", "llc", "ltd", "limited", "corp", "corporation", "co", "plc", "gmbh", "sa", "ag", "bv"}


def normalize_company(name: str) -> str:
    """'Clearwater Industries Inc.' -> 'clearwater industries'. Keeps words like 'Group'/'Europe'."""
    words = re.findall(r"[a-z0-9]+", (name or "").lower())
    return " ".join(w for w in words if w not in _LEGAL_SUFFIXES)


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_company(a), normalize_company(b)).ratio()


def clean_domain(value: str) -> str:
    """Accepts 'a@x.example', 'https://www.x.example/path' or 'x.example' -> 'x.example'."""
    v = (value or "").strip().lower()
    v = v.split("@")[-1]
    v = re.sub(r"^https?://", "", v).split("/")[0]
    return v[4:] if v.startswith("www.") else v
