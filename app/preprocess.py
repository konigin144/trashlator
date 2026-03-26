from __future__ import annotations

import re


URLISH_PREFIXES = (
    "http",
    "https",
    "www",
)

URLISH_TLDS = (
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "io",
    "co",
    "uk",
    "de",
    "fr",
    "pl",
    "jp",
    "ru",
    "cn",
    "html",
)


def is_url_like_text(text: str) -> bool:
    """
    Heuristic detector for records that are basically URLs
    or URL-derived strings stripped of punctuation.

    Examples:
    - httpswwwinquirercomnews...
    - httpswwwwnycstudiosorgpodcasts...
    """
    if not text:
        return False

    normalized = text.strip().lower()
    if not normalized:
        return False

    # If there are spaces inside, it is less likely to be a single URL-ish blob.
    if " " in normalized:
        return False

    # Obvious starts
    if normalized.startswith(URLISH_PREFIXES):
        return True

    # Lots of URL-ish domain markers inside one long token
    tld_hits = sum(1 for tld in URLISH_TLDS if tld in normalized)
    if tld_hits >= 2 and len(normalized) >= 20:
        return True

    # Very long alnum blob containing common web-ish fragments
    web_markers = ("html", "www", "http", "https", "login", "account", "verify")
    marker_hits = sum(1 for marker in web_markers if marker in normalized)
    if marker_hits >= 2 and len(normalized) >= 25:
        return True

    return False