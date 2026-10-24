"""HTTP client for SEC EDGAR, with fair-access guarantees built in.

Three obligations are enforced here rather than left to callers:

1. A descriptive ``User-Agent`` carrying a contact email, read from
   ``SEC_USER_AGENT``. The SEC blocks clients that omit it.
2. At most 10 requests per second (:mod:`acquirer_engine.edgar.ratelimit`).
3. Every response cached on disk (:mod:`acquirer_engine.edgar.cache`), so a rerun costs
   EDGAR nothing.

Setting ``EDGAR_OFFLINE=1`` turns the client into a cache-only reader that raises
on a miss. CI runs with it set, so a test that would touch the network fails
loudly instead of quietly making a request.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from acquirer_engine.edgar.cache import CacheEntry, DiskCache
from acquirer_engine.edgar.ratelimit import RateLimiter

logger = logging.getLogger(__name__)

EDGAR_HOST = "https://www.sec.gov"
DATA_HOST = "https://data.sec.gov"
FTS_HOST = "https://efts.sec.gov"

#: Full-text search pages 100 documents at a time and will not page past 10,000
#: for one query, so callers partition by date rather than paging indefinitely.
FTS_PAGE_SIZE = 100
FTS_MAX_REACHABLE = 10_000

#: Submissions and companyfacts JSON change as companies file; re-read daily.
MUTABLE_MAX_AGE = 24 * 60 * 60

#: Retry budget for a throttle or a 5xx. Full-text search returns sporadic 500s
#: under load -- observed on roughly one request in fifty, clearing within
#: seconds -- and a discovery sweep issues several hundred, so a short budget
#: means a run reliably dies partway through on an error that was never real.
#: Six attempts back off 1+2+4+8+16s, so roughly half a minute of patience.
#: Every successful response is cached, so a run that still fails resumes
#: almost free -- only the requests that never landed are reissued.
RETRY_ATTEMPTS = 6
RETRY_MAX_WAIT = 60

_PLACEHOLDER_EMAILS = ("example.com", "example.org", "your.name", "youremail")


class OfflineCacheMiss(RuntimeError):
    """Raised when offline mode is on and a URL is not in the cache."""


class ThrottledOrUnavailable(httpx.HTTPStatusError):
    """A retryable EDGAR response: 429 Too Many Requests, or any 5xx.

    Distinguished from an ordinary ``HTTPStatusError`` so that a 404 — which no
    amount of backoff will fix, and which is normal for a company with no XBRL
    facts — fails on the first attempt.
    """


class MissingUserAgent(RuntimeError):
    """Raised when ``SEC_USER_AGENT`` is absent or not a usable contact string."""


def validate_user_agent(user_agent: str | None) -> str:
    """Check that a User-Agent string satisfies SEC fair-access guidance.

    The SEC asks for a descriptive name plus a contact email. A placeholder
    address is worse than none, because it looks compliant while giving them no
    way to reach us, so it is rejected explicitly.

    Args:
        user_agent: Candidate string, typically from ``SEC_USER_AGENT``.

    Returns:
        The validated string, stripped.

    Raises:
        MissingUserAgent: If empty, missing an email, or still a placeholder.
    """
    candidate = (user_agent or "").strip()
    if not candidate:
        raise MissingUserAgent(
            "SEC_USER_AGENT is not set. SEC fair access requires a descriptive "
            "User-Agent with a contact email, e.g. "
            "'ma-acquirer-engine (you@domain.com)'. See .env.example."
        )
    if not re.search(r"[^@\s]+@[^@\s]+\.[^@\s]+", candidate):
        raise MissingUserAgent(f"SEC_USER_AGENT={candidate!r} contains no contact email address.")
    lowered = candidate.lower()
    for placeholder in _PLACEHOLDER_EMAILS:
        if placeholder in lowered:
            raise MissingUserAgent(
                f"SEC_USER_AGENT={candidate!r} still holds the .env.example "
                "placeholder. Put a real contact address in .env."
            )
    return candidate


def _ssl_verify() -> str | bool:
    """Resolve the TLS trust store.

    Machines behind a TLS-intercepting corporate proxy present a certificate
    signed by an internal CA that ``certifi`` does not know. ``EDGAR_CA_BUNDLE``
    (or the conventional ``SSL_CERT_FILE``) points at a bundle that does.
    """
    bundle = os.environ.get("EDGAR_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if bundle and Path(bundle).exists():
        return bundle
    return True


def is_offline(env: dict[str, str] | None = None) -> bool:
    """Return whether cache-only mode is active."""
    source = os.environ if env is None else env
    return source.get("EDGAR_OFFLINE", "").strip() not in ("", "0", "false", "False")


# --------------------------------------------------------------------------- #
# URL builders
# --------------------------------------------------------------------------- #


def normalize_cik(cik: str | int) -> str:
    """Return a CIK as the 10-digit zero-padded string EDGAR APIs expect.

    Args:
        cik: A CIK as an int, a bare string, or already prefixed (``CIK0000320193``).

    Returns:
        Ten digits, e.g. ``"0000320193"``.

    Raises:
        ValueError: If no digits are present or the value is too long.
    """
    digits = re.sub(r"\D", "", str(cik))
    if not digits:
        raise ValueError(f"no digits in CIK {cik!r}")
    if len(digits.lstrip("0")) > 10:
        raise ValueError(f"CIK {cik!r} is longer than 10 digits")
    return digits.zfill(10)


def company_tickers_url() -> str:
    """URL of the ticker-to-CIK map for the whole of EDGAR."""
    return f"{EDGAR_HOST}/files/company_tickers.json"


def submissions_url(cik: str | int) -> str:
    """URL of a company's filing index (most recent ~1000 filings plus page refs)."""
    return f"{DATA_HOST}/submissions/CIK{normalize_cik(cik)}.json"


def submissions_page_url(name: str) -> str:
    """URL of an older submissions page named in ``filings.files[].name``."""
    return f"{DATA_HOST}/submissions/{name}"


def companyfacts_url(cik: str | int) -> str:
    """URL of a company's full XBRL fact set."""
    return f"{DATA_HOST}/api/xbrl/companyfacts/CIK{normalize_cik(cik)}.json"



def full_text_search_url(
    query: str,
    *,
    forms: str | None = "8-K",
    start: date | None = None,
    end: date | None = None,
    offset: int = 0,
) -> str:
    """Build an EDGAR full-text search URL.

    Search is *document*-level: a hit names the exhibit that matched and carries
    the filer's CIK, SIC and state plus the containing filing's 8-K items. One
    request therefore replaces a submissions lookup, a directory listing and a
    document-type lookup.

    Args:
        query: Search expression. Wrap a phrase in double quotes to require it
            verbatim; an unquoted phrase matches any of its terms.
        forms: Comma-separated form filter, or ``None`` for all forms.
        start: Inclusive earliest filing date.
        end: Inclusive latest filing date. **For a backtest this must be the
            ``as_of_date``**, never today.
        offset: Zero-based offset into the result set, a multiple of
            :data:`FTS_PAGE_SIZE`.

    Returns:
        An absolute URL.

    Raises:
        ValueError: If ``offset`` is past the reachable window.
    """
    if offset >= FTS_MAX_REACHABLE:
        raise ValueError(
            f"offset={offset} is past the {FTS_MAX_REACHABLE}-document limit "
            "full-text search will page to. Narrow the date range instead."
        )
    params = {"q": query}
    if forms:
        params["forms"] = forms
    if start is not None:
        params["startdt"] = start.isoformat()
    if end is not None:
        params["enddt"] = end.isoformat()
    if offset:
        params["from"] = str(offset)
    return f"{FTS_HOST}/LATEST/search-index?{urlencode(params)}"


def filing_dir_url(cik: str | int, accession: str) -> str:
    """URL of the archive directory holding one filing's documents."""
    bare = re.sub(r"\D", "", accession)
    return f"{EDGAR_HOST}/Archives/edgar/data/{int(normalize_cik(cik))}/{bare}"


def filing_index_url(cik: str | int, accession: str) -> str:
    """URL of the JSON directory listing for one filing."""
    return f"{filing_dir_url(cik, accession)}/index.json"


def filing_document_url(cik: str | int, accession: str, document: str) -> str:
    """URL of a single document inside a filing."""
    return f"{filing_dir_url(cik, accession)}/{document}"


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class EdgarClient:
    """Cached, rate-limited, politely identified EDGAR reader."""

    def __init__(
        self,
        user_agent: str | None = None,
        *,
        cache: DiskCache | None = None,
        limiter: RateLimiter | None = None,
        offline: bool | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialise the client.

        Args:
            user_agent: Overrides ``SEC_USER_AGENT``. Validated either way.
            cache: Response cache. Defaults to ``data/edgar_cache`` (git-ignored).
            limiter: Rate limiter. Defaults to 10 requests/second.
            offline: Cache-only mode. Defaults to the ``EDGAR_OFFLINE`` env var.
            transport: httpx transport, for injecting recorded fixtures in tests.
            timeout: Per-request timeout in seconds.
        """
        self.cache = cache if cache is not None else DiskCache()
        self.limiter = limiter if limiter is not None else RateLimiter()
        self.offline = is_offline() if offline is None else offline
        self._timeout = timeout
        self._transport = transport
        self._client: httpx.Client | None = None
        # In offline mode nothing is fetched, so a missing User-Agent must not
        # stop a test suite from reading fixtures.
        self.user_agent = (
            (user_agent or os.environ.get("SEC_USER_AGENT", "")).strip()
            if self.offline
            else validate_user_agent(user_agent or os.environ.get("SEC_USER_AGENT"))
        )
        self.requests_made = 0
        self.cache_hits = 0
        self.seconds_rate_limited = 0.0
        # The agent fetches from several threads at once, and these counters go
        # into the run ledger. An audit trail that is approximately right because
        # two threads lost an increment is not much of an audit trail.
        self._counters = threading.Lock()

    # -- lifecycle -------------------------------------------------------- #

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={
                    "User-Agent": self.user_agent,
                    "Accept-Encoding": "gzip, deflate",
                },
                timeout=self._timeout,
                transport=self._transport,
                verify=_ssl_verify(),
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> EdgarClient:
        """Enter a context manager that closes the connection pool on exit."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the connection pool."""
        self.close()

    # -- fetching --------------------------------------------------------- #

    def get(self, url: str, *, max_age: float | None = None) -> CacheEntry:
        """Fetch a URL, serving from cache when possible.

        Args:
            url: Absolute URL on an SEC host.
            max_age: Maximum acceptable cache age in seconds. ``None`` (the
                default) accepts any cached copy, which is correct for immutable
                ``/Archives/`` documents. Pass :data:`MUTABLE_MAX_AGE` for
                submissions and companyfacts JSON.

        Returns:
            The cached or freshly fetched entry.

        Raises:
            OfflineCacheMiss: In offline mode with nothing cached for the URL.
            httpx.HTTPStatusError: On a non-retryable error response.
        """
        cached = self.cache.get(url, max_age=max_age)
        if cached is not None:
            with self._counters:
                self.cache_hits += 1
            return cached
        if self.offline:
            raise OfflineCacheMiss(
                f"{url} is not cached and EDGAR_OFFLINE is set. Unset EDGAR_OFFLINE "
                "to fetch it live through the rate limiter."
            )
        response = self._fetch(url)
        return self.cache.put(url, response.status_code, response.content, dict(response.headers))

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, ThrottledOrUnavailable)),
        wait=wait_exponential(multiplier=1, min=1, max=RETRY_MAX_WAIT),
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        reraise=True,
    )
    def _fetch(self, url: str) -> httpx.Response:
        """Perform one rate-limited request, retrying throttles and 5xx."""
        waited = self.limiter.acquire()
        with self._counters:
            self.seconds_rate_limited += waited
            self.requests_made += 1
        response = self._http().get(url)
        if response.status_code == 429 or response.status_code >= 500:
            # Retryable: raise so tenacity backs off.
            raise ThrottledOrUnavailable(
                f"EDGAR returned {response.status_code} for {url}",
                request=response.request,
                response=response,
            )
        if response.status_code == 403:
            raise MissingUserAgent(
                f"EDGAR returned 403 for {url}. This usually means the "
                f"User-Agent {self.user_agent!r} was rejected, or the request "
                "rate was too high."
            )
        response.raise_for_status()
        return response

    # -- typed helpers ---------------------------------------------------- #

    def get_json(self, url: str, *, max_age: float | None = None) -> Any:
        """Fetch a URL and parse the body as JSON."""
        return self.get(url, max_age=max_age).json()

    def get_text(self, url: str, *, max_age: float | None = None) -> str:
        """Fetch a URL and decode the body as text."""
        return self.get(url, max_age=max_age).text()

    def submissions(self, cik: str | int) -> Any:
        """Fetch a company's submissions index."""
        return self.get_json(submissions_url(cik), max_age=MUTABLE_MAX_AGE)

    def companyfacts(self, cik: str | int) -> Any:
        """Fetch a company's XBRL facts."""
        return self.get_json(companyfacts_url(cik), max_age=MUTABLE_MAX_AGE)

    def company_tickers(self) -> Any:
        """Fetch the EDGAR ticker-to-CIK map."""
        return self.get_json(company_tickers_url(), max_age=MUTABLE_MAX_AGE)


    def full_text_search(
        self,
        query: str,
        *,
        forms: str | None = "8-K",
        start: date | None = None,
        end: date | None = None,
        offset: int = 0,
    ) -> Any:
        """Run one page of an EDGAR full-text search.

        Search results are cached like any other response. They are treated as
        immutable despite being a query: a search bounded by an explicit end date
        is reproducible, and re-running it would change results only by adding
        filings dated after that bound, which a backtest must not see.
        """
        return self.get_json(
            full_text_search_url(query, forms=forms, start=start, end=end, offset=offset)
        )

    def stats(self) -> dict[str, float]:
        """Return counters for the run ledger."""
        with self._counters:
            return {
                "requests_made": self.requests_made,
                "cache_hits": self.cache_hits,
                "seconds_rate_limited": round(self.seconds_rate_limited, 3),
            }
