"""Content-addressed disk cache for EDGAR responses.

Every EDGAR response is cached so that reruns never hit SEC servers. This is a
fair-access obligation, not an optimisation: a rerun must cost the SEC nothing.

Two kinds of resource are cached here, and they differ in mutability. Documents
under ``/Archives/`` are immutable once filed and may be cached forever.
``submissions`` and ``companyfacts`` JSON change as companies file, so callers
pass ``max_age`` when freshness matters.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE_ROOT = Path(os.environ.get("EDGAR_CACHE_DIR", "data/edgar_cache"))


def cache_key(url: str) -> str:
    """Return the stable cache key for a URL.

    Args:
        url: Absolute URL, including any query string.

    Returns:
        A 64-character hex digest.
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A cached HTTP response.

    Attributes:
        url: The URL that produced this body.
        status: HTTP status code as originally received.
        body: Raw response bytes.
        headers: Response headers, lowercased keys.
        fetched_at: Unix timestamp of the original fetch.
    """

    url: str
    status: int
    body: bytes
    headers: dict[str, str]
    fetched_at: float

    @property
    def age_seconds(self) -> float:
        """Return how long ago this entry was fetched."""
        return max(0.0, time.time() - self.fetched_at)

    def json(self) -> object:
        """Parse the body as JSON."""
        return json.loads(self.body)

    def text(self, encoding: str = "utf-8") -> str:
        """Decode the body as text, replacing undecodable bytes.

        EDGAR serves a mix of encodings and some older filings contain invalid
        sequences, so decoding is deliberately lenient.
        """
        return self.body.decode(encoding, errors="replace")


class DiskCache:
    """Stores responses under ``root`` keyed by a hash of the URL."""

    def __init__(self, root: Path | str = DEFAULT_CACHE_ROOT) -> None:
        """Initialise the cache.

        Args:
            root: Directory to store entries in. Created on first write.
        """
        self.root = Path(root)

    def _paths(self, url: str) -> tuple[Path, Path]:
        key = cache_key(url)
        # Shard on the first two hex characters; a flat directory with tens of
        # thousands of entries is slow to list on macOS.
        directory = self.root / key[:2]
        return directory / f"{key}.body", directory / f"{key}.meta.json"

    def get(self, url: str, max_age: float | None = None) -> CacheEntry | None:
        """Return the cached entry for a URL, if present and fresh enough.

        Args:
            url: The URL to look up.
            max_age: Maximum acceptable age in seconds. ``None`` accepts any age,
                which is correct for immutable ``/Archives/`` documents.

        Returns:
            The entry, or ``None`` on a miss or when the entry is too old.
        """
        body_path, meta_path = self._paths(url)
        if not (body_path.exists() and meta_path.exists()):
            return None
        meta = json.loads(meta_path.read_text())
        entry = CacheEntry(
            url=meta["url"],
            status=meta["status"],
            body=body_path.read_bytes(),
            headers=meta.get("headers", {}),
            fetched_at=meta["fetched_at"],
        )
        if max_age is not None and entry.age_seconds > max_age:
            return None
        return entry

    def put(
        self,
        url: str,
        status: int,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> CacheEntry:
        """Write a response to the cache and return the stored entry.

        The write is atomic: the body goes to a temporary file that is renamed
        into place, so an interrupted run cannot leave a truncated body that a
        later run would trust.

        The temporary name carries the process and thread id because the agent
        fetches concurrently: two threads asked for the same URL at the same moment
        would otherwise write the same temporary file, and the body that got renamed
        into place would be an interleaving of both.
        """
        body_path, meta_path = self._paths(url)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        entry = CacheEntry(
            url=url,
            status=status,
            body=body,
            headers={k.lower(): v for k, v in (headers or {}).items()},
            fetched_at=time.time(),
        )
        tmp = body_path.with_suffix(f".body.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(body)
        os.replace(tmp, body_path)
        meta_path.write_text(
            json.dumps(
                {
                    "url": entry.url,
                    "status": entry.status,
                    "headers": entry.headers,
                    "fetched_at": entry.fetched_at,
                    "bytes": len(body),
                },
                indent=2,
            )
        )
        return entry

    def __contains__(self, url: object) -> bool:
        """Return whether a URL has any cached entry, regardless of age."""
        return isinstance(url, str) and self.get(url) is not None
