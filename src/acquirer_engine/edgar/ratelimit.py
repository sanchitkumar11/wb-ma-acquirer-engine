"""Request rate limiting for SEC fair access.

The SEC asks automated clients to stay at or below 10 requests per second. This
module enforces that with a sliding window over the timestamps of recent
requests, which is stricter than a token bucket: a bucket that has been idle
accumulates credit and then permits a burst above the stated ceiling.

The clock and sleep function are injectable so the guarantee can be tested
deterministically instead of by timing a real run.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

SEC_MAX_REQUESTS_PER_SECOND = 10


class RateLimiter:
    """Blocks callers so that no 1-second window contains more than N requests.

    Thread-safe: the Phase 3 agent fans out over candidate buyers with bounded
    concurrency, and all of those workers share one limiter.
    """

    def __init__(
        self,
        max_per_second: int = SEC_MAX_REQUESTS_PER_SECOND,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialise the limiter.

        Args:
            max_per_second: Ceiling on requests in any 1-second window.
            monotonic: Clock source, injectable for tests.
            sleep: Sleep function, injectable for tests.

        Raises:
            ValueError: If ``max_per_second`` is not positive, or exceeds the
                SEC's stated ceiling.
        """
        if max_per_second < 1:
            raise ValueError("max_per_second must be at least 1")
        if max_per_second > SEC_MAX_REQUESTS_PER_SECOND:
            raise ValueError(
                f"max_per_second={max_per_second} exceeds the SEC fair-access "
                f"ceiling of {SEC_MAX_REQUESTS_PER_SECOND} requests/second"
            )
        self.max_per_second = max_per_second
        self._monotonic = monotonic
        self._sleep = sleep
        self._recent: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Reserve one request slot, blocking if necessary.

        Returns:
            The number of seconds spent waiting, which the client records so the
            run ledger shows how much wall-clock time went to rate limiting.
        """
        waited = 0.0
        with self._lock:
            while True:
                now = self._monotonic()
                self._evict(now)
                if len(self._recent) < self.max_per_second:
                    self._recent.append(now)
                    return waited
                # The window is full. Wait until the oldest request ages out.
                delay = 1.0 - (now - self._recent[0])
                if delay <= 0:
                    continue
                self._sleep(delay)
                waited += delay

    def _evict(self, now: float) -> None:
        while self._recent and now - self._recent[0] >= 1.0:
            self._recent.popleft()

    def observed_rate(self) -> int:
        """Return how many requests fall inside the current 1-second window."""
        with self._lock:
            self._evict(self._monotonic())
            return len(self._recent)
