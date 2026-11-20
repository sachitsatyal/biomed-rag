"""In-memory rate limiting for /ask: protects the free LLM quota on a public demo.

Three sliding windows: per visitor per minute, per visitor per day, and a global daily cap.
Single-process only (fine for one Space container); multi-server setups would use Redis.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import Request

MINUTE = 60.0
DAY = 24 * 60 * 60.0


@dataclass
class Blocked:
    reason: str
    retry_after: float  # seconds until a slot frees up


class RateLimiter:
    def __init__(self, per_minute: int, per_day: int, global_per_day: int, clock=time.time):
        self.per_minute = per_minute
        self.per_day = per_day
        self.global_per_day = global_per_day
        self.clock = clock
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.global_hits: deque[float] = deque()

    @classmethod
    def from_env(cls) -> RateLimiter:
        return cls(
            per_minute=int(os.environ.get("RATE_LIMIT_PER_MINUTE", "5")),
            per_day=int(os.environ.get("RATE_LIMIT_PER_DAY", "50")),
            global_per_day=int(os.environ.get("RATE_LIMIT_GLOBAL_PER_DAY", "300")),
        )

    def check(self, client: str) -> Blocked | None:
        """Record a request if allowed; otherwise return why it was blocked."""
        now = self.clock()
        hits = self.hits[client]
        for window in (hits, self.global_hits):
            while window and now - window[0] >= DAY:  # forget requests older than a day
                window.popleft()

        if len(self.global_hits) >= self.global_per_day:
            wait = DAY - (now - self.global_hits[0])
            return Blocked("the demo reached its daily question limit", wait)
        if len(hits) >= self.per_day:
            return Blocked("you reached your daily question limit", DAY - (now - hits[0]))
        if len(hits) >= self.per_minute and now - hits[-self.per_minute] < MINUTE:
            wait = MINUTE - (now - hits[-self.per_minute])
            return Blocked("too many questions per minute", wait)

        hits.append(now)
        self.global_hits.append(now)
        return None


def client_id(request: Request) -> str:
    """Visitor's IP. Behind a proxy (like Hugging Face) the real IP is in X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
