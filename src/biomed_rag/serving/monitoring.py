"""Lightweight monitoring: one JSON log line per request + rolling stats for GET /stats.

Questions are never logged as text (privacy): only their length and a short hash, which is
enough to spot repeated questions without storing what people asked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass

logger = logging.getLogger("biomed_rag.requests")


@dataclass
class RequestRecord:
    mode: str
    seconds: float
    question_chars: int
    question_hash: str
    tokens: int = 0
    abstained: bool = False
    used_live: bool = False
    error: str | None = None


def question_hash(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:12]


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q between 0 and 100)."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(q / 100 * len(ordered)) - 1))
    return ordered[index]


class Monitor:
    """Keeps the last `window` requests in memory (resets when the app restarts)."""

    def __init__(self, window: int = 500, clock=time.time):
        self.records: deque[RequestRecord] = deque(maxlen=window)
        self.total = 0
        self.clock = clock
        self.started = clock()

    def record(self, record: RequestRecord) -> None:
        self.records.append(record)
        self.total += 1
        logger.info(json.dumps({"event": "ask", **asdict(record)}))

    def stats(self) -> dict:
        records = list(self.records)
        ok = [r for r in records if r.error is None]
        latencies = [r.seconds for r in ok]
        return {
            "uptime_seconds": round(self.clock() - self.started),
            "requests_total": self.total,
            "window": len(records),
            "errors": len(records) - len(ok),
            "error_rate": round((len(records) - len(ok)) / len(records), 3) if records else 0.0,
            "abstain_rate": round(sum(r.abstained for r in ok) / len(ok), 3) if ok else 0.0,
            "live_pubmed_rate": round(sum(r.used_live for r in ok) / len(ok), 3) if ok else 0.0,
            "latency_p50_s": round(percentile(latencies, 50), 2) if ok else None,
            "latency_p95_s": round(percentile(latencies, 95), 2) if ok else None,
            "avg_tokens": round(statistics.mean(r.tokens for r in ok)) if ok else None,
            "by_mode": {m: sum(r.mode == m for r in records) for m in ("rag", "agent")},
            "recent_errors": [r.error for r in records if r.error][-5:],
        }


def setup_logging() -> None:
    """Make our INFO logs visible next to uvicorn's (in the terminal and in Space logs)."""
    for name in ("biomed_rag.requests", "biomed_rag.api"):
        log = logging.getLogger(name)
        if not log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s %(message)s"))
            log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
