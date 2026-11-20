from starlette.requests import Request

from biomed_rag.serving.rate_limit import DAY, RateLimiter, client_id


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def limiter(clock, per_minute=2, per_day=5, global_per_day=100) -> RateLimiter:
    return RateLimiter(per_minute, per_day, global_per_day, clock=clock)


def test_allows_up_to_per_minute_then_blocks_until_window_passes():
    clock = Clock()
    rl = limiter(clock)
    assert rl.check("a") is None
    assert rl.check("a") is None
    blocked = rl.check("a")
    assert blocked.reason == "too many questions per minute"
    assert blocked.retry_after == 60
    clock.now += 61
    assert rl.check("a") is None


def test_visitors_are_limited_separately():
    rl = limiter(Clock(), per_minute=1)
    assert rl.check("a") is None
    assert rl.check("b") is None
    assert rl.check("a") is not None


def test_daily_limit_per_visitor_resets_after_a_day():
    clock = Clock()
    rl = limiter(clock, per_minute=100, per_day=3)
    for _ in range(3):
        assert rl.check("a") is None
    assert rl.check("a").reason == "you reached your daily question limit"
    clock.now += DAY
    assert rl.check("a") is None


def test_global_daily_cap_blocks_everyone():
    rl = limiter(Clock(), per_minute=100, global_per_day=2)
    assert rl.check("a") is None
    assert rl.check("b") is None
    assert rl.check("c").reason == "the demo reached its daily question limit"


def test_blocked_requests_do_not_use_up_quota():
    clock = Clock()
    rl = limiter(clock, per_minute=1, per_day=2)
    rl.check("a")
    for _ in range(5):
        rl.check("a")  # all blocked by the minute limit
    clock.now += 61
    assert rl.check("a") is None  # still has its 2nd daily question


def make_request(headers: dict, host="1.2.3.4") -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw, "client": (host, 5000)})


def test_client_id_prefers_forwarded_ip():
    assert client_id(make_request({"X-Forwarded-For": "9.9.9.9, 10.0.0.1"})) == "9.9.9.9"
    assert client_id(make_request({})) == "1.2.3.4"
