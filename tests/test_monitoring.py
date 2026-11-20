from biomed_rag.serving.monitoring import Monitor, RequestRecord, percentile, question_hash


def rec(seconds=1.0, **kwargs) -> RequestRecord:
    return RequestRecord(
        mode="agent", seconds=seconds, question_chars=10, question_hash="h", **kwargs
    )


def test_question_hash_ignores_case_and_spaces_and_hides_text():
    assert question_hash("Does APOE4 matter?") == question_hash("  does apoe4 matter? ")
    assert "apoe4" not in question_hash("Does APOE4 matter?")


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile([3.0], 95) == 3.0


def test_empty_stats():
    stats = Monitor().stats()
    assert stats["requests_total"] == 0
    assert stats["latency_p50_s"] is None


def test_stats_summarise_requests():
    monitor = Monitor(clock=lambda: 100.0)
    monitor.record(rec(2.0, tokens=100, used_live=True))
    monitor.record(rec(4.0, tokens=300, abstained=True))
    monitor.record(rec(0.5, error="RateLimitError"))
    stats = monitor.stats()
    assert stats["requests_total"] == 3
    assert stats["errors"] == 1
    assert stats["error_rate"] == 0.333
    assert stats["abstain_rate"] == 0.5  # errors are excluded from answer-quality rates
    assert stats["live_pubmed_rate"] == 0.5
    assert stats["avg_tokens"] == 200
    assert stats["recent_errors"] == ["RateLimitError"]


def test_window_keeps_only_recent_requests():
    monitor = Monitor(window=2)
    for _ in range(5):
        monitor.record(rec())
    assert monitor.stats()["requests_total"] == 5
    assert monitor.stats()["window"] == 2
