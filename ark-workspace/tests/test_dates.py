from src.dates import days_between


def test_single_day():
    assert days_between("2026-01-01", "2026-01-01") == ["2026-01-01"]


def test_multiple_days_inclusive():
    assert days_between("2026-01-01", "2026-01-03") == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
    ]


def test_range_crossing_a_month_boundary():
    assert days_between("2026-01-30", "2026-02-02") == [
        "2026-01-30",
        "2026-01-31",
        "2026-02-01",
        "2026-02-02",
    ]


def test_end_before_start():
    assert days_between("2026-01-03", "2026-01-01") == []
