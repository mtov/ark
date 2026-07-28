from datetime import date, timedelta

def days_between(start: str, end: str):
    current = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    if stop < current:
        return []
    result = []
    while current <= stop:
        result.append(current.isoformat())
        current += timedelta(days=1)
    return result
