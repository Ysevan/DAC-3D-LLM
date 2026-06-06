from __future__ import annotations
import statistics
import time
import requests

BASE_URL = "http://127.0.0.1:7890"
HEADERS = {
    "x-dac3d-session-id": "perf-session",
    "x-dac3d-operator-id": "tester",
    "x-dac3d-roles": "operator",
}

def stats(values):
    values = sorted(values)
    p95 = values[int(round(0.95 * (len(values) - 1)))]
    return {
        "min": min(values),
        "max": max(values),
        "avg": statistics.mean(values),
        "p95": p95,
        "stddev": statistics.pstdev(values),
    }

def measure_get(path, times=20):
    values, failures = [], 0
    for _ in range(times):
        start = time.perf_counter()
        try:
            r = requests.get(BASE_URL + path, headers=HEADERS, timeout=15)
            r.raise_for_status()
            values.append((time.perf_counter() - start) * 1000)
        except Exception:
            failures += 1
    return values, failures

if __name__ == "__main__":
    for name, path in [("health", "/api/health"), ("runtime", "/api/runtime")]:
        values, failures = measure_get(path)
        print(name, stats(values) if values else {}, "failures=", failures)
