from __future__ import annotations

import re

FAILED_TEST_LINE = re.compile(r"^FAILED\s+(.+?)\s+-\s+(.+)$")
ASSERT_EQUALS_LINE = re.compile(r"^E\s+AssertionError:\s+assert\s+(.+?)\s+==\s+(.+)$")
FAILED_TEST_HEADER = re.compile(r"^_+\s+([A-Za-z0-9_]+)\s+_+$")


def summarize_test_failure_output(test_output: str) -> str:
    summary_lines = [
        "Post-apply tests failed. The runtime workspace has been reset to the original source state.",
        "Use the failed test details below to produce a different patch.",
    ]
    failed_cases: list[str] = []
    current_test: str | None = None

    for raw_line in test_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        header_match = FAILED_TEST_HEADER.match(line)
        if header_match:
            current_test = header_match.group(1)
            continue

        failed_match = FAILED_TEST_LINE.match(line)
        if failed_match:
            current_test = failed_match.group(1)
            failed_cases.append(f"- {current_test}: {failed_match.group(2)}")
            continue

        if current_test is None:
            continue

        assertion_match = ASSERT_EQUALS_LINE.match(line)
        if assertion_match:
            got_value = assertion_match.group(1)
            expected_value = assertion_match.group(2)
            failed_cases.append(f"- {current_test}: expected {expected_value}, got {got_value}")
            current_test = None

    if failed_cases:
        summary_lines.extend(failed_cases)
    else:
        summary_lines.append(test_output.strip())

    return "\n".join(summary_lines)
