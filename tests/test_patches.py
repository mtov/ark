from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ark import patches
from ark.patches import auto_repair_patch_text, validate_patch, validate_and_repair_patch


def write_sample_file(workspace_path: Path) -> None:
    (workspace_path / "hello.txt").write_text("hello\nworld\n", encoding="utf-8")


def test_validate_patch_accepts_clean_patch(tmp_path: Path) -> None:
    write_sample_file(tmp_path)
    patch_text = """--- a/hello.txt
+++ b/hello.txt
@@ -1,2 +1,2 @@
-hello
+hi
 world
"""

    validate_patch(tmp_path, patch_text)


def test_validate_patch_rejects_non_applicable_patch(tmp_path: Path) -> None:
    write_sample_file(tmp_path)
    patch_text = """--- a/hello.txt
+++ b/hello.txt
@@ -1,2 +1,2 @@
-missing
+hi
 world
"""

    with pytest.raises(ValueError, match="Patch does not apply cleanly"):
        validate_patch(tmp_path, patch_text)


def test_auto_repair_patch_text_strips_code_fences_and_prose() -> None:
    patch_text = """Here is the patch:

```diff
--- a/hello.txt
+++ b/hello.txt
@@ -1,2 +1,2 @@
-hello
+hi
 world
```
"""

    repaired = auto_repair_patch_text(patch_text)

    assert repaired == """--- a/hello.txt
+++ b/hello.txt
@@ -1,2 +1,2 @@
-hello
+hi
 world
"""


def test_auto_repair_patch_text_strips_trailing_patch_wrapper_markers() -> None:
    patch_text = """--- a/hello.txt
+++ b/hello.txt
@@ -1,2 +1,2 @@
-hello
+hi
 world
*** End Patch
"""

    repaired = auto_repair_patch_text(patch_text)

    assert repaired == """--- a/hello.txt
+++ b/hello.txt
@@ -1,2 +1,2 @@
-hello
+hi
 world
"""


def test_validate_and_repair_patch_fixes_missing_context_prefix(tmp_path: Path) -> None:
    write_sample_file(tmp_path)
    patch_text = """--- a/hello.txt
+++ b/hello.txt
@@ -1,2 +1,2 @@
-hello
+hi
world
"""

    repaired = validate_and_repair_patch(tmp_path, patch_text)

    assert repaired == """--- a/hello.txt
+++ b/hello.txt
@@ -1,2 +1,2 @@
-hello
+hi
 world
"""


def test_validate_and_repair_patch_recalculates_hunk_header_counts(tmp_path: Path) -> None:
    write_sample_file(tmp_path)
    patch_text = """--- a/hello.txt
+++ b/hello.txt
@@ -1,6 +1,14 @@
-hello
+hi
 world
"""

    repaired = validate_and_repair_patch(tmp_path, patch_text)

    assert repaired == """--- a/hello.txt
+++ b/hello.txt
@@ -1,2 +1,2 @@
-hello
+hi
 world
"""


def test_validate_and_repair_patch_rebuilds_large_misaligned_hunk(tmp_path: Path) -> None:
    file_path = tmp_path / "src"
    file_path.mkdir()
    order_rules_path = file_path / "order_rules.py"
    order_rules_path.write_text(
        """from __future__ import annotations


def subtotal(items: list[dict]) -> float:
    return round(sum(item["price"] * item["quantity"] for item in items), 2)


def eligible_subtotal(items: list[dict]) -> float:
    eligible_items = []
    for item in items:
        if item.get("is_cancelled", False):
            continue
        if item.get("is_promotional", False):
            continue
        eligible_items.append(item)

    return round(
        sum(item["price"] * item["quantity"] for item in eligible_items),
        2,
    )


def reward_points(items: list[dict]) -> int:
    eligible_items = []
    for item in items:
        if item.get("is_cancelled", False):
            continue
        if item.get("is_promotional", False):
            continue
        eligible_items.append(item)

    points = sum(int(item["price"] * item["quantity"]) for item in eligible_items)
    return points // 10


def qualifying_item_count(items: list[dict]) -> int:
    eligible_items = []
    for item in items:
        if item.get("is_cancelled", False):
            continue
        if item.get("is_promotional", False):
            continue
        eligible_items.append(item)

    return sum(item["quantity"] for item in eligible_items)


def order_snapshot(items: list[dict]) -> dict:
    return {
        "subtotal": subtotal(items),
        "eligibleSubtotal": eligible_subtotal(items),
        "rewardPoints": reward_points(items),
        "qualifyingItemCount": qualifying_item_count(items),
    }
""",
        encoding="utf-8",
    )
    patch_text = """--- a/src/order_rules.py
+++ b/src/order_rules.py
@@ -1,38 +1,35 @@
 from __future__ import annotations
 
 
+def _eligible_items(items: list[dict]) -> list[dict]:
+    eligible_items = []
+    for item in items:
+        if item.get("is_cancelled", False):
+            continue
+        if item.get("is_promotional", False):
+            continue
+        eligible_items.append(item)
+    return eligible_items
+
+
 def subtotal(items: list[dict]) -> float:
     return round(sum(item["price"] * item["quantity"] for item in items), 2)
 
 
 def eligible_subtotal(items: list[dict]) -> float:
-    eligible_items = []
-    for item in items:
-        if item.get("is_cancelled", False):
-            continue
-        if item.get("is_promotional", False):
-            continue
-        eligible_items.append(item)
-
     return round(
-        sum(item["price"] * item["quantity"] for item in eligible_items),
+        sum(item["price"] * item["quantity"] for item in _eligible_items(items)),
         2,
     )
 
 
 def reward_points(items: list[dict]) -> int:
-    eligible_items = []
-    for item in items:
-        if item.get("is_cancelled", False):
-            continue
-        if item.get("is_promotional", False):
-            continue
-        eligible_items.append(item)
-
-    points = sum(int(item["price"] * item["quantity"]) for item in eligible_items)
+    points = sum(int(item["price"] * item["quantity"]) for item in _eligible_items(items))
     return points // 10
 
 
 def qualifying_item_count(items: list[dict]) -> int:
-    eligible_items = []
-    for item in items:
-        if item.get("is_cancelled", False):
-            continue
-        if item.get("is_promotional", False):
-            continue
-        eligible_items.append(item)
-
-    return sum(item["quantity"] for item in eligible_items)
+    return sum(item["quantity"] for item in _eligible_items(items))
"""

    repaired = validate_and_repair_patch(tmp_path, patch_text)

    assert "def _eligible_items" in repaired
    assert "def order_snapshot" in repaired


def test_validate_and_repair_patch_accepts_trailing_end_patch_marker(tmp_path: Path) -> None:
    order_rules_path = tmp_path / "order_rules.py"
    order_rules_path.write_text(
        "hello\nworld\n",
        encoding="utf-8",
    )
    patch_text = """--- a/order_rules.py
+++ b/order_rules.py
@@ -1,2 +1,2 @@
-hello
+hi
 world
*** End Patch
"""

    repaired = validate_and_repair_patch(tmp_path, patch_text)

    assert repaired == """--- a/order_rules.py
+++ b/order_rules.py
@@ -1,2 +1,2 @@
-hello
+hi
 world
"""


def test_run_mutating_command_requires_approval(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "n")

    with pytest.raises(PermissionError, match="User rejected command"):
        patches._run_mutating_command(["git", "status"], tmp_path)


def test_run_mutating_command_executes_after_approval(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "y")

    def fake_run(command: list[str], cwd: Path, capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(patches.subprocess, "run", fake_run)

    result = patches._run_mutating_command(["git", "status"], tmp_path)

    assert result.returncode == 0


def test_run_mutating_command_prints_preview_when_provided(capsys, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "n")

    with pytest.raises(PermissionError):
        patches._run_mutating_command(
            ["git", "apply", "patch.txt"],
            tmp_path,
            preview="--- a/file.py\n+++ b/file.py",
        )

    captured = capsys.readouterr()
    assert "Command preview:" in captured.out
    assert "--- a/file.py" in captured.out


def test_validate_and_repair_patch_reports_original_block_when_rebuild_fails(tmp_path: Path) -> None:
    file_path = tmp_path / "src"
    file_path.mkdir()
    pricing_path = file_path / "pricing.py"
    pricing_path.write_text(
        'from decimal import Decimal\n\n\ndef calculate_discount(subtotal):\n    if subtotal >= Decimal("100.00"):\n        return subtotal * Decimal("0.10")\n    return Decimal("0.00")\n',
        encoding="utf-8",
    )
    patch_text = """--- a/src/pricing.py
+++ b/src/pricing.py
@@ -1,6 +1,10 @@
 from decimal import Decimal
 
 
+def calculate_subtotal(items):
+    return sum((item["unit_price"] * item["quantity"] for item in items), start=0)
+
+
 def calculate_discount(subtotal):
     if subtotal >= Decimal("100.00"):
         return subtotal * Decimal("0.10")
         return Decimal("0.00")
"""

    with pytest.raises(ValueError, match="Failed to match the original block") as exc_info:
        validate_and_repair_patch(tmp_path, patch_text)

    message = str(exc_info.value)
    assert "src/pricing.py" in message
    assert 'if subtotal >= Decimal("100.00"):' in message
    assert 'return Decimal("0.00")' in message
