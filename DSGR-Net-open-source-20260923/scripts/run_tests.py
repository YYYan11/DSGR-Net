#!/usr/bin/env python3
"""Run the release smoke tests without requiring an external test runner."""

import importlib
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

TESTS = (
    ("test_demo_forward", "test_demo_forward_is_finite_and_has_expected_shapes"),
    ("test_water_balance", "test_hard_recurrence_closes_daily_water_balance"),
    ("test_irrigation_gradient", "test_irrigation_gradient_matches_central_difference"),
)


def main():
    failures = 0
    for module_name, function_name in TESTS:
        label = f"{module_name}.{function_name}"
        try:
            module = importlib.import_module(module_name)
            getattr(module, function_name)()
        except Exception:
            failures += 1
            print(f"FAIL {label}")
            traceback.print_exc()
        else:
            print(f"PASS {label}")
    if failures:
        print(f"TESTS FAILED: {failures}/{len(TESTS)}")
        return 1
    print(f"TESTS PASSED: {len(TESTS)}/{len(TESTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
