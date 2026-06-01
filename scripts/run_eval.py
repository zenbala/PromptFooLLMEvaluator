#!/usr/bin/env python3
"""
scripts/run_eval.py
───────────────────
Runs the Promptfoo evaluation and prints a human-readable summary.

Usage:
    python scripts/run_eval.py                  # full suite
    python scripts/run_eval.py --test happy     # single suite
    python scripts/run_eval.py --watch          # watch mode
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


RESULTS_FILE = Path("results/latest.json")
SUITE_MAP = {
    "happy": "tests/happy_path.yaml",
    "edge": "tests/edge_cases.yaml",
    "hallucination": "tests/hallucination.yaml",
    "safety": "tests/safety.yaml",
}


def run_promptfoo(test_file: Optional[str] = None, watch: bool = False) -> int:
    cmd = ["npx", "promptfoo", "eval", "--output", str(RESULTS_FILE)]
    if test_file:
        cmd += ["--tests", test_file]
    if watch:
        cmd.append("--watch")

    print(f"\n🚀 Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def parse_results() -> None:
    if not RESULTS_FILE.exists():
        print("⚠️  No results file found. Did the eval run successfully?")
        return

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    results = data.get("results", {})
    table = results.get("table", {})
    head = table.get("head", {})
    body = table.get("body", [])

    providers = [p.get("label", p.get("id", "?")) for p in head.get("providers", [])]
    print(f"\n{'═'*70}")
    print(f"  PROMPTFOO RAG VALIDATION — RESULTS SUMMARY")
    print(f"{'═'*70}")
    print(f"  Providers tested: {', '.join(providers)}")
    print(f"  Total test cases: {len(body)}")

    passed = failed = errors = 0
    for row in body:
        for cell in row.get("outputs", []):
            for assertion in cell.get("gradingResult", {}).get("componentResults", []):
                if assertion.get("pass"):
                    passed += 1
                else:
                    failed += 1
            if cell.get("error"):
                errors += 1

    total = passed + failed
    pct = round(passed / total * 100, 1) if total else 0

    print(f"\n  ✅ Passed assertions : {passed}")
    print(f"  ❌ Failed assertions : {failed}")
    print(f"  💥 Errors            : {errors}")
    print(f"  📊 Pass rate         : {pct}%")

    # Per-test breakdown
    print(f"\n{'─'*70}")
    print(f"  PER-TEST BREAKDOWN")
    print(f"{'─'*70}")
    for i, row in enumerate(body, 1):
        desc = row.get("description") or row.get("test", {}).get("description", f"Test {i}")
        row_passed = all(
            all(a.get("pass") for a in cell.get("gradingResult", {}).get("componentResults", []))
            for cell in row.get("outputs", [])
        )
        icon = "✅" if row_passed else "❌"
        print(f"  {icon} [{i:02d}] {desc}")

    print(f"\n  📁 Full report: {RESULTS_FILE}\n")
    print(f"  👉 View interactive report: npx promptfoo view\n")


def main():
    parser = argparse.ArgumentParser(description="Run Promptfoo RAG validation")
    parser.add_argument("--test", choices=list(SUITE_MAP.keys()), help="Run a specific test suite")
    parser.add_argument("--watch", action="store_true", help="Enable watch mode")
    parser.add_argument("--summary-only", action="store_true", help="Parse existing results only")
    args = parser.parse_args()

    if args.summary_only:
        parse_results()
        return

    test_file = SUITE_MAP.get(args.test)
    code = run_promptfoo(test_file, args.watch)

    if code == 0:
        parse_results()
    else:
        print(f"\n⚠️  Promptfoo exited with code {code}. Check output above for details.")
        sys.exit(code)


if __name__ == "__main__":
    main()
