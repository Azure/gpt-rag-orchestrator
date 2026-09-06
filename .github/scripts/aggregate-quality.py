"""Validate actual job results and fresh reports in the unprivileged aggregate job."""

import argparse
import json
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--needs", required=True)
    args = parser.parse_args()
    checker = runpy.run_path(str(Path(__file__).with_name("check-quality.py")))
    needs = json.loads(args.needs)
    if set(needs) != {"quality", "tests", "frontend"} or any(
        job["result"] != "success" for job in needs.values()
    ):
        raise SystemExit("Required test/frontend/quality job was not successful")
    reports = sorted(args.reports.rglob("quality-*.json"))
    if len(reports) != len(checker["CHECKS"]):
        raise SystemExit("Missing or duplicate quality report")
    seen = set()
    policy = set()
    for path in reports:
        check = path.stem.removeprefix("quality-")
        if check not in checker["CHECKS"] or check in seen:
            raise SystemExit("Unexpected or duplicated check")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not checker["report_valid"](
            report, args.base_sha, args.head_sha, {check}
        ):
            raise SystemExit(f"Failed, stale or invalid {check} report")
        policy.add(report["policy_sha"])
        seen.add(check)
    if len(policy) != 1:
        raise SystemExit("Quality jobs used different policies")
    print("quality-gate: passed")


if __name__ == "__main__":
    main()
