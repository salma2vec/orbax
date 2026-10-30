"""Validation-only pytest hook to preserve errors before multihost teardown."""
import os
import sys


def pytest_runtest_logstart(nodeid, location):
    print(f"[Rank {os.environ.get('JAX_PROCESS_ID')}] START {nodeid}", file=sys.__stderr__, flush=True)


def pytest_runtest_logreport(report):
    if report.failed:
        print(f"[Rank {os.environ.get('JAX_PROCESS_ID')}] FAILED {report.nodeid} phase={report.when}\n{report.longreprtext}", file=sys.__stderr__, flush=True)
