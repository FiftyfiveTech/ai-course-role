"""
Pytest configuration: a silent test skip is a FAIL.

Any test that skips without being explicitly marked @pytest.mark.skip
or xfail causes the suite to exit non-zero. Missing env-var guards must
raise errors, not skip silently.
"""

import pytest


def pytest_runtest_makereport(item, call):
    """Convert unexpected skips into failures."""
    pass


def pytest_collection_modifyitems(config, items):
    """Nothing collected = nothing wrong at collection time."""
    pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    outcome = yield
    return outcome


def pytest_runtest_logreport(report):
    """
    If a test was skipped but NOT marked with @pytest.mark.skip or
    @pytest.mark.skipif, treat it as a failure.
    """
    if report.skipped:
        markers = {m.name for m in report.item.own_markers}
        # Allow explicit skip markers and xfail
        allowed = {"skip", "skipif", "xfail"}
        if not markers & allowed:
            report.outcome = "failed"
            report.longrepr = (
                f"[ROLE-001] Unexpected skip in {report.nodeid} — "
                "a silent skip is a FAIL. Add an explicit @pytest.mark.skip "
                "or fix the missing dependency."
            )
