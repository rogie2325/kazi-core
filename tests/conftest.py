import logging

import pytest


@pytest.fixture(autouse=True)
def _restore_kazi_logger():
    """
    Reset the kazi logger after every test.

    configure_logging() sets kazi.propagate=False so it owns its own output.
    That bleeds into any subsequent test that uses caplog (which requires
    propagation to reach pytest's root handler). Restoring here means tests
    are order-independent regardless of which configure_logging test ran first.
    """
    yield
    lg = logging.getLogger("kazi")
    lg.propagate = True
