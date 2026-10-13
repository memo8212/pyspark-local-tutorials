"""Pytest configuration: a session-scoped local SparkSession for all tests."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def spark():
    from common.session import get_spark, stop_spark
    s = get_spark("pytest-local", shuffle_partitions=4)
    yield s
    stop_spark(s)
