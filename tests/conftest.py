"""Pytest configuration for lattifai-captions tests."""

import pytest


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path
