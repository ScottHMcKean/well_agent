"""Shared test fixtures for all tests."""
import pytest
import tempfile
import yaml
from pathlib import Path


@pytest.fixture
def test_config():
    """Create a comprehensive test configuration for testing."""
    return {
        "catalog": "test_catalog",
        "schema": "test_schema",
        "download": {
            "output_dir": "test_download",
            "base_url": "https://api.github.com/repos/petrobras/3W/contents",
            "max_files": 2,
            "max_dirs": 1,
            "skip_existing": True,
            "delay_seconds": 0.05,
        },
        "process": {
            "table": "test_well_data"
        },
        "train": {
            "test_ratio": 0.2,
            "base_model_name": "test_hydrate_xgboost"
        },
        "deploy": {
            "base_hydrate_deploy_name": "test_hydrate_predict"
        },
        "agentify": {
            "agent_name": "test_3w_well_agent"
        }
    }


@pytest.fixture
def test_config_file(test_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def minimal_download_config():
    """Create a minimal config with just download section for testing."""
    return {
        "download": {
            "output_dir": "test_download",
            "base_url": "https://api.github.com/repos/petrobras/3W/contents",
            "skip_existing": True,
        }
    }


@pytest.fixture
def minimal_download_config_file(minimal_download_config):
    """Create a temporary minimal config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(minimal_download_config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    Path(config_path).unlink(missing_ok=True)