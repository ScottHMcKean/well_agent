import pytest
import tempfile
import yaml
from pathlib import Path
from hydrate.download import DatasetDownloader
from hydrate.utils import DotConfig, load_config


# test_config_file fixture is now provided by conftest.py


def test_load_config():
    """Test that we can load the main config file."""
    config = load_config("fixtures/config_test.yaml")

    assert "download" in config
    assert isinstance(config["download"], dict)

    download_config = config["download"]
    expected_keys = ["output_dir", "skip_existing", "base_url"]

    for key in expected_keys:
        assert key in download_config, f"Expected key '{key}' in download config"


def test_dotconfig_with_downloader():
    """Test DotConfig with DatasetDownloader."""
    config = DotConfig("fixtures/config_test.yaml")
    downloader = DatasetDownloader(config)

    # Verify downloader was initialized correctly
    assert hasattr(downloader, "download_config")
    assert hasattr(downloader, "base_url")
    assert "github.com" in downloader.base_url


def test_dotconfig_custom_path(test_config_file):
    """Test DotConfig with custom config path."""
    config = DotConfig(test_config_file)
    downloader = DatasetDownloader(config)

    # Verify config was loaded correctly
    assert downloader.download_config.max_files == 2
    assert downloader.download_config.max_dirs == 1
    assert downloader.download_config.delay_seconds == 0.05


def test_downloader_requires_download_section():
    """Test that DatasetDownloader validates config has download section."""
    # Create config without download section
    import tempfile

    config_content = {"other_section": {"value": "test"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_content, f)
        temp_path = f.name

    try:
        config = DotConfig(temp_path)
        with pytest.raises(ValueError, match="Config must have 'download' section"):
            DatasetDownloader(config)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_config_validation():
    """Test that the config file has expected structure."""
    config = load_config("fixtures/config_test.yaml")
    download_config = config.get("download", {})

    # Check that key configuration options are present
    assert "output_dir" in download_config
    assert "base_url" in download_config

    # Check types
    assert isinstance(download_config["output_dir"], str)
    assert isinstance(download_config["base_url"], str)

    # Check max_files can be either int or None
    max_files = download_config.get("max_files")
    assert max_files is None or isinstance(max_files, int)


def test_max_files_configuration():
    """Test that max_files configuration works correctly."""
    config = load_config("fixtures/config_test.yaml")
    download_config = config.get("download", {})

    max_files = download_config.get("max_files")

    # Should be either None (no limit) or positive integer
    if max_files is not None:
        assert isinstance(max_files, int)
        assert max_files > 0
