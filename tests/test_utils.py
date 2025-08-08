import pytest
from well_agent.utils import DotDict, DotConfig


@pytest.fixture
def sample_dict():
    """Create a sample nested dictionary for testing."""
    return {
        "level1": {"level2": {"value": "nested_value"}, "simple_value": 42},
        "top_level": "top_value",
        "list_value": [1, 2, 3],
    }


def test_dotdict_basic_access(sample_dict):
    """Test basic dot notation access to dictionary values."""
    dot_dict = DotDict(sample_dict)

    # Test simple access
    assert dot_dict.top_level == "top_value"
    assert dot_dict.list_value == [1, 2, 3]

    # Test nested access
    assert dot_dict.level1.simple_value == 42
    assert dot_dict.level1.level2.value == "nested_value"


def test_dotdict_nested_returns_dotdict(sample_dict):
    """Test that nested dictionaries are returned as DotDict instances."""
    dot_dict = DotDict(sample_dict)

    # Accessing nested dict should return DotDict instance
    nested = dot_dict.level1
    assert isinstance(nested, DotDict)


def test_dotconfig_loads_config():
    """Test that DotConfig loads config.yaml and provides dot notation access."""
    config = DotConfig("fixtures/config_test.yaml")

    # Should be able to access config values with dot notation
    assert hasattr(config, "download")
    assert config.download.output_dir == "/Volumes/shm/3w/bronze/"


def test_dotdict_attribute_error_for_missing_keys():
    """Test that accessing non-existent keys raises appropriate error."""
    dot_dict = DotDict({"existing_key": "value"})

    with pytest.raises(AttributeError):
        _ = dot_dict.non_existent_key


def test_dotconfig_with_test_config_fixture(test_config_file):
    """Example test using the shared test config fixture."""
    config = DotConfig(test_config_file)

    # Verify the test config structure
    assert config.catalog == "test_catalog"
    assert config.schema == "test_schema"
    assert config.download.output_dir == "test_download"
    assert config.download.max_files == 2
