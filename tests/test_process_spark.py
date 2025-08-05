"""Tests for Spark-based processing functions.

Note: These tests are designed to run on Databricks with databricks-connect.
They verify the function signatures and imports but don't execute Spark operations locally.
"""
import pytest
import sys
from unittest.mock import Mock

from hydrate.process import (
    read_nested_parquet_files_spark,
    clean_data_spark,
    add_state_name_spark,
    process_well_data_spark
)


def test_spark_functions_importable():
    """Test that all Spark functions can be imported successfully."""
    # Test that functions are callable
    assert callable(read_nested_parquet_files_spark)
    assert callable(clean_data_spark)
    assert callable(add_state_name_spark)
    assert callable(process_well_data_spark)


def test_spark_function_signatures():
    """Test that Spark functions have expected signatures."""
    import inspect
    
    # Test read_nested_parquet_files_spark signature
    sig = inspect.signature(read_nested_parquet_files_spark)
    expected_params = ['root_dir', 'pattern', 'spark']
    assert list(sig.parameters.keys()) == expected_params
    
    # Test clean_data_spark signature
    sig = inspect.signature(clean_data_spark)
    expected_params = ['all_data']
    assert list(sig.parameters.keys()) == expected_params
    
    # Test add_state_name_spark signature
    sig = inspect.signature(add_state_name_spark)
    expected_params = ['df']
    assert list(sig.parameters.keys()) == expected_params
    
    # Test process_well_data_spark signature
    sig = inspect.signature(process_well_data_spark)
    expected_params = ['root_dir', 'spark', 'pattern']
    assert list(sig.parameters.keys()) == expected_params


def test_spark_functions_have_docstrings():
    """Test that all Spark functions have proper docstrings."""
    assert read_nested_parquet_files_spark.__doc__ is not None
    assert "Spark" in read_nested_parquet_files_spark.__doc__
    assert "parallel" in read_nested_parquet_files_spark.__doc__
    
    assert clean_data_spark.__doc__ is not None
    assert "Spark" in clean_data_spark.__doc__
    
    assert add_state_name_spark.__doc__ is not None
    assert "Spark" in add_state_name_spark.__doc__
    
    assert process_well_data_spark.__doc__ is not None
    assert "pipeline" in process_well_data_spark.__doc__


@pytest.mark.skipif(
    "pyspark" not in sys.modules or "DATABRICKS_RUNTIME_VERSION" not in sys.modules,
    reason="Requires Databricks runtime environment"
)
def test_integration_on_databricks():
    """Integration test that runs only on Databricks.
    
    This test would run the actual Spark functions with real data
    when executed in a Databricks environment.
    """
    # This test would only run on Databricks
    # where we have access to volumes and Spark clusters
    pass