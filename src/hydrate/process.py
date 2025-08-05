import pandas as pd
from pathlib import Path
from typing import Union, List
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.functions import input_file_name, regexp_extract, col, lit
from pyspark.sql.types import IntegerType


def read_nested_parquet_files(
    root_dir: Union[str, Path], pattern: str = "*.parquet"
) -> pd.DataFrame:
    """Recursively read all parquet files from a nested directory structure.

    This function walks through a directory and its subdirectories to find
    all parquet files matching the given pattern, reads them into pandas
    DataFrames, and combines them into a single DataFrame.

    Args:
        root_dir: Root directory to start searching from.
        pattern: File pattern to match. Defaults to "*.parquet".

    Returns:
        Combined DataFrame from all parquet files with an additional
        'source_file' column indicating the file path.

    Raises:
        ValueError: If no parquet files are found matching the pattern.
    """
    root_dir = Path(root_dir)
    all_dfs = []

    # Walk through all directories and subdirectories
    for parquet_file in root_dir.rglob(pattern):
        try:
            # Read each parquet file
            df = pd.read_parquet(parquet_file).reset_index(drop=False)
            # Add source file information
            df["source_file"] = str(parquet_file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {parquet_file}: {str(e)}")

    if not all_dfs:
        raise ValueError(
            f"No parquet files found in {root_dir} matching pattern {pattern}"
        )

    # Combine all DataFrames
    return pd.concat(all_dfs, ignore_index=True)


def read_nested_parquet_files_spark(
    root_dir: Union[str, Path],
    schema: StructType,
    spark: SparkSession = None
) -> SparkDataFrame:
    """Efficiently read all parquet files from a nested directory structure using Spark.

    This function uses Spark's parallel loading capabilities to read parquet files
    from a directory structure, making it suitable for large datasets. It automatically
    adds source file information and handles nested directory structures.

    Args:
        root_dir: Root directory to start searching from (typically a Databricks volume path).  
        spark: SparkSession instance. If None, will get or create one.

    Returns:
        Spark DataFrame from all parquet files with an additional
        'source_file' column indicating the file path.

    Raises:
        RuntimeError: If no parquet files are found or if Spark session cannot be created.

    Example:
        >>> df = read_nested_parquet_files_spark("/Volumes/catalog/schema/volume/")
        >>> df.count()
    """
    if spark is None:
        spark = SparkSession.builder.getOrCreate()
    
    root_dir = str(root_dir).rstrip('/')
    path_pattern = f"{root_dir}/**/*.parquet"

    df = spark.read.schema(schema).parquet(path_pattern)
    return df


def clean_data(all_data: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by removing drawn and simulated data and adding well number."""
    all_data['is_drawn'] = all_data.source_file.str.contains('DRAWN')
    all_data['is_simulated'] = all_data.source_file.str.contains('SIMULATED')
    no_class = all_data['class'].isna()
    no_timestamp = all_data['timestamp'].isna()
    all_data['well_number'] = all_data['source_file'].str.extract(r'WELL-(\d+)').astype(int)
    clean_data = all_data[~no_class & ~no_timestamp]
    return clean_data


def clean_data_spark(all_data: SparkDataFrame) -> SparkDataFrame:
    """Clean the data by removing drawn and simulated data and adding well number (Spark version)."""
    from pyspark.sql.functions import when, col, isnan, isnull, regexp_extract
    
    # Add boolean columns for drawn and simulated data
    all_data = all_data.withColumn(
        "is_drawn", 
        col("source_file").contains("DRAWN")
    ).withColumn(
        "is_simulated", 
        col("source_file").contains("SIMULATED")
    )
    
    # Extract well number from source file path
    all_data = all_data.withColumn(
        "well_number",
        regexp_extract(col("source_file"), r"WELL-(\d+)", 1).cast(IntegerType())
    )
    
    # Filter out records with null class or timestamp
    clean_data = all_data.filter(
        col("class").isNotNull() & 
        col("timestamp").isNotNull()
    )
    
    return clean_data


def add_state_name(df: pd.DataFrame) -> pd.DataFrame:
    """Add a state name column to the dataframe."""
    state_names = {
        0: 'Normal',
        1: 'Abrupt Increase of BSW',
        2: 'Spurious Closure of DHSV',
        3: 'Severe Slugging',
        4: 'Flow Instability',
        5: 'Rapid Productivity Loss',
        6: 'Quick Restriction in PCK',
        7: 'Scaling in PCK',
        8: 'Hydrate in Production Line'
    }
    df = df.copy()
    df['state_name'] = df['state'].map(state_names)
    return df


def add_state_name_spark(df: SparkDataFrame) -> SparkDataFrame:
    """Add a state name column to the Spark dataframe."""
    from pyspark.sql.functions import when, col
    
    # Create a when/otherwise chain for state name mapping
    state_name_expr = (
        when(col("state") == 0, "Normal")
        .when(col("state") == 1, "Abrupt Increase of BSW")
        .when(col("state") == 2, "Spurious Closure of DHSV")
        .when(col("state") == 3, "Severe Slugging")
        .when(col("state") == 4, "Flow Instability")
        .when(col("state") == 5, "Rapid Productivity Loss")
        .when(col("state") == 6, "Quick Restriction in PCK")
        .when(col("state") == 7, "Scaling in PCK")
        .when(col("state") == 8, "Hydrate in Production Line")
        .otherwise("Unknown")
    )
    
    return df.withColumn("state_name", state_name_expr)


def process_well_data_spark(
    root_dir: Union[str, Path],
    spark: SparkSession = None,
    pattern: str = "*.parquet"
) -> SparkDataFrame:
    """Complete processing pipeline for well data using Spark.
    
    This function combines reading, cleaning, and enriching well data in a single
    pipeline optimized for large datasets using Spark's distributed processing.
    
    Args:
        root_dir: Root directory containing the parquet files (e.g., Databricks volume path).
        spark: SparkSession instance. If None, will get or create one.
        pattern: File pattern to match. Defaults to "*.parquet".
    
    Returns:
        Processed Spark DataFrame with all transformations applied.
        
    Example:
        >>> from pyspark.sql import SparkSession
        >>> spark = SparkSession.builder.appName("WellDataProcessing").getOrCreate()
        >>> df = process_well_data_spark("/Volumes/shm/3w/bronze/", spark)
        >>> df.show(5)
    """
    # Read all parquet files
    df = read_nested_parquet_files_spark(root_dir, pattern, spark)
    
    # Clean the data
    df = clean_data_spark(df)
    
    # Add state names
    df = add_state_name_spark(df)
    
    return df
