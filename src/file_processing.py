import pandas as pd
from pathlib import Path
from typing import Union, List


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
