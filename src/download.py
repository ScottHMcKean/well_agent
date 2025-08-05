#!/usr/bin/env python3
"""
3W Dataset Downloader

Downloads parquet files from the Petrobras 3W dataset on GitHub.
Uses injected configuration object for flexibility and performance.
"""

import requests
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse
from .utils import DotConfig


class DatasetDownloader:
    """Downloads files from the Petrobras 3W dataset using injected configuration."""

    def __init__(self, config: DotConfig):
        """
        Initialize downloader with configuration object.

        Args:
            config: DotConfig object with dot notation access
        """
        self.config = config

        # Validate that config has required download section
        if not hasattr(config, "download"):
            raise ValueError("Config must have 'download' section")

        self.download_config = config.download

        # Get configuration values with defaults using dot notation
        self.base_url = (
            self.download_config.base_url
            or "https://api.github.com/repos/petrobras/3W/contents"
        )
        self.session = requests.Session()

    def get_directory_contents(self, path: str = "dataset") -> List[Dict]:
        """Get contents of a directory from GitHub API."""
        url = f"{self.base_url}/{path}"
        response = self.session.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching {url}: {response.status_code}")
            return []

    def download_file(self, download_url: str, local_path: Path) -> bool:
        """Download a single file."""
        try:
            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if we should skip existing files
            skip_existing = (
                self.download_config.skip_existing
                if hasattr(self.download_config, "skip_existing")
                else True
            )
            if skip_existing and local_path.exists():
                return True

            print(f"Downloading: {local_path.name}")
            response = self.session.get(download_url, stream=True)

            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✓ Downloaded: {local_path}")
                return True
            else:
                print(f"✗ Failed to download {download_url}: {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Error downloading {download_url}: {e}")
            return False

    def get_all_parquet_files(self, max_dirs: Optional[int] = None) -> List[Dict]:
        """Get list of all parquet files in the dataset."""
        files = []

        # Get dataset subdirectories
        subdirs = self.get_directory_contents("dataset")
        print(f"Found {len(subdirs)} total directories")

        # Apply max_dirs limit properly
        if max_dirs and max_dirs > 0:
            print(f"Applying max_dirs limit: {max_dirs}")
            subdirs = subdirs[:max_dirs]
            print(f"Limited to first {max_dirs} directories")
        else:
            print("No directory limit applied")

        for subdir in subdirs:
            if subdir.get("type") == "dir":
                subdir_name = subdir["name"]
                print(f"Scanning directory: {subdir_name}")

                # Get files in subdirectory
                subdir_files = self.get_directory_contents(f"dataset/{subdir_name}")

                for file_info in subdir_files:
                    if file_info.get("type") == "file" and file_info.get(
                        "name", ""
                    ).endswith(".parquet"):
                        files.append(
                            {
                                "name": file_info["name"],
                                "path": file_info["path"],
                                "download_url": file_info["download_url"],
                                "size": file_info["size"],
                                "subdir": subdir_name,
                            }
                        )

        return files

    def download_dataset(self) -> None:
        """Download the dataset according to configuration."""

        # Get configuration values using dot notation
        output_dir = self.download_config.output_dir or "3w_dataset"
        max_files = self.download_config.max_files
        max_dirs = self.download_config.max_dirs
        delay_seconds = self.download_config.delay_seconds or 0.1

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"Configuration loaded from config")
        print(f"Output directory: {output_dir}")
        print(f"Max files setting: {max_files} (type: {type(max_files)})")
        print(f"Max directories setting: {max_dirs} (type: {type(max_dirs)})")

        if max_files and max_files > 0:
            print(f"Will limit to {max_files} files")
        else:
            print("No file limit (downloading all files)")
        if max_dirs and max_dirs > 0:
            print(f"Will limit to {max_dirs} directories")
        else:
            print("No directory limit (scanning all directories)")

        print(f"Getting file list from 3W dataset...")

        # Apply max_dirs limit when getting files
        files = self.get_all_parquet_files(max_dirs=max_dirs)

        print(f"Found {len(files)} total parquet files")

        # Apply max_files limit properly
        if max_files and max_files > 0:
            print(f"Applying max_files limit: {max_files}")
            files = files[:max_files]
            print(f"Limited to first {max_files} files")
        else:
            print("No file limit applied")

        print(f"Will download {len(files)} parquet files")

        successful_downloads = 0
        total_size = 0
        skipped_files = 0

        for i, file_info in enumerate(files, 1):
            local_path = output_path / file_info["subdir"] / file_info["name"]

            # Skip if file already exists
            if local_path.exists():
                print(f"[{i}/{len(files)}] Skipping existing file: {file_info['name']}")
                successful_downloads += 1
                skipped_files += 1
                continue

            print(
                f"[{i}/{len(files)}] Downloading {file_info['name']} ({file_info['size']} bytes)"
            )

            if self.download_file(file_info["download_url"], local_path):
                successful_downloads += 1
                total_size += file_info["size"]

            # Delay between downloads
            time.sleep(delay_seconds)

        print(f"\nDownload complete!")
        print(f"Successfully processed: {successful_downloads}/{len(files)} files")
        if skipped_files > 0:
            print(f"Skipped existing files: {skipped_files}")
        print(f"New downloads: {successful_downloads - skipped_files}")
        print(f"Total new data size: {total_size / (1024*1024):.2f} MB")
        print(f"Files saved to: {output_path.absolute()}")


def main():
    """Main entry point for testing the downloader."""
    # Import here to avoid circular imports
    from .utils import DotConfig

    config = DotConfig()
    downloader = DatasetDownloader(config)
    downloader.download_dataset()


if __name__ == "__main__":
    main()
