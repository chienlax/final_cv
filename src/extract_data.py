"""
Extract JSONL.gz files to uncompressed JSONL for faster loading.

Usage:
    python src/extract_data.py
"""

import gzip
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")

# Files to extract
FILES = [
    "Beauty_and_Personal_Care.jsonl.gz",
    "Clothing_Shoes_and_Jewelry.jsonl.gz",
    "meta_Beauty_and_Personal_Care.jsonl.gz",
    "meta_Clothing_Shoes_and_Jewelry.jsonl.gz",
]


def extract_gzip(input_path: Path, output_path: Path) -> None:
    """Extract a gzipped file."""
    logger.info(f"Extracting {input_path.name} -> {output_path.name}")
    
    with gzip.open(input_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Report sizes
    input_size = input_path.stat().st_size / 1024 / 1024 / 1024
    output_size = output_path.stat().st_size / 1024 / 1024 / 1024
    logger.info(f"  {input_size:.2f} GB -> {output_size:.2f} GB (ratio: {output_size/input_size:.1f}x)")


def main():
    """Extract all JSONL.gz files."""
    logger.info("Starting extraction...")
    
    for filename in FILES:
        input_path = DATA_DIR / filename
        output_path = DATA_DIR / filename.replace(".gz", "")
        
        if not input_path.exists():
            logger.warning(f"File not found: {input_path}")
            continue
        
        if output_path.exists():
            logger.info(f"Already extracted: {output_path.name}")
            continue
        
        extract_gzip(input_path, output_path)
    
    logger.info("Extraction complete!")


if __name__ == "__main__":
    main()
