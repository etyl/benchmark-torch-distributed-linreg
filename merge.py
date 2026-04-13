import pandas as pd
import sys

def merge_parquet_files(file1: str, file2: str, output: str) -> None:
    """
    Merge two parquet files and save the result.
    
    Args:
        file1: Path to the first parquet file
        file2: Path to the second parquet file
        output: Path to the output parquet file
    """
    df1 = pd.read_parquet(file1)
    df2 = pd.read_parquet(file2)
    
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df.to_parquet(output)
    
    print(f"Merged {len(df1)} + {len(df2)} rows into {output}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge.py <file1.parquet> <file2.parquet> <output.parquet>")
        sys.exit(1)
    
    merge_parquet_files(sys.argv[1], sys.argv[2], sys.argv[3])