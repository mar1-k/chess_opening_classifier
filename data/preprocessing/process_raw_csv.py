import pandas as pd
import os

def process_chess_data(input_file: str, output_file: str, chunk_size: int = 100000):
    """
    Process a large chess games CSV file to extract specific columns.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
        chunk_size (int): Number of rows to process at once
    """
    # Create empty output file
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Process the file in chunks
    chunks_processed = 0
    total_rows_processed = 0
    
    print("Processing data...")
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Select only the required columns
        reduced_chunk = chunk[['ECO', 'Opening', 'AN']]
        
        # Write to CSV
        # If it's the first chunk, write with headers
        # If not, append without headers
        reduced_chunk.to_csv(
            output_file,
            mode='a',
            header=(chunks_processed == 0),
            index=False
        )
        
        chunks_processed += 1
        total_rows_processed += len(chunk)
        
        # Print progress update every 10 chunks
        if chunks_processed % 10 == 0:
            print(f"Processed {total_rows_processed:,} rows...")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "chess_games_raw.csv" 
    OUTPUT_FILE = "chess_games_columns_reduced.csv"
    CHUNK_SIZE = 100000 
    
    # Process the data
    process_chess_data(INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE)
    
    # Verify the output
    print("\nVerifying output file...")
    output_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)  # Size in MB
    print(f"Output file size: {output_size:.2f} MB")
    
    # Read a few rows to verify content
    print("\nFirst few rows of the output file:")
    print(pd.read_csv(OUTPUT_FILE, nrows=5))