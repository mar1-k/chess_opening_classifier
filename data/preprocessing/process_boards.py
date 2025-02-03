import pandas as pd
import chess
import chess.pgn
import io
import re
from multiprocessing import Pool, cpu_count
from functools import partial

def clean_moves(an_text):
    """Clean the move text by removing evaluation comments and move numbers."""
    # Remove evaluation comments like { [%eval 0.23] }
    cleaned = re.sub(r'\{[^}]*\}', '', an_text)
    # Remove move numbers and result
    cleaned = re.sub(r'\d+\.', '', cleaned)
    # Remove the game result
    cleaned = re.sub(r'1-0|0-1|1/2-1/2|\*', '', cleaned)
    return cleaned.strip()

def game_to_positions(game_data):
    """Convert a single game's moves to a list of positions with metadata."""
    eco, opening, moves_text = game_data
    
    # Clean the moves text
    moves_text = clean_moves(moves_text)
    
    try:
        # Create a new game and board
        game = chess.pgn.read_game(io.StringIO(moves_text))
        if game is None:
            return []
        
        board = game.board()
        positions = []
        
        # Play through each move and record the position
        for move_num, move in enumerate(game.mainline_moves(), start=1):
            board.push(move)
            positions.append({
                'ECO': eco,
                'Opening': opening,
                'Position': board.fen(),
                'MoveNumber': move_num
            })
        
        return positions
    except Exception as e:
        print(f"Error processing game: {opening}")
        print(f"Error: {str(e)}")
        return []

def process_chunk(chunk):
    """Process a chunk of games in parallel."""
    # Convert chunk to list of tuples for processing
    games_data = list(zip(chunk['ECO'], chunk['Opening'], chunk['AN']))
    
    # Process each game in the chunk
    all_positions = []
    for game_data in games_data:
        positions = game_to_positions(game_data)
        all_positions.extend(positions)
    
    return all_positions

def process_chess_games(input_file: str, output_file: str, chunk_size: int = 10000):
    """
    Process chess games from CSV and convert to FEN positions using parallel processing.
    
    Args:
        input_file (str): Path to input CSV with ECO, Opening, AN columns
        output_file (str): Path to output CSV
        chunk_size (int): Number of games to process at once
    """
    # Initialize multiprocessing pool
    num_cores = cpu_count()
    pool = Pool(num_cores)
    print(f"Using {num_cores} CPU cores")
    
    chunks_processed = 0
    total_games_processed = 0
    
    print("Processing games...")
    
    # Process file in chunks
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Split chunk into sub-chunks for parallel processing
        sub_chunk_size = chunk_size // num_cores
        sub_chunks = [chunk[i:i + sub_chunk_size] for i in range(0, len(chunk), sub_chunk_size)]
        
        # Process sub-chunks in parallel
        results = pool.map(process_chunk, sub_chunks)
        
        # Combine results
        all_positions = []
        for result in results:
            all_positions.extend(result)
        
        # Convert to DataFrame
        positions_df = pd.DataFrame(all_positions)
        
        # Write to CSV
        positions_df.to_csv(
            output_file,
            mode='a' if chunks_processed > 0 else 'w',
            header=(chunks_processed == 0),
            index=False
        )
        
        chunks_processed += 1
        total_games_processed += len(chunk)
        print(f"Processed {total_games_processed:,} games... ({len(all_positions):,} positions)")
        
    pool.close()
    pool.join()

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "data/chess_games_columns_reduced.csv"
    OUTPUT_FILE = "data/chess_boards.csv"
    CHUNK_SIZE = 10000
    
    # Process the games
    process_chess_games(INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE)
    
    # Verify output
    print("\nVerifying output file...")
    output_sample = pd.read_csv(OUTPUT_FILE, nrows=5)
    print("\nFirst few rows of the output file:")
    print(output_sample)