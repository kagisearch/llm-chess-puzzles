import chess
import os
from glicko2 import Glicko2, WIN, DRAW, LOSS
import llms
import pandas as pd
import random
import concurrent.futures
import time # Optional: for adding delays if needed

# --- Model Selection ---
# Define the list of LLM models to test here.
# Add or remove model identifiers string from this list to change which models are evaluated.
# Example models: 'gpt-3.5-turbo', 'gpt-4-turbo-preview', 'gpt-4', 'claude-3-opus-20240229', etc.
MODELS_TO_TEST = [
    'gpt-4o',
    'gpt-4.1',
    'claude-3-7-sonnet-20250219',
    'deepseek-chat',
#    'gpt-4-turbo-preview',
    # 'gpt-3.5-turbo',
    # 'gpt-4',
    # 'claude-3-opus-20240229',
    # 'mistral-large-latest',
    # 'open-mixtral-8x7b',
    # 'claude-3-sonnet-20240229',
    # 'claude-3-haiku-20240307',
    # 'claude-instant-1.2',
    # 'gemini-1.5-pro-latest',
#     'gemini-2.5-pro-exp-03-25', # Example of a commented-out model
     'gpt-4.5-preview'                  # Example of a commented-out model
]
# --- End Model Selection ---

def calculate_elo_change2(player_rating, opponent_rating, result):
    if result == 1:
        new_player_rating, new_opponent_rating = env.rate_1vs1(player_rating, opponent_rating, drawn=False)
    elif result == 0.5:
        new_player_rating, new_opponent_rating = env.rate_1vs1(player_rating, opponent_rating, drawn=True)
    else:  # result == 0
        new_opponent_rating, new_player_rating = env.rate_1vs1(opponent_rating, player_rating, drawn=False)

def load_puzzles(filename="lichess_db_puzzle.csv"):
    """
    Load puzzles from a CSV file.
    
    Returns: DataFrame containing puzzles.
    """
    return pd.read_csv(filename)

def select_random_puzzles(puzzles):
    """
    Select a random puzzle from the DataFrame.
    
    Returns: A single puzzle as a Series.
    """
    sorted_puzzles = puzzles.assign(SortKey=puzzles['NbPlays'] * puzzles['Popularity']).sort_values(by='SortKey', ascending=False).head(100000)
    return sorted_puzzles.sample(n=1000)

def solve_puzzle_with_gpt(puzzle, model, model_name):
    """
    Attempt to solve a selected chess puzzle using a given LLM model.

    puzzle: A Series containing puzzle information.
    model: An initialized llms model object.
    model_name: The name of the model being used (for logging).
    """
    board = chess.Board(puzzle['FEN'])
    moves = puzzle['Moves'].split()
    
    #print(f"Solving puzzle: {puzzle['PuzzleId']} with FEN: {puzzle['FEN']}")
    # Play the first move on the board
    first_move = moves[0]
    #print(f"Playing first move: {first_move}")
    board.push_uci(first_move)

    # Time the move generation
    start_time = time.time()
    # Ask the model for the move and get detailed results
    move_uci, raw_response, parsed_san, error_msg = gpt_chess_move(
        board, 'white' if board.turn == chess.WHITE else 'black', model, model_name
    )
    end_time = time.time()
    duration = end_time - start_time

    # Print model name and its raw output
    print(f"{model_name}: {raw_response.strip()}") # Use strip() to clean potential newlines

    result = 0 # Default to incorrect
    outcome_str = "Incorrect"
    if move_uci is None: # Check if move generation/parsing failed
        result = -1 # Indicate an error or illegal move scenario
        outcome_str = f"Illegal/Error ({error_msg or 'Move generation failed'})" # Include error details if available
    elif len(moves) > 1 and move_uci == moves[1]:
        result = 1 # Correct
        outcome_str = "Correct"
    # else: Incorrect move (already default)

    # Print the outcome immediately after the model's response
    print(f" -> {outcome_str}")

    return result, duration # Return the numerical result and time taken


def gpt_chess_move(board, color, model, model_name):
    """
    Ask the provided LLM model for a chess move.

    board: chess.Board object representing the current game state.
    color: 'white' or 'black' indicating which player's move to generate.
    model: An initialized llms model object.
    model_name: The name of the model being used (for logging).

    Returns: A tuple: (uci_move, raw_response, parsed_san, error_message)
             uci_move: UCI string if successful, else None.
             raw_response: Raw text from the model.
             parsed_san: The cleaned SAN extracted, or attempted SAN.
             error_message: String with error details if any, else None.
    """
    # Updated prompt to emphasize legality and SAN format, and forbid check/mate symbols.
    prompt = f"You are a chess expert. Given the FEN '{board.fen()}', it is {color}'s turn to move. Provide the single best *legal* move in Standard Algebraic Notation (SAN). Examples: 'Nf3', 'O-O', 'Rxe5', 'b8=Q'. Do not include commentary, checks ('+'), or checkmates ('#'). Output only the move."

    #print(prompt)
    raw_response = ""
    move_san = ""
    try:
        # Use the provided model object directly
        response = model.complete(
            prompt=prompt, temperature=0.01 # Low temperature for deterministic best move
        )
        raw_response = response.text # Store raw response

        # Attempt to parse the response
        text_response = raw_response.strip().lstrip('.')
        # Take the first word and remove potential check/mate indicators ('+', '#')
        move_san = text_response.split()[0].rstrip('+#')
        move = board.parse_san(move_san)  # Converts SAN to move object
        return move.uci(), raw_response, move_san, None # Success
    except chess.InvalidMoveError as e:
        # Handle specific illegal move errors (move is syntactically valid SAN but illegal on board)
        error_msg = f"Illegal SAN move '{move_san}': {e}"
        print(error_msg) # Keep this print for debugging illegal moves
        return None, raw_response, move_san, error_msg
    except chess.IllegalMoveError as e:
        # Handle other parsing errors (e.g., invalid SAN format)
        error_msg = f"Invalid SAN format '{move_san}': {e}"
        print(error_msg) # Keep this print for debugging invalid SAN
        return None, raw_response, move_san, error_msg
    except Exception as e:
        # Handle other exceptions (e.g., API errors, network issues)
        error_msg = f"Error during GPT query or processing: {e}"
        # Include the problematic SAN if available, otherwise use raw_response
        problematic_input = move_san if move_san else raw_response
        print(f"Error during GPT query or parsing SAN '{problematic_input}': {e}")
        # Return None for uci_move, include raw_response if available
        return None, raw_response, problematic_input, error_msg


def play_chess_with_gpt():
    board = chess.Board()

    while not board.is_game_over():
        color = 'white' if board.turn == chess.WHITE else 'black'
        move_uci = gpt_chess_move(board, color)
        
        if move_uci and chess.Move.from_uci(move_uci) in board.legal_moves:
            move = chess.Move.from_uci(move_uci)
            board.push(move)
            print(f"{color.capitalize()}'s move: {move_uci}")
            print(board)
        else:
            print(f"Received an illegal or no move from GPT for {color}: {move_uci}")
            break

    print("Game over. Result:", board.result())


# this is a mode to just have models play chess in real time
#if __name__ == "__main__":
    #model1=llms.init('gpt-4-turbo-preview')
    #model2=llms.init('gpt-4')
    #play_chess_with_gpt()
    
if __name__ == "__main__":
   
    # Code to randomly select puzzles from Lichess dataset
    #puzzles = load_puzzles()
    #puzzles=select_random_puzzles(puzzles)
    #puzzles.to_csv('puzzles.csv', index=False) # Keep this commented unless regenerating puzzles

    puzzles = pd.read_csv('puzzles.csv')
    # Optional: Limit number of puzzles for faster testing
    # puzzles = puzzles.head(50)

    env = Glicko2(tau=0.5) # Initialize Glicko2 environment

    # --- Initialize Models and States ---
    print("Initializing models...")
    models = {}
    model_states = {}
    for model_name in MODELS_TO_TEST:
        try:
            models[model_name] = llms.init(model_name)
            model_states[model_name] = {
                'rating_obj': env.create_rating(), # Initial rating (1500, 350, 0.06)
                'score': 0,
                'wins': 0,      # Correct solves
                'losses': 0,    # Incorrect solves (excluding illegal)
                'illegal_moves': 0,
                'total_processed': 0,
                'total_time': 0.0 # Add total time tracker
            }
            print(f"Initialized {model_name}")
        except Exception as e:
            print(f"Failed to initialize {model_name}: {e}")
            # Remove model if initialization failed
            if model_name in MODELS_TO_TEST:
                 MODELS_TO_TEST.remove(model_name)

    print("\nStarting puzzle solving...")
    # --- Process Puzzles in Parallel for Each Model ---
    # Use max_workers to control concurrency, adjust based on API limits and system resources
    # Setting a lower number like 5 or 10 might be safer for API rate limits.
    MAX_WORKERS = len(MODELS_TO_TEST) # Can adjust this number

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for index, puzzle in puzzles.iterrows():
            puzzle_index = index + 1
            print(f"\n--- Processing Puzzle {puzzle_index}/{len(puzzles)} (ID: {puzzle['PuzzleId']}, Rating: {puzzle['Rating']}) ---")

            # Extract and print the correct move for the puzzle
            correct_moves = puzzle['Moves'].split()
            if len(correct_moves) > 1:
                print(f"Correct Move: {correct_moves[1]}")
            else:
                print("Correct Move: (Not available - puzzle has fewer than 2 moves)")

            puzzle_rating = puzzle['Rating']
            puzzle_rating_deviation = puzzle['RatingDeviation']
            # Ensure deviation is not too low for Glicko2 calculations
            puzzle_rating_deviation = max(puzzle_rating_deviation, 10) # Set a minimum deviation
            r_puzzle = env.create_rating(puzzle_rating, puzzle_rating_deviation)

            future_to_model = {}
            for model_name in MODELS_TO_TEST:
                if model_name in models: # Ensure model was initialized
                    # Pass model_name to solve_puzzle_with_gpt
                    future = executor.submit(solve_puzzle_with_gpt, puzzle, models[model_name], model_name)
                    future_to_model[future] = model_name

            results_for_puzzle = {}
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                duration = 0.0 # Default duration if exception occurs before timing
                try:
                    # Unpack the result and duration
                    result, duration = future.result() # result: 1 (correct), 0 (incorrect), -1 (illegal/error)
                    results_for_puzzle[model_name] = (result, duration)
                except Exception as exc:
                    print(f'{model_name} generated an exception for puzzle {puzzle_index}: {exc}')
                    results_for_puzzle[model_name] = (-1, duration) # Treat exceptions as errors/illegal, keep duration if available

            # --- Update States and Report Per-Puzzle Results ---
            print(f"Puzzle {puzzle_index} Results:")
            for model_name in MODELS_TO_TEST:
                 if model_name in models: # Check if model exists and was processed
                    state = model_states[model_name]
                    # Get result and duration, provide defaults if missing
                    result, duration = results_for_puzzle.get(model_name, (-2, 0.0))

                    state['total_processed'] += 1
                    state['total_time'] += duration # Accumulate time
                    r_model = state['rating_obj']
                    outcome_str = ""

                    if result == 1: # Correct
                        state['wins'] += 1
                        state['score'] += 1
                        new_r_model, _ = env.rate_1vs1(r_model, r_puzzle, drawn=False)
                        outcome_str = "Correct"
                    elif result == 0: # Incorrect
                        state['losses'] += 1
                        # Score doesn't change
                        _, new_r_model = env.rate_1vs1(r_puzzle, r_model, drawn=False)
                        outcome_str = "Incorrect"
                    elif result == -1: # Illegal / Error
                        state['illegal_moves'] += 1
                        # Treat as loss for rating purposes
                        _, new_r_model = env.rate_1vs1(r_puzzle, r_model, drawn=False)
                        outcome_str = "Illegal/Error"
                    else: # Should not happen normally
                         new_r_model = r_model # No change if result is unexpected
                         outcome_str = "Unknown"


                    state['rating_obj'] = new_r_model # Update rating object in state

                    # Per-puzzle status line including time
                    print(f"  - {model_name}: {outcome_str} ({state['wins']}/{state['total_processed']}), Elo: {int(state['rating_obj'].mu)}, Time: {duration:.2f}s")


    # --- Final Summary ---
    print("\n--- Final Results ---")
    for model_name in MODELS_TO_TEST:
         if model_name in models:
            state = model_states[model_name]
            final_elo = int(state['rating_obj'].mu)
            # Calculate adjusted Elo (optional, based on previous logic)
            total_games_for_adj = state['wins'] + state['losses'] + state['illegal_moves']
            adjusted_elo = final_elo
            if total_games_for_adj > 0:
                 # Penalize based on illegal moves proportion relative to all attempts
                 adjusted_elo = int(final_elo * (1 - state['illegal_moves'] / state['total_processed']))

            print(f"{model_name}:")
            print(f"  Score: {state['score']}")
            accuracy = (state['wins'] / state['total_processed'] * 100) if state['total_processed'] > 0 else 0
            avg_time = state['total_time'] / state['total_processed'] if state['total_processed'] > 0 else 0
            print(f"  Solved Correctly: {state['wins']} / {state['total_processed']} ({accuracy:.2f}% accuracy)")
            print(f"  Illegal Moves: {state['illegal_moves']}")
            print(f"  Final Glicko Elo: {final_elo}")
            print(f"  Adjusted Elo (penalizing illegal): {adjusted_elo}")
            print(f"  Average Time per Puzzle: {avg_time:.2f}s")
            print("-" * 20)
 
