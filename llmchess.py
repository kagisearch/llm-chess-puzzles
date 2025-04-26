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
#    'gpt-4o',
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
     'gemini-2.5-pro-exp-03-25', # Example of a commented-out model
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

def solve_puzzle_with_gpt(puzzle, model):
    """
    Attempt to solve a selected chess puzzle using a given LLM model.

    puzzle: A Series containing puzzle information.
    model: An initialized llms model object.
    puzzle: A Series containing puzzle information.
    """
    board = chess.Board(puzzle['FEN'])
    moves = puzzle['Moves'].split()
    
    #print(f"Solving puzzle: {puzzle['PuzzleId']} with FEN: {puzzle['FEN']}")
    # Play the first move on the board
    first_move = moves[0]
    #print(f"Playing first move: {first_move}")
    board.push_uci(first_move)
    # Now ask the provided model to solve for the next move
    move_uci = gpt_chess_move(board, 'white' if board.turn == chess.WHITE else 'black', model)
    if move_uci is None: # Check for None explicitly, as empty string could be ambiguous
        return -1 # Indicate an error or illegal move scenario
    # Check if the predicted move matches the second move in the puzzle solution
    if len(moves) > 1 and move_uci != moves[1]:
        #print(f"GPT failed to solve the puzzle. Expected {moves[1]} but got {move_uci}")
        return 0
    else:
        #print("GPT successfully predicted the next move!")
        return 1


def gpt_chess_move(board, color, model):
    """
    Ask the provided LLM model for a chess move.

    board: chess.Board object representing the current game state.
    color: 'white' or 'black' indicating which player's move to generate.
    model: An initialized llms model object.

    Returns: A move in UCI format (e.g., 'e2e4') suggested by the model, or None on error.
    """
    # Updated prompt to emphasize legality and SAN format, and forbid check/mate symbols.
    prompt = f"You are a chess expert. Given the FEN '{board.fen()}', it is {color}'s turn to move. Provide the single best *legal* move in Standard Algebraic Notation (SAN). Examples: 'Nf3', 'O-O', 'Rxe5', 'b8=Q'. Do not include commentary, checks ('+'), or checkmates ('#'). Output only the move."

    #print(prompt)
    # Removed global model selection logic
    try:
        # Use the provided model object directly
        response = model.complete(
            prompt=prompt, temperature=0.01 # Low temperature for deterministic best move
        )

        print(response.text)
        text_response = response.text.strip().lstrip('.')
        # Take the first word and remove potential check/mate indicators ('+', '#')
        move_san = text_response.split()[0].rstrip('+#')
        move = board.parse_san(move_san)  # Converts SAN to move object
        return move.uci()  # Converts move object to UCI format
    except Exception as e:
        # Include the problematic SAN in the error message for easier debugging
        print(f"Error during GPT query or parsing SAN '{move_san}': {e}")
        print(move_san)
        return None


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
                'total_processed': 0
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
            puzzle_rating = puzzle['Rating']
            puzzle_rating_deviation = puzzle['RatingDeviation']
            # Ensure deviation is not too low for Glicko2 calculations
            puzzle_rating_deviation = max(puzzle_rating_deviation, 10) # Set a minimum deviation
            r_puzzle = env.create_rating(puzzle_rating, puzzle_rating_deviation)

            future_to_model = {
                executor.submit(solve_puzzle_with_gpt, puzzle, models[model_name]): model_name
                for model_name in MODELS_TO_TEST if model_name in models # Ensure model was initialized
            }

            results_for_puzzle = {}
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result() # result: 1 (correct), 0 (incorrect), -1 (illegal/error)
                    results_for_puzzle[model_name] = result
                except Exception as exc:
                    print(f'{model_name} generated an exception for puzzle {puzzle_index}: {exc}')
                    results_for_puzzle[model_name] = -1 # Treat exceptions as errors/illegal

            # --- Update States and Report Per-Puzzle Results ---
            print(f"Puzzle {puzzle_index} Results:")
            for model_name in MODELS_TO_TEST:
                 if model_name in models: # Check if model exists and was processed
                    state = model_states[model_name]
                    result = results_for_puzzle.get(model_name, -2) # -2 if model didn't run for some reason

                    state['total_processed'] += 1
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

                    # Per-puzzle status line
                    print(f"  - {model_name}: {outcome_str} ({state['wins']}/{state['total_processed']}), Elo: {int(state['rating_obj'].mu)}")


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
            print(f"  Solved Correctly: {state['wins']} / {state['total_processed']} ({state['wins']/state['total_processed']:.2%} accuracy)")
            print(f"  Illegal Moves: {state['illegal_moves']}")
            print(f"  Final Glicko Elo: {final_elo}")
            print(f"  Adjusted Elo (penalizing illegal): {adjusted_elo}")
            print("-" * 20)
 
