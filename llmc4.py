import os

import llms

# Initialize GPT models
model1 = llms.init('gpt-4o')
model2 = llms.init('gpt-4o')
#model2 = llms.init('claude-3-opus-20240229')


class Connect4:
    def __init__(self):
        self.board = [[' ' for _ in range(7)] for _ in range(6)]
        self.is_over = False
        self.winner = None
        self.current_player = 1  # Initialize current_player
        self.history = []  # Initialize history to track moves

    def __str__(self):
        board_str = "+".join(["-" * 3 for _ in range(7)]) + "\n"
        for row in self.board:
            board_str += "|".join([" " + col + " " for col in row]) + "\n"
        return board_str

    def get_board_notation(self):
        columns = []
        for col in range(7):
            column_str = ""
            for row in reversed(range(6)):
                if self.board[row][col] == 'X':
                    column_str += 'X'
                elif self.board[row][col] == 'O':
                    column_str += 'O'
                
            columns.append(column_str if column_str.strip() else '.')
        return ",".join(columns)

    def is_game_over(self):
        return self.is_over

    def is_winner(self, player):
        return self.winner == player

    def is_valid_move(self, column):
        return self.board[0][column] == ' '

    def make_move(self, column):
        player_symbol = 'X' if self.current_player == 1 else 'O'
        for row in reversed(self.board):
            if row[column] == ' ':
                row[column] = player_symbol
                break
        self.history.append({"role": "user", "content": f"Player {player_symbol} plays column {column}. Board state: {self.get_board_notation()}"})
        self.check_winner()

    def check_winner(self):
        # Horizontal, vertical, and diagonal checks
        for row in range(6):
            for col in range(7):
                if self.board[row][col] != ' ':
                    if col <= 3 and all(self.board[row][col+i] == self.board[row][col] for i in range(4)):
                        self.is_over = True
                        self.winner = 1 if self.board[row][col] == 'X' else 2
                        return
                    if row <= 2 and all(self.board[row+i][col] == self.board[row][col] for i in range(4)):
                        self.is_over = True
                        self.winner = 1 if self.board[row][col] == 'X' else 2
                        return
                    if row <= 2 and col <= 3 and all(self.board[row+i][col+i] == self.board[row][col] for i in range(4)):
                        self.is_over = True
                        self.winner = 1 if self.board[row][col] == 'X' else 2
                        return
                    if row >= 3 and col <= 3 and all(self.board[row-i][col+i] == self.board[row][col] for i in range(4)):
                        self.is_over = True
                        self.winner = 1 if self.board[row][col] == 'X' else 2
                        return
        # Check for draw
        if all(self.board[0][col] != ' ' for col in range(7)):
            self.is_over = True
            self.winner = None

def gpt_connect4_move(game, model):
    """
    Ask GPT for a Connect Four move.
    game: Connect4 game object representing the current game state.
    model: The GPT model to use for generating the move.
    
    Returns: Column number (0-6) where the move is to be made.
    """
    board_notation = game.get_board_notation()
    print(board_notation)
    system_prompt = f"""You are AI playing a game of connect four. The current connect four board state is represented by columns from bottom to top, using 'X' for black discs, 'O' for white discs, and '.' for empty cells. 
    
    For example: O,X,X,XOXO,.,.,.

    Describes this board state:
    Column 0: O
    Column 1: X.
    Column 2: X
    Column 3: XOXO (from bottom up)
    Column 4: Empty. 
    Column 5: Empty.
    Column 6: Empty.

    You will be given board state and whose turn is it play (X or O). You have to Output the best column (0-6) to place your disc stricly in the following format <column>output number (0 - 6) for column</column> For example <columnn>3</column>"""
    
    player_symbol = 'X' if game.current_player == 1 else 'O'
    prompt=f"The board state is: {board_notation}\nPlayer {player_symbol} to move. "
    try:
        #print(prompt)
#        print(game.history)
        
        response = model.complete(prompt=prompt, system_message=system_prompt, max_tokens=500)#, history=game.history)
        
        print(response.text)
        # Extract the column number from the <column> tag
        import re
        match = re.search(r'<column>(\d+)</column>', response.text)
        if match:
            column = int(match.group(1))
        else:
            raise ValueError("No valid column found in the response.")
        #game.history.append({"role": "assistant", "content": response.text})
        return column
    except Exception as e:
        print(f"Error during GPT query or parsing: {e}")
        return None

def play_connect4_with_gpt():
    game = Connect4()
   
    while not game.is_game_over():
        model = model1 if game.current_player == 1 else model2  # Use game.current_player instead of current_player
        column = gpt_connect4_move(game, model)
        
        if column is not None and game.is_valid_move(column):
            game.make_move(column)
            print(f"Player {'X' if game.current_player == 1 else 'O'}'s move: Column {column}")  # Use game.current_player
            print(game)
            game.current_player = 3 - game.current_player  # Update current_player within the game object
        else:
            print(f"Received an illegal or no move from GPT for Player {game.current_player}: Column {column}")  # Use game.current_player
            break

    if game.is_winner(1):
        print("Player 1 wins!")
    elif game.is_winner(2):
        print("Player 2 wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    play_connect4_with_gpt()