"""
Shared pytest fixtures for the LLM Chess project.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
import pandas as pd


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def mock_chess_board():
    """Mock chess board for testing chess operations."""
    import chess
    board = chess.Board()
    return board


@pytest.fixture
def sample_puzzle_data():
    """Sample puzzle data for testing."""
    data = {
        'PuzzleId': ['00001', '00002', '00003'],
        'FEN': [
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            '8/8/8/8/8/8/8/8 w - - 0 1',
            'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4'
        ],
        'Moves': ['e2e4', 'a1a8', 'd1h5'],
        'Rating': [1200, 1500, 1800],
        'RatingDeviation': [350, 80, 75],
        'Popularity': [85, 92, 78],
        'NbPlays': [1000, 1500, 800],
        'Themes': ['fork', 'pin', 'skewer'],
        'GameUrl': ['lichess.org/game1', 'lichess.org/game2', 'lichess.org/game3']
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_llm_model():
    """Mock LLM model for testing without API calls."""
    class MockLLMModel:
        def __init__(self, model_name="mock-model"):
            self.model_name = model_name
            
        def complete(self, prompt, **kwargs):
            """Return a mock response for completion."""
            return "e2e4"  # Mock chess move
            
        def chat(self, messages, **kwargs):
            """Return a mock response for chat."""
            return {"content": "e2e4"}  # Mock chess move response
    
    return MockLLMModel()


@pytest.fixture
def mock_glicko_rating():
    """Mock Glicko2 rating for testing."""
    from glicko2 import Glicko2
    env = Glicko2()
    return env.create_rating()


@pytest.fixture
def mock_connect4_game():
    """Mock Connect4 game state for testing."""
    class MockConnect4:
        def __init__(self):
            self.board = [[' ' for _ in range(7)] for _ in range(6)]
            self.is_over = False
            self.winner = None
            self.current_player = 1
            self.history = []
            
        def make_move(self, column):
            """Mock move making."""
            if column < 0 or column >= 7:
                return False
            return True
            
        def is_valid_move(self, column):
            """Check if move is valid."""
            return 0 <= column < 7 and self.board[0][column] == ' '
    
    return MockConnect4()


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing."""
    csv_path = os.path.join(temp_dir, "test_puzzles.csv")
    data = {
        'PuzzleId': ['test001', 'test002'],
        'FEN': [
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            '8/8/8/8/8/8/8/8 w - - 0 1'
        ],
        'Moves': ['e2e4', 'a1a8'],
        'Rating': [1200, 1500],
        'RatingDeviation': [350, 80],
        'Popularity': [85, 92],
        'NbPlays': [1000, 1500],
        'Themes': ['fork', 'pin'],
        'GameUrl': ['lichess.org/test1', 'lichess.org/test2']
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_env_variables():
    """Mock environment variables for testing."""
    original_env = os.environ.copy()
    test_env = {
        'OPENAI_API_KEY': 'test-key-123',
        'ANTHROPIC_API_KEY': 'test-anthropic-key-456',
    }
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_concurrent_futures():
    """Mock concurrent futures for testing parallel execution."""
    import concurrent.futures
    
    class MockFuture:
        def __init__(self, result):
            self._result = result
            
        def result(self, timeout=None):
            return self._result
            
        def done(self):
            return True
    
    class MockExecutor:
        def __init__(self):
            pass
            
        def submit(self, fn, *args, **kwargs):
            result = fn(*args, **kwargs)
            return MockFuture(result)
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
    
    return MockExecutor