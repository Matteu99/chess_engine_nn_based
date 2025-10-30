import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
import chess
import chess.pgn
import random
from networktraining import AdvancedChessPolicyNetwork

# Monte Carlo Tree Search (MCTS) for AlphaZero
class MCTS:
    def __init__(self, model, n_simulations=800):
        self.model = model
        self.n_simulations = n_simulations
        self.tree = {} ### Saving each node position in FEN

    def search(self, board): # The only function called on an instance
        for _ in range(self.n_simulations):
            self.simulate(board.copy()) # Copying the board game and making simulation to find the best move
        return self.select_best_move(board)

    def simulate(self, board): # Simulation phase: Gets Policy Head and Value Head for a current position and make a move till the game is over
        states = []
        while not board.is_game_over():
            states.append(board.copy())
            policy, value = self.predict(board)
            move = self.select_move(policy, board)
            board.push(move)
            if len(states) > 50:  # Steps limit
                break
        
        # Gets the final value of the position and backpropagate it
        final_value = self.evaluate_end_state(board)
        for state in states:
            self.backpropagate(state, final_value)
            final_value = -final_value # Opposite position value for each perspective

    def select_best_move(self, board):
        legal_moves = list(board.legal_moves)
        fen = board.fen()

        if fen in self.tree:
            move_probs = self.tree[board.fen()]['policy']
            best_move = legal_moves[np.argmax(move_probs)]
        else:
            best_move = random.choice(legal_moves)
        return best_move

    # Return the value of a current position
    def evaluate_end_state(self, board):
        if board.is_checkmate():
            return -1  # Lose for the player who is on move
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0  # Draw
        else:
            return 0 # if the game is still on

    def select_move(self, policy, board):
        legal_moves = list(board.legal_moves)
        move_probs = []
        for move in legal_moves:
            move_uci = move.uci()
            idx = self.policy_move_index(move_uci)
            move_probs.append(policy[idx])
        move_probs = np.array(move_probs) / sum(move_probs)
        return np.random.choice(legal_moves, p=move_probs)

    def predict(self, board):
        input_tensor = self.board_to_input(board)
        input_tensor = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)  # Reorganization (batch_size, channels, height, width)
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            policy, value = self.model(input_tensor)
        policy = policy.cpu().numpy().flatten()
        value = value.cpu().item()
        legal_moves = list(board.legal_moves)
        legal_mask = np.zeros_like(policy)
        for move in legal_moves:
            idx = self.policy_move_index(move.uci())
            legal_mask[idx] = 1.0
        policy *= legal_mask
        if policy.sum() > 0:
            policy /= policy.sum()

        fen = board.fen()
        if fen not in self.tree:
            self.tree[fen] = {
                'policy': policy,
                'N': 0,
                'W': 0,
                'Q': 0
            }
        return policy, value

    def backpropagate(self, state, value):
        fen = state.fen()
        if fen not in self.tree:
            self.tree[fen] = {'N': 0, 'W': 0, 'Q': 0}
        else:
            self.tree[fen]['N'] += 1
            self.tree[fen]['W'] += value
            self.tree[fen]['Q'] = self.tree[fen]['W'] / self.tree[fen]['N']

    def board_to_input(self, board): # Encode the board into (8, 8, 6) tensor
        tensor = np.zeros((8, 8, 6), dtype=np.float32)
        piece_to_channel = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                row, col = divmod(square, 8)
                channel = piece_to_channel[piece.piece_type]
                if piece.color == chess.WHITE:
                    tensor[row, col, channel] = 1
                else:
                    tensor[row, col, channel] = -1
        return torch.tensor(tensor, dtype=torch.float32)

    def policy_move_index(self, move_uci): # Loading LabelEncoder and return move index
        with open('/kaggle/input/label-encoder/label_encoder.pkl', 'rb') as le_file:
            label_encoder = pickle.load(le_file)
        return label_encoder.transform([move_uci])[0]

# Generating chess games data using MCTS and saving into PGN file
def play_and_save_games(model, n_games=10, n_simulations=1, output_filename="games.pgn"):
    mcts = MCTS(model, n_simulations=n_simulations)
    games = []

    for game_idx in tqdm(range(n_games), desc="Playing games"):
        board = chess.Board()
        game = chess.pgn.Game()
        node = game
        while not board.is_game_over():
            move = mcts.search(board)
            board.push(move)
            node = node.add_variation(move) # saving moves in pgn file
            print(len(games))

        games.append(game)

    # Save games to the PGN file
    with open(output_filename, "w") as pgn_file:
        for game in games:
            pgn_file.write(str(game) + "\n\n")
    print(f"Saved {n_games} games to {output_filename}")


model = AdvancedChessPolicyNetwork(input_channels=6, output_size=4672) # Create an instance of a class

# Load nn model if it exists
if os.path.exists("/kaggle/input/42-64acc-new-model/advanced_policy_value_network13.pth"):
    model.load_state_dict(torch.load("/kaggle/input/42-64acc-new-model/advanced_policy_value_network13.pth"))
    print("Loaded existing model weights.")
else:
    print("No pretrained model found. Initializing new model.")

# Plan 'n_games' number of games, with 'n_simulations' simulations and save them to the 'game.pgn'
play_and_save_games(model, n_games=10, n_simulations=50, output_filename="games.pgn")