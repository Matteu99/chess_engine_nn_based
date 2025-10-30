import chess
import chess.pgn
import numpy as np
import h5py
import pickle


# Chess position converted into (8, 8, 6) tensor
def board_to_tensor(board):
    tensor = np.zeros((8, 8, 6), dtype=np.float16)

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
            tensor[row, col, channel] = 1.0 if piece.color == chess.WHITE else -1.0

    return tensor

# Returning game value position based on the final result of the game
def game_result(result, turn):
    if turn == True: # White move
        if result == "1-0":
            position_value = 1  # White won
        elif result == "0-1":
            position_value = -1  # Black won
        else:
            position_value = 0  # Draw
    else: # Black move
        if result == "1-0":
            position_value = -1  # White won
        elif result == "0-1":
            position_value = 1  # Black won
        else:
            position_value = 0  # Draw

    return position_value

def legal_moves_in_position(board, label_encoder):
    legal_indices = []
    for move in board.legal_moves:
        move_uci = move.uci()
        if move_uci in label_encoder.classes_:
            move_idx = label_encoder.transform([move_uci])[0]
            legal_indices.append(move_idx)
    return legal_indices

# Function for preprocessing games from PGN files and save to X i y
def preprocess_pgn_to_h5(pgn_filename, x_filename='X_data.h5', y_filename='y_data.h5', position_values='position_values.h5', lm_filename = "legal_moves.h5",batch_size=100):
    # Load LabelEncoder from current folder
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)

    # Open PGN file and initialize HDF5 files to save data
    with open(pgn_filename, 'r') as pgn_file, \
            h5py.File(x_filename, 'w') as x_file, \
            h5py.File(y_filename, 'w') as y_file, \
            h5py.File(position_values, 'w') as value_file, \
            h5py.File(lm_filename, 'w') as lm_file:


        # HDF5 dataset prep (dynamic memory allocation)
        x_dataset = x_file.create_dataset(
            'X', (0, 8, 8, 6), maxshape=(None, 8, 8, 6), dtype='float16',
            chunks=True, compression="gzip", compression_opts=4
        )
        y_dataset = y_file.create_dataset(
            'y', (0,), maxshape=(None,), dtype='int16',
            chunks=True, compression="gzip", compression_opts=4
        )
        pos_values_dataset = value_file.create_dataset(
            'pos_values', (0,), maxshape=(None,), dtype='int16',
            chunks=True, compression="gzip", compression_opts=4
        )
        lm_dataset = lm_file.create_dataset(
            'lm', (0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.int16), ### This dtype is responsible for different vector size
            chunks=True, compression="gzip", compression_opts=4
        )

        game_count = 0

        # Batch buffers
        X_buffer = []
        y_buffer = []
        pos_values_buffer = []
        lm_buffer = []

        # Through all the games in pgn file
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            board = game.board()
            result = game.headers.get("Result") # Get the result of the game
            idx = 0

            for move in game.mainline_moves():
                # Chess position into (8,8, 6) tensor
                x_tensor = board_to_tensor(board)

                # Move coding with LabelEncoder
                move_uci = move.uci()
                if move_uci in label_encoder.classes_:
                    y_encoded = label_encoder.transform([move_uci])[0]

                    # Adding data to the buffer
                    X_buffer.append(x_tensor)
                    y_buffer.append(y_encoded)
                    pos_values_buffer.append(game_result(result, board.turn)) # Set a value for a position based on end game result
                    lm_buffer.append(legal_moves_in_position(board, label_encoder))

                    # Buffer save
                    if len(X_buffer) >= batch_size:
                        save_batch_to_h5(x_dataset, y_dataset, pos_values_dataset, lm_dataset, X_buffer, y_buffer, pos_values_buffer, lm_buffer)
                        X_buffer, y_buffer, pos_values_buffer, lm_buffer = [], [], [], []

                    # Chess board update
                    board.push(move)

                    idx += 1

            print(game_count)
            game_count += 1
            if game_count % 100 == 0:
                print(f"{game_count} games were processed")

        # Save the remaining buffer
        if len(X_buffer) > 0:
            save_batch_to_h5(x_dataset, y_dataset, pos_values_dataset, lm_dataset, X_buffer, y_buffer, pos_values_buffer, lm_buffer)

# Save batch to HDF5 file
def save_batch_to_h5(x_dataset, y_dataset, pos_values_dataset, lm_dataset, X_buffer, y_buffer, pos_values_buffer, lm_buffer):
    X_batch_np = np.array(X_buffer, dtype='float16')
    y_batch_np = np.array(y_buffer, dtype='int16')
    pos_values_batch_np = np.array(pos_values_buffer, dtype='int16')
    lm_batch_np = np.array(lm_buffer, dtype=h5py.vlen_dtype(np.int16))


    x_dataset.resize(x_dataset.shape[0] + X_batch_np.shape[0], axis=0)
    x_dataset[-X_batch_np.shape[0]:] = X_batch_np

    y_dataset.resize(y_dataset.shape[0] + y_batch_np.shape[0], axis=0)
    y_dataset[-y_batch_np.shape[0]:] = y_batch_np

    pos_values_dataset.resize(pos_values_dataset.shape[0] + pos_values_batch_np.shape[0], axis=0)
    pos_values_dataset[-pos_values_batch_np.shape[0]:] = pos_values_batch_np

    lm_dataset.resize(lm_dataset.shape[0] + lm_batch_np.shape[0], axis=0)
    lm_dataset[-lm_batch_np.shape[0]:] = lm_batch_np

# Calling the preprocess function
preprocess_pgn_to_h5(
    pgn_filename=r"ficsgamesdb_202408_standard2000_nomovetimes_402382.pgn", # path to the pgn file with chess games
    x_filename='X_data.h5',
    y_filename='y_data.h5',
    position_values='position_values.h5',
    lm_filename = "legal_moves.h5",
    batch_size=100
)



