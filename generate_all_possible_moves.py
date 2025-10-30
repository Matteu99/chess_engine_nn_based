import itertools
import chess
from sklearn.preprocessing import LabelEncoder
import pickle

# Generate a list of all possible moves in chess
def generate_all_possible_moves():
    moves = []

    # Przechodzenie przez wszystkie możliwe pozycje i generowanie ruchów
    for from_square, to_square in itertools.product(chess.SQUARES, repeat=2):
        move = chess.Move(from_square, to_square)
        moves.append(move.uci())

    # Dodanie promocji (zwykłe oraz z biciem) dla białych i czarnych pionków
    promotion_pieces = ['q', 'r', 'b', 'n']

    # Promocje dla białych (7. rząd -> 8. rząd)
    for file in range(8):
        from_square = chess.square(file, 6)

        # Zwykła promocja do przodu
        to_square = chess.square(file, 7)
        for piece in promotion_pieces:
            move = chess.Move(from_square, to_square, promotion=chess.Piece.from_symbol(piece.lower()).piece_type)
            moves.append(move.uci())

        # Bicie w lewo w trakcie promocji
        if file > 0:  # jeśli nie jest na linii 'a'
            to_square = chess.square(file - 1, 7)
            for piece in promotion_pieces:
                move = chess.Move(from_square, to_square, promotion=chess.Piece.from_symbol(piece.lower()).piece_type)
                moves.append(move.uci())

        # Bicie w prawo w trakcie promocji
        if file < 7:  # jeśli nie jest na linii 'h'
            to_square = chess.square(file + 1, 7)
            for piece in promotion_pieces:
                move = chess.Move(from_square, to_square, promotion=chess.Piece.from_symbol(piece.lower()).piece_type)
                moves.append(move.uci())

    # Promocje dla czarnych (2. rząd -> 1. rząd)
    for file in range(8):
        from_square = chess.square(file, 1)

        # Zwykła promocja do przodu
        to_square = chess.square(file, 0)
        for piece in promotion_pieces:
            move = chess.Move(from_square, to_square, promotion=chess.Piece.from_symbol(piece.lower()).piece_type)
            moves.append(move.uci())

        # Bicie w lewo w trakcie promocji
        if file > 0:  # jeśli nie jest na linii 'a'
            to_square = chess.square(file - 1, 0)
            for piece in promotion_pieces:
                move = chess.Move(from_square, to_square, promotion=chess.Piece.from_symbol(piece.lower()).piece_type)
                moves.append(move.uci())

        # Bicie w prawo w trakcie promocji
        if file < 7:  # jeśli nie jest na linii 'h'
            to_square = chess.square(file + 1, 0)
            for piece in promotion_pieces:
                move = chess.Move(from_square, to_square, promotion=chess.Piece.from_symbol(piece.lower()).piece_type)
                moves.append(move.uci())

    # Dodanie ruchów roszady - krótka i długa, dla obu kolorów
    moves.append("e1g1")  # Biała krótka roszada
    moves.append("e1c1")  # Biała długa roszada
    moves.append("e8g8")  # Czarna krótka roszada
    moves.append("e8c8")  # Czarna długa roszada

    # Usuń duplikaty i zwróć listę
    return list(set(moves))

# Save the label encoder in .pkl file
def save_label_encoder(all_possible_moves):
    label_encoder = LabelEncoder()
    label_encoder.fit(all_possible_moves)

    # Save the encoder in .pkl format
    with open('label_encoder.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)
    print("Label Encoder has been saved.")


all_possible_moves = generate_all_possible_moves()
print(f"Number of possible chess moves: {len(all_possible_moves)}")

save_label_encoder(all_possible_moves)


