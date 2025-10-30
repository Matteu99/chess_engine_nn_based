# ♟️ Chess Engine based on Neural-network

**Chess Engine - Based on Neural-network** is a chess engine project built on a **neural network** and **Monte Carlo Tree Search (MCTS)** inspired by AlpaZero algorithm.  
The goal of this project is to create a model capable of learning to play chess from real games and using its trained neural network to make intelligent move decisions.

---

## 🧠 Project Overview

This project combines traditional chess engine concepts with machine learning.  
The neural network is trained using real chess game data (PGN format), which must be downloaded from the internet and placed in the project directory.

The system operates in two main stages:
1. **Data preparation and neural network training**
2. **Decision-making using MCTS with the trained network**

---

## ⚙️ Project Structure

| File / Folder | Description |
|----------------|--------------|
| `generate_all_possible_moves.py` | Generates all possible chess moves according to game rules (piece movement, castling, promotions, en passant, etc.). Creates a move database used in later processing. |
| `preprocessing_chess_games.py` | Processes chess games in **PGN** format, preparing input data for the neural network (e.g., matrices representing board states and corresponding moves). |
| `networktraining.py` | Contains the `networktraining` function responsible for training the neural network model on prepared data. |
| `monte_carlo_tree_search.py` | Implements the **Monte Carlo Tree Search (MCTS)** algorithm integrated with the trained neural network to make optimal move decisions. |

---

## 🧩 Requirements

Make sure you have installed:
- Python 3.8+
- Libraries: `numpy`, `pandas`, `tensorflow` / `keras`, `chess` (or `python-chess`), `pickle`, etc.

Install dependencies with:
```bash
pip install -r requirements.txt
```
*(if the file `requirements.txt` is available)*

---

## 🚀 Usage

1. **Download chess data**
   - Download chess games in `.pgn` format (e.g., from Lichess, Chess.com, or Kaggle)
   - Place the `.pgn` files in the project directory

2. **Generate all possible moves**
   ```bash
   python generate_all_possible_moves.py
   ```

3. **Preprocess chess games**
   ```bash
   python preprocessing_chess_games.py
   ```

4. **Train the neural network**
   - Open `networktraining.py`
   - Run the function `networktraining()` (e.g., in Python shell or notebook)

5. **Run the MCTS with neural network**
   ```bash
   python monte_carlo_tree_search.py
   ```

---

## 🧾 Engineering Thesis Attachment

The chess engine project was an engineering thesis at Warsaw University of Technology. For more detailed information, feel free to contact me via email: **mati44567@gmail.com**

---

## 🧑‍💻 Author

**Mateusz [Matteu99]**

Repository: [github.com/Matteu99/chess_engine_nn_based](https://github.com/Matteu99/chess_engine_nn_based)

---


## 📈 Future Improvements

- Expand the network and dataset for better accuracy  
- Add a GUI to play against the engine  
- Integrate with external engines (e.g., Stockfish) for benchmarking  
- Optimize MCTS for higher performance
