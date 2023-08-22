import chess
import random


def parse_uci_position(uci_position):
    board = chess.Board()
    uci_moves = uci_position.split()

    for move in uci_moves[3:]:
        chess_move = chess.Move.from_uci(move)
        board.push(chess_move)

    return board


def uci():
    board = chess.Board()

    while True:
        command = input()
        if command == "uci":
            print("id name random_agent")
            print("id author YourName")
            print("uciok")

        elif command == "isready":
            print("readyok")

        elif command == "ucinewgame":
            board = chess.Board()

        elif command.startswith("position"):
            board = parse_uci_position(command)
        elif command.startswith("go"):
            # Extract any additional parameters from the "go" command if needed
            # (e.g., time, depth, move-time, etc.)

            # Call the MCTS function to find the best move
            best_move = random.choice(list(board.legal_moves))
            print("bestmove", best_move)
        elif command == "quit":
            break

if __name__ == "__main__":
    uci()