import io
import tqdm
import chess.pgn
import numpy as np
import h5py
from eval import board_to_numpy
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("stockfish-windows-x86-64-avx2.exe")


def read_pgn(n, skip=0):
    batch_size = 25000
    with open("Caissabase_2022_12_24/database.pgn", "r", encoding="utf-8") as f:
        for _ in range(skip):
            line = f.readline()
            while line and not line.strip().startswith("[Event"):
                line = f.readline()
            if not line:
                print("end of file")
                return
        print(f"SKIPPED {skip} GAMES")
        for i in range(n):
            print(f"READ {batch_size * i} GAMES")
            h5f = h5py.File(f"data/data_{i}.h5", "w")
            game_dataset = h5f.create_dataset("games", (0, 13, 8, 8), dtype="i1", maxshape=(None, 13, 8, 8),
                                              compression="gzip")
            label_dataset = h5f.create_dataset("labels", (0,), dtype="i1", maxshape=(None,))
            batch = []
            for _ in range(batch_size):
                game_text = ""
                line = f.readline()
                while line and not line.strip().startswith("[Event"):
                    game_text += line
                    line = f.readline()
                if game_text:
                    batch.append(game_text)
                if not line:
                    print("end of file")
                    break
            if not batch:
                print("end of file")
                break

            games = []
            labels = []
            for b in tqdm.tqdm(batch):
                pgn = io.StringIO(b)
                game = chess.pgn.read_game(pgn)
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    games.append(board_to_numpy(board))
                    info = engine.analyse(board, chess.engine.Limit(depth=0))
                    value = (info['score'].wdl().pov(chess.WHITE).expectation() - 0.5) * 2
                    labels.append(value)
            games = np.array(games)
            labels = np.array(labels)

            game_dataset.resize((len(game_dataset) + len(games), 13, 8, 8))
            label_dataset.resize((len(label_dataset) + len(labels),))
            game_dataset[-len(games):] = games
            label_dataset[-len(labels):] = labels
            h5f.close()


if __name__ == "__main__":
    read_pgn(40)
    engine.quit()
