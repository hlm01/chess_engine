from test import MCTS
import eval
import chess
import torch

def main():
    nn = eval.NeuralNetwork().cuda()
    nn.load_state_dict(torch.load("model.pt"))
    nn.eval()
    board = chess.Board()
    tree = MCTS.MCTS(board, nn, 1.5)
    print(tree.search(10000))


if __name__ == "__main__":
    main()