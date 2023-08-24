import math
from time import perf_counter

import chess
import numpy as np
import torch

import eval


class Node:
    def __init__(self, parent, untried, action, terminal, turn):
        self.wins = 0
        self.visits = 0
        self.parent = parent
        self.action = action
        self.untried = untried
        self.children = []
        self.terminal = terminal
        self.turn = turn


class MCTS:
    NOT_TERMINAL = -999

    def __init__(self, board, net, c):
        self.initial = board.copy()
        self.board = board.copy()
        self.net = net
        self.root = Node(None, list(board.legal_moves), None, self.NOT_TERMINAL, board.turn)
        self.c = c
        self.iters = 0

    def push_move(self, move):
        self.initial.push(move)
        for action, child in self.root.children:
            if action == move:
                self.root = child
                self.root.parent = None
                print(f"found {move} with {child.visits} visits")
                return
        self.root = Node(None, list(self.initial.legal_moves), None, self.NOT_TERMINAL, self.initial.turn)

    def search(self, budget=10000):
        print(self.root.turn)
        start = perf_counter()
        self.iters = 0
        while self.iters < budget:
            self.board = self.initial.copy()
            outcome = self.tree_policy(self.root)
            self.backup(*outcome)
            self.iters += 1
        print(f"searched {self.iters} nodes in {perf_counter() - start} seconds")
        most_visits = -1
        best = None
        for action, c in self.root.children:
            if c.visits > most_visits:
                most_visits = c.visits
                best = action
        self.push_move(best)
        return best

    def tree_policy(self, s):
        while s.terminal == self.NOT_TERMINAL:
            if s.untried:
                return self.expand(s)
            else:
                s = self.best_child(s)
        return s, s.terminal, 1

    def expand(self, s):
        batch = np.empty((len(s.untried), 13, 8, 8), dtype=np.float32)
        count = len(s.untried)
        wins = 0
        for i, a in enumerate(s.untried):
            self.board.push(a)
            terminal = self.NOT_TERMINAL
            if self.board.is_game_over():
                if self.board.outcome().winner == chess.WHITE:
                    terminal = 1
                elif self.board.outcome().winner == chess.BLACK:
                    terminal = -1
                else:
                    terminal = 0
            else:
                board_tensor = eval.board_to_numpy(self.board)
                batch[i] = board_tensor
            child = Node(s, list(self.board.legal_moves), a, terminal, not s.turn)
            if terminal != self.NOT_TERMINAL:
                child.wins = terminal if s.turn == chess.WHITE else -terminal
                child.visits = 1
                wins += terminal
            s.children.append((a, child))
            self.board.pop()

        if batch.shape[0] > 0:
            batch = torch.tensor(batch, device="cuda")
            with torch.no_grad():
                values = self.net(batch).cpu().numpy()
            for i, (_, child) in enumerate(s.children):
                if child.terminal == self.NOT_TERMINAL:
                    child.wins = values[i] if s.turn == chess.WHITE else -values[i]
                    wins += values[i]
                    child.visits = 1
        s.untried = []
        self.iters += count
        return s, wins, count

    def best_child(self, node):
        best_child_node = None
        best_ucb = -1000
        for a, child in node.children:
            # UCB calculation
            ucb = child.wins / child.visits + self.c * math.sqrt(math.log(node.visits) / child.visits)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child_node = child
        self.board.push(best_child_node.action)
        return best_child_node

    def backup(self, node, result, visits):
        while node is not self.root:
            node.visits += visits
            if node.parent.turn == chess.WHITE:
                node.wins += result
            else:
                node.wins -= result
            node = node.parent
        self.root.visits += visits


def parse_uci_position(uci_position):
    board = chess.Board()
    uci_moves = uci_position.split()

    for move in uci_moves[3:]:
        chess_move = chess.Move.from_uci(move)
        board.push(chess_move)

    return board


def uci():
    board = chess.Board()

    nn = eval.NeuralNetwork().cuda()
    nn.load_state_dict(torch.load("model.pt"))
    nn.eval()
    tree = MCTS(board, nn, 1.4)
    previous = ""
    while True:
        command = input()
        if command == "uci":
            print("id name YourChessEngine")
            print("id author YourName")
            print("uciok")

        elif command == "isready":
            print("readyok")

        elif command == "ucinewgame":
            board = chess.Board()

        elif command.startswith("position"):
            if command != previous and not command.strip().endswith("startpos"):
                tree.push_move(chess.Move.from_uci(command.split()[-1]))
                previous = command
        elif command.startswith("go"):
            # Extract any additional parameters from the "go" command if needed
            # (e.g., time, depth, move-time, etc.)
            # Call the MCTS function to find the best move
            if len(board.move_stack) < 2:
                print("bestmove", tree.search(20000))
            else:
                print(f"info score cp {round(nn.predict(tree.initial) * 100)}")
                print("bestmove", tree.search(100000))
        elif command == "quit":
            break


if __name__ == "__main__":
    uci()
