import chess
import torch
from torch import nn
import numpy as np

device = "cuda"


def board_to_numpy(board):
    tensor = np.zeros((13, 8, 8), dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece is not None:
            if piece.color == chess.WHITE:
                tensor[piece.piece_type - 1][i // 8][i % 8] = 1
            else:
                tensor[piece.piece_type + 5][i // 8][i % 8] = 1
    tensor[12] = 1 if board.turn == chess.WHITE else 0
    return tensor


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_channels = 256
        self.conv_stack = nn.Sequential(nn.Conv2d(13, self.num_channels, 3, padding=1),
                                        nn.BatchNorm2d(self.num_channels),
                                        nn.ReLU(),
                                        nn.Conv2d(self.num_channels, self.num_channels, 3, padding=1),
                                        nn.BatchNorm2d(self.num_channels),
                                        nn.ReLU(),
                                        nn.Conv2d(self.num_channels, self.num_channels, 3, padding=1),
                                        nn.BatchNorm2d(self.num_channels),
                                        nn.ReLU(),
                                        nn.Conv2d(self.num_channels, self.num_channels, 3, padding=1),
                                        nn.BatchNorm2d(self.num_channels),
                                        nn.ReLU())
        self.linear = nn.Linear(self.num_channels * 8 * 8, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_stack(x)
        x = x.view(-1, self.num_channels * 8 * 8)
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return torch.tanh(x)

    def predict(self, state):
        tensor = torch.tensor(board_to_numpy(state)).unsqueeze(0).to(device)
        with torch.no_grad():
            v = self(tensor).item()
            return v
