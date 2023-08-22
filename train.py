from time import perf_counter

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import random
import eval

device = "cuda"


class ChessDataset(Dataset):
    def __init__(self, index):
        with h5py.File(f"data/data_{index}.h5", "r") as f:
            self.games = np.array(f["games"])
            self.labels = np.array(f["labels"])

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        return self.games[idx], self.labels[idx]


class ValidationDataset(Dataset):
    def __init__(self, index):
        with h5py.File(f"data/data_{index}.h5", "r") as f:
            self.games = np.array(f["games"])[:2000]
            self.labels = np.array(f["labels"])[:2000]

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        return self.games[idx], self.labels[idx]


def train(data):
    loader = DataLoader(data, batch_size=256, shuffle=True, drop_last=True)
    valid = DataLoader(ValidationDataset(0), batch_size=256, shuffle=True, drop_last=True)
    model = eval.NeuralNetwork().cuda()
    model.load_state_dict(torch.load("model.pt"))
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    start = perf_counter()
    for epoch in range(1):
        for idx, (games, labels) in enumerate(loader):
            games = games.float().cuda()
            labels = labels.float().cuda()
            optimizer.zero_grad()
            output = model(games).squeeze()
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            if idx % 200 == 0:
                time = perf_counter()
                print(f"epoch {epoch} batch {idx} loss {loss.item()} time {time - start}")
                start = time
        torch.save(model.state_dict(), "model.pt")

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for idx, (games, labels) in enumerate(valid):
                games = games.float().cuda()
                labels = labels.float().cuda()
                output = model(games).squeeze()
                loss = loss_fn(output, labels)
                total_loss += loss.item()
            print(f"VALIDATION LOSS {total_loss / len(valid)}")
        model.train()


if __name__ == "__main__":
    batches = list(range(1,40))
    random.shuffle(batches)
    for i,c in enumerate(batches):
        print(f"TRAINING ON BATCH {i} data {c}")  # 7
        dataset = ChessDataset(i)
        train(dataset)
