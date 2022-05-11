import torch
import torch.nn as nn
import torch.optim as optim
from utils import image_func
import numpy as np
from random import random
from sklearn.utils import shuffle



__all__ = ["AlexNet", "AlexNet_Weights", "alexnet"]


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(file_path: str, learning_rate: float, epoch_num: int, batch_size: int):

    # training with either cpu or cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AlexNet()  # to compile the model
    # to send the model for training on either cuda or cpu
    model = model.to(device=device)

    # Loss and optimizer
    # learning_rate = 1e-4 #I picked this because it seems to be the most used by experts
    load_model = True
    criterion = nn.CrossEntropyLoss()
    # Adam seems to be the most popular for deep learning
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    data = image_func.imread_list(file_path)
    targets = np.zeros((len(data)))

    for epoch in range(epoch_num):  # I decided to train the model for 50 epochs
        loss_ep = 0
        data, targets = shuffle(data, targets, random_state=0)
        train_data = np.array(data)
        data = torch.from_numpy(train_data)
        train_targets = np.array(targets)
        targets = torch.from_numpy(train_targets)
        data = data.to(device=device)
        targets = targets.to(device=device)

        for i in range(len(data)):
            data[i] = image_func.resize(data[i], 227, 3)

        for batch_idx, (data, targets) in range(0, len(data), batch_size):
            # Forward Pass
            optimizer.zero_grad()
            batch_data = data[batch_idx:batch_idx+batch_size]
            batch_targets = targets[batch_idx:batch_idx+batch_size]
            scores = model(batch_data)
            loss = criterion(scores, batch_targets)
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
        print(f"Loss in epoch {epoch} :::: {loss_ep/len(data)}")

if __name__ == "__main__":
    file_path = ""
    train(file_path, 0.001, 10, 8)


