from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from utils import dataset as dataset_utils

INPUT_IMG_SIZE = 227
INPUT_IMG_CHANNELS = 3


def save(dest: str, model: nn.Module, optimizer, avg_loss: float = 0, epoch: int = 0, step: int = 0):

    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'step_num': step
        },
        dest,
    )


def train(model, data_loader: DataLoader, state_dest: str, learning_rate: float = 1e-1, epoch_num: int = 50):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device=device)

    for epoch in range(epoch_num):

        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adadelta(model.parameters())
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for _, data in enumerate(data_loader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        print("epoch {}: {}".format(epoch, running_loss))

        save(state_dest, model, optimizer, running_loss, epoch)


if __name__ == "__main__":

    motion_label = "./labels/train.csv"
    state_path = "./states/alexnet_state_v1.pt"
    test_ratio = 0.2

    data, targets, test_data, test_targets, train_file_list, _ = dataset_utils.read_dataset(motion_label)

    epoch = 50
    batch_size = 16

    train(data, targets, state_path, epoch_num=epoch, batch_size=batch_size)
