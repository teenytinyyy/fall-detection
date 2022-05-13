import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from model.alexnet import AlexNet


def build_confusion_matrix(n_c: int = 10) -> np.ndarray:
    confusion_mat = np.zeros((n_c, n_c))
    return confusion_mat


def load_state(src: str, model: nn.Module, optimizer=None):

    checkpoint = torch.load(src, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def evaluate(model, state_path: str, data_loader: DataLoader, num_classes: int = 2):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    load_state(state_path, model)

    model = model.to(device=device)

    confusion_mat = build_confusion_matrix(num_classes)
    for _, data in enumerate(data_loader, 0):
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        scores = model(inputs)

        predictions = torch.topk(scores, k=1).indices.squeeze()

        idx_pairs = np.array([predictions.tolist(), labels.tolist()]).T.tolist()

        for idx_pair in idx_pairs:
            confusion_mat[tuple(idx_pair)] += 1

    print(confusion_mat)


if __name__ == '__main__':

    state_path = "states/alexnet_state_v0.pt"

    label_path = "dataset/test/test.csv"
