import torchvision
from torchvision import models
from model.alexnet import AlexNet, AlexNetV0
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import transforms

from utils import dataset as dataset_utils
from utils import train as train_utils
from utils import evaluate as eval_utils


if __name__ == "__main__":

    cifar_dataset_path = "./dataset/cifar/data_batch_1"

    motion_label = "./labels/motion_15_label.csv"
    state_path = "./states/mobilenetv3_state_v6.pt"
    test_ratio = 0.2
    img_size = 227
    balance_rate = 1
    epoch = 30
    learning_rate = 1e-2
    batch_size = 128

    # transform = torch.nn.Sequential(
    #     transforms.Resize((img_size, img_size)),
    #     transforms.RandomAffine((-30, 30), (0, 0.05), (0.6, 1.2))
    # )

    transform = transforms.Compose([transforms.RandomGrayscale(), transforms.RandomAffine((-45, 45), (0.2, 0.1)), transforms.ToTensor()])

    # dataset = dataset_utils.ImageDataset(motion_label, transform)
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    print("dataset loaded")

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # model = models.mobilenet_v3_large(pretrained=True)
    model = AlexNetV0()

    train_utils.train(model, train_loader, state_path)

    eval_utils.evaluate(model, state_path, train_loader, num_classes=10)
