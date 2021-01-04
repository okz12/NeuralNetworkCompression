import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model", help="VGG Model to train",
                    type=str)
parser.add_argument("device", help="Device to train on",
                    type=str)
args = parser.parse_args()

import os
if os.getcwd().split('/')[-1] == "scripts":
    os.chdir('..')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from src.train import train
from src.vgg import VGG
torch.manual_seed(0)
batch_size = 128

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

vgg_model = args.model
model=VGG(vgg_model)
if torch.cuda.is_available():
    model.cuda()
    model.to('cuda:{}'.format(args.device))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = 30, gamma = 0.1)
train(model, trainloader, testloader, optimizer, criterion, 80, writer=None, scheduler=scheduler)
torch.save(model.state_dict(), "./models/{}.pt".format(vgg_model.lower()))