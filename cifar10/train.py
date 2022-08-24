# Imports
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import tqdm
import zipfile
import re
from collections import defaultdict
from math import sqrt
import tqdm
import pickle
from typing import Tuple, Any
from torchinfo import summary

from xai_tracking.xai import *
from xai_tracking.nn import VGG_net, resnet50, resnet18
# from ridgeplot import ridge_plot
# torch.multiprocessing.set_sharing_strategy('file_system')
import multiprocessing
import os
PATH = os.path.dirname(__file__)

def main():
    example_wise = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]
    )
    train_transform = transforms.Compose(
        [transforms.RandomCrop(size=[32,32], padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    batch_size = 128

    if not example_wise:
        ghost_samples = 40
    else:
        ghost_samples = 0
    trainset = WrappedDataset(CIFAR10(root='cifar10/data', train=True,
                                            download=True, transform=train_transform))
    if example_wise:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=ClassSampler(trainset, batch_size, True, ghost_samples), num_workers=10, persistent_workers=True)
    testset = WrappedDataset(CIFAR10(root='cifar10/data', train=False,
                                        download=True, transform=transform))
    testloader = torch.utils.data.DataLoader(testset, batch_size=2 * batch_size,
                                            shuffle=False)
    # problem mit shuffle true
    # brauchen eigenen sampler eventuell?
    # so? https://discuss.pytorch.org/t/index-concept-in-torch-utils-data-dataloader/72449/6
    # https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.__getitem__
    print("loaded data...")
    CLASS_NAMES = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = VGG_net(ghost_samples=ghost_samples).to_sequential()
    # net = resnet18(num_classes=10).to_sequential()
    # net.load_state_dict(torch.load("network10.pt"))
    # summary(net, input_size=(batch_size, 3, 32, 32))
    net = net.to(device)
    
    if example_wise:
        optimizer = WrappedSGD(net.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0, history_file="/raid/pfahler/cifar10_history")
        optimizer.hookup(net)
    else:
        # optimizer = WrappedOptimizer(torch.optim.SGD, history_file="/raid/pfahler/tmp/cifar10_batchwise_history.hdf5")
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001) # momentum=0.9, weight_decay=5e-4 

    # optimizer = WrappedSGD(net.parameters(), lr=0.1, history_file="/raid/pfahler/cifar10_history")

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    loss = nn.CrossEntropyLoss(reduction="none")
    # loss = nn.BCEWithLogitsLoss()

    for e in range(200):
        print("starting epoch: " + str(e))
        correct = 0
        total = 0
        total_loss = 0
        net.train()
        pbar = tqdm.tqdm(trainloader)

        for data in pbar:
            # get the inputs; data is a list of [inputs, labels, index of batch]
            inputs, labels, ind = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            if ghost_samples > 0:
                l = loss(outputs, labels)[:-ghost_samples].mean()
                total += labels[:-ghost_samples].size(0) 
                correct += (predicted == labels)[:-ghost_samples].sum().item()
                total_loss += l.item() * labels[:-ghost_samples].size(0) 

            else:
                l = loss(outputs, labels).mean()
                total += labels.size(0) 
                correct += (predicted == labels).sum().item()
                total_loss += l.item() * labels.size(0) 
            pbar.set_description(f"Loss: {total_loss / total:.4f} Accuracy: {1.0 * correct / total:.4f}")
            l.backward()
            if example_wise:
                optimizer.step(ids=ind, labels=labels)
            else:
                # optimizer.step(ids=ind[:-ghost_samples], labels=labels[:-ghost_samples])
                optimizer.step()


        if example_wise:
            torch.save(net.state_dict(), os.path.join("/raid/pfahler/cifar10_checkpoints", f"cifar10_{e}.pt"))
        # torch.save(net.state_dict(), "network.pt")
        # torch.save(optimizer, "optimizer.pt")
        print("Training loss is: " + str(l.item()))
        train_acc = correct / total
        print("Training accuracy is: " + str(train_acc))
    
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for data in testloader:
                inputs, labels, ind = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                predicted = outputs.argmax(1)
                total += labels.size(0) 
                correct += (predicted == labels).sum().item()
        test_acc = correct / total
        print("Test accuracy is: " + str(test_acc))
        scheduler.step()
        net = net.to(device)
    if example_wise:
        torch.save(net.state_dict(), os.path.join(PATH, "cifar10.pt"))
    else:
        torch.save(net.state_dict(), os.path.join(PATH, "batchwise_cifar10.pt"))

    # torch.save(net.cpu().state_dict(), "cifar_log.pt")  
    optimizer.done()
    print("done with everything")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()

