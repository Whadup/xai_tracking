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

from xai_tracking.xai import *
from xai_tracking.nn import VGG_net
# from ridgeplot import ridge_plot
torch.multiprocessing.set_sharing_strategy('file_system')
import os
PATH = os.path.dirname(__file__)

class CIFAR10_ind(CIFAR10):
    """
    Wraps the CIFAR10 dataset to also return example ids.
    """
    
    def __init__(
            self,
            *args, **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x = super().__getitem__(index)
        return (*x, index)



def main():
    example_wise = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(size=[32,32], padding=4),
         transform]
    )
    batch_size = 128

    if not example_wise:
        ghost_samples = 40
    else:
        ghost_samples = 0
    trainset = WrappedDataset(CIFAR10(root='cifar10/data', train=True,
                                            download=True, transform=train_transform))
    if example_wise:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=ClassSampler(trainset, batch_size, True, ghost_samples), num_workers=28)
    testset = WrappedDataset(CIFAR10(root='cifar10/data', train=False,
                                        download=True, transform=transform))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=8)
    # problem mit shuffle true
    # brauchen eigenen sampler eventuell?
    # so? https://discuss.pytorch.org/t/index-concept-in-torch-utils-data-dataloader/72449/6
    # https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.__getitem__
    print("loaded data...")
    CLASS_NAMES = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = VGG_net(ghost_samples=ghost_samples).to_sequential()
    # net.load_state_dict(torch.load("network10.pt"))
    net = net.to(device)

    
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.00666, weight_decay=0.0340)
    optimizer = WrappedSGD(net.parameters(), lr=0.01, history_file="/raid/pfahler/cifar10.hdf5")
    optimizer.hookup(net)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    loss = nn.CrossEntropyLoss(reduction="none")
    # loss = nn.BCEWithLogitsLoss()

    for e in range(15):
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
            optimizer.step(ids=ind, labels=labels)


        torch.save(net.state_dict(), os.path.join(PATH, "cifar10.pt"))
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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0) 
                correct += (predicted == labels).sum().item()
        test_acc = correct/total
        print("Test accuracy is: " + str(test_acc))
        scheduler.step()
        net = net.to(device)
    torch.save(net.state_dict(), os.path.join(PATH, "cifar10.pt"))
    # torch.save(net.cpu().state_dict(), "cifar_log.pt")  
    print("done with everything")

if __name__ == "__main__":
    main()

