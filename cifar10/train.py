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

from xai import *
from vgg import VGG_net
from ridgeplot import ridge_plot
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 128
    ghost_samples = 40
    trainset = WrappedDataset(CIFAR10(root='cifar10/data', train=True,
                                            download=True, transform=transform))
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
    net = VGG_net().to_sequential()
    # net.load_state_dict(torch.load("network10.pt"))
    net = net.to(device)

    
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.00666, weight_decay=0.0340)
    optimizer = WrappedOptimizer(torch.optim.SGD, history_file="/raid//cifar10_tmp.hdf5")(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    loss = nn.CrossEntropyLoss(reduction="none")
    # loss = nn.BCEWithLogitsLoss()

    for e in range(35):
        print("starting epoch: " + str(e))
        correct = 0
        total = 0

        net.train()
        pbar = tqdm.tqdm(trainloader)
        for data in pbar:
            # get the inputs; data is a list of [inputs, labels, index of batch]
            inputs, labels, ind = data
            inputs, labels = inputs.to(device), labels.to(device)
            if len(labels) < 128:
                continue
            # print(CLASS_NAMES[labels[:1].item()])
            # contributions, preactivations, cosines, dot_products, norms, l = explain(net, inputs[:1])
            # classes, weights = class_statistics(contributions, preactivations, cosines, norms, l)
            # for layer in classes:
            #     dirs, sample_weights = classes[layer], weights[layer]
            #     dirs = {CLASS_NAMES[y]:d for y,d in dirs.items()}
            #     sample_weights = {CLASS_NAMES[y]:d for y,d in sample_weights.items()}
            #     plot = ridge_plot(dirs, sample_weights=sample_weights)
            #     plot.write_html(f"{layer}.html")
            # print(list([c.shape for c in contributions]))
            # print(list([c.shape for c in preactivations]))
            # print(list([c.shape for c in cosines]))
            # print(l.shape)
            # for c in cosines:
            #     print(torch.histc(c,21,c.min(),c.max()))
            # asdfs
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # print(torch.nn.functional.one_hot(labels, num_classes=10))
            # l = loss(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
            l = loss(outputs, labels)[:-ghost_samples].mean()

            _, predicted = torch.max(outputs.data, 1)
            total += labels[:-ghost_samples].size(0) 
            correct += (predicted == labels)[:-ghost_samples].sum().item()
            pbar.set_description(f"Loss: {l.item():.4f} Accuracy: {1.0 * correct / total:.4f}")
            l.backward()
            optimizer.step()
            optimizer.archive(ids=ind, labels=labels)

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

