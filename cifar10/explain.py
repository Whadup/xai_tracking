# Imports
import io
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
from plotly.subplots import make_subplots
import plotly.express as px
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool
from xai import *
from vgg import VGG_net, GHOST_SAMPLES
from ridgeplot import ridge_plot, image
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import os
PATH = os.path.dirname(__file__)

class ClassSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size - GHOST_SAMPLES
        self.drop_last = drop_last
        self.generator = None

    def __iter__(self):
        self.classes = {}
        for i, (d, y, j) in enumerate(self.data_source):
            if y not in self.classes:
                self.classes[y] = []
            self.classes[y].append(i)
        for y in self.classes:
            self.classes[y] = torch.LongTensor(self.classes[y])[torch.randperm(len(self.classes[y]))]
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        while len(self.classes):
            cls = list(self.classes.keys())[torch.randint(high=len(self.classes), size=(1,)).item()]
            batch = self.classes[cls][:self.batch_size].tolist()
            self.classes[cls] = self.classes[cls][self.batch_size:]
            # if self.drop_last and len(self.classes[cls]) < self.batch_size:
            #     del self.classes[cls]
            for y in self.classes:
                for x in self.classes[y][:GHOST_SAMPLES//10]:
                    batch.append(x.item())
                self.classes[y] = self.classes[y][GHOST_SAMPLES//10:]
            self.classes = {x:y for x,y in self.classes.items() if len(y) >= (GHOST_SAMPLES//10 + self.batch_size)}
                # if self.drop_last and len(self.classes[y]) < self.batch_size:
                #     del self.classes[y]
            yield batch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 1
    trainset = WrappedDataset(CIFAR10(root='cifar10/data', train=True,
                                            download=True, transform=transform))
    testset = WrappedDataset(CIFAR10(root='cifar10/data', train=False,
                                         download=True, transform=transform))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=8)
    # problem mit shuffle true
    # brauchen eigenen sampler eventuell?
    # so? https://discuss.pytorch.org/t/index-concept-in-torch-utils-data-dataloader/72449/6
    # https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.__getitem__
    print("loaded data...")
    CLASS_NAMES = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = VGG_net().to_sequential()
    net.load_state_dict(torch.load(os.path.join(PATH, "cifar10.pt")))
    net = net.to(device)
    net = net.eval()

    pbar = tqdm.tqdm(testloader)
    
    for data in pbar:
        # get the inputs; data is a list of [inputs, labels, index of batch]
        inputs, labels, ind = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        prob = torch.nn.functional.softmax(net(inputs[:1]).view(-1), dim=0)
        pred = prob.argmax()
        print(prob[pred])
        if CLASS_NAMES[pred.item()] != "frog":
            continue
        if pred != labels[:1].item():
            print("missclassified")
            continue

        print(CLASS_NAMES[labels[:1].item()], CLASS_NAMES[pred.item()])
        contributions, preactivations, cosines, dot_products, norms, l, ids = explain(net, inputs[:1], history_file="/raid/cifar10.hdf5")
        classes, weights = class_statistics(contributions, preactivations, cosines, norms, l)
        slides = ""
        with io.StringIO() as f:
            image(np.uint8(255 * (inputs[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5))).write_html(f, include_plotlyjs=False, full_html=False)
            slides += f"<section><h2>Input</h2>{f.getvalue()}</section>"
        for i, layer in enumerate(list(classes.keys())[::-1]):
            dirs, sample_weights = classes[layer], weights[layer]
            dirs = {CLASS_NAMES[y]:d for y,d in dirs.items() if y >= 0}
            sample_weights = {CLASS_NAMES[y]:d for y,d in sample_weights.items()}
            slides += "<section>"
            plot = ridge_plot(dirs, sample_weights=sample_weights, highlight=CLASS_NAMES[pred.item()])
            with io.StringIO() as f:
                plot.write_html(f, include_plotlyjs=False, full_html=False)
                slides += f"<section><h2>{layer}</h2>{f.getvalue()}</section>"
            dots = cosines[i]
            most_influential = torch.zeros(len(trainset))#ids[dots.argmax()]
            for ii, dot in zip(ids, dots):
                most_influential[ii] += dot.clamp(min=0)
            most_influential = torch.topk(most_influential, 8).indices
            with io.StringIO() as f:
                fig = make_subplots(rows=2, cols=4)
                fig.add_trace(px.imshow(np.uint8(255 * (inputs[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5))).data[0], row=1, col=1)
                for j, img in enumerate(most_influential[:7]):
                    j += 1
                    img = img.item()
                    img = trainset[img][0]
                    fig.add_trace(px.imshow(np.uint8(255 * (img.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5))).data[0], row= 1 + j // 4, col = 1 +  j % 4)
                fig.write_html(f)
                slides += f"<section>{f.getvalue()}</section>"
            slides += "</section>"

        with open(os.path.join(PATH, "plots.html"), "w") as output:
            output.write(open("static/header.html", "r").read().format(slides=slides))
        # print(list([c.shape for c in contributions]))
        print(list([c.shape for c in preactivations]))
        print(list([c.shape for c in cosines]))
        print(l.shape)
        # for c in cosines:
        #     print(torch.histc(c,21,c.min(),c.max()))
        return

if __name__ == "__main__":
    main()

