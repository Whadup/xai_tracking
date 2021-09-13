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
from .train import Model
import os
PATH = os.path.dirname(__file__)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    dataset = PygGraphPropPredDataset(name="ogbg-ppa", root="/raid/ogbg-ppa") 

    split_idx = dataset.get_idx_split() 
    trainset = dataset[split_idx["train"]]
    batch_size = 32
    ghost_samples = dataset.num_classes // 2
    print(dataset.num_classes)
    print(trainset[0])
    trainset = WrappedDataset(trainset)
    # ClassSampler(WrappedDataset(dataset[split_idx["test"]]), batch_size, True, ghost_samples)
    # trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    sampler = ClassSampler(trainset, batch_size, True, ghost_samples)
    trainloader = DataLoader(trainset, batch_sampler=sampler, num_workers=8)
    testloader = DataLoader(dataset[split_idx["test"]], batch_size=1, shuffle=True)

    # problem mit shuffle true
    # brauchen eigenen sampler eventuell?
    # so? https://discuss.pytorch.org/t/index-concept-in-torch-utils-data-dataloader/72449/6
    # https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.__getitem__
    print("loaded data...")
    CLASS_NAMES = list([f"c{i+1}" for i in range(dataset.num_classes)])
    net = Model(dataset.num_classes, ghost_samples=ghost_samples).to_sequential()
    net.load_state_dict(torch.load(os.path.join(PATH, "gnn.pt")))
    net = net.to(device)
    net = net.eval()

    pbar = tqdm.tqdm(testloader)
    
    for data in pbar:
        # get the inputs; data is a list of [inputs, labels, index of batch]
        batched_data = data.cuda()
        labels = batched_data.y.view(-1)
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        outputs = net((x, edge_index, edge_attr, batch))[0]
        outputs = global_mean_pool(outputs, batch)
        
        prob = torch.nn.functional.softmax(outputs[:1].view(-1), dim=0)
        pred = prob.argmax()
        print(prob[pred])
        print(labels[0], pred, outputs[:1].view(-1))
        if pred != labels[:1].item():
            print("missclassified")
            continue
        classes_to_plot = torch.topk(prob, 10)[1].tolist()
        print(CLASS_NAMES[labels[:1].item()], CLASS_NAMES[pred.item()])
        contributions, preactivations, cosines, dot_products, norms, l, ids = explain(net, (x, edge_index, edge_attr, batch), history_file="/raid/gnn.hdf5")
        classes, weights = class_statistics(contributions, preactivations, cosines, norms, l)
        slides = ""
        with io.StringIO() as f:
            # image(np.uint8(255 * (inputs[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5))).write_html(f, include_plotlyjs=False, full_html=False)
            slides += f"<section><h2>Input</h2>{labels[0] + 1}</section>"
        for i, layer in enumerate(list(classes.keys())[::-1]):
            dirs, sample_weights = classes[layer], weights[layer]
            dirs = {CLASS_NAMES[y]:d for y,d in sorted(dirs.items(), key=lambda x: CLASS_NAMES[x[0]]) if y >= 0 and y in classes_to_plot}
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
            # with io.StringIO() as f:
            #     fig = make_subplots(rows=2, cols=4)
            #     for j, img in enumerate(most_influential[:8]):
            #         img = img.item()
            #         img = trainset[img][0]
            #         fig.add_trace(px.imshow(np.uint8(255 * (img.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5))).data[0], row= 1 + j // 4, col = 1 +  j % 4)
            #     fig.write_html(f)
            #     slides += f"<section>{f.getvalue()}</section>"
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

