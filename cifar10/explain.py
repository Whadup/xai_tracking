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
from xai_tracking.xai import *
from xai_tracking.nn import VGG_net, GHOST_SAMPLES
from xai_tracking.ridgeplot import ridge_plot, image
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import os
PATH = os.path.dirname(__file__)



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    raw_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    batch_size = 1
    trainset = WrappedDataset(CIFAR10(root='cifar10/data', train=True,
                                            download=True, transform=raw_transform))
    dataset_labels = np.array([y for _, y, _ in trainset])
    testset = WrappedDataset(CIFAR10(root='cifar10/data', train=False,
                                         download=True, transform=transform))
    
    raw_testset = WrappedDataset(CIFAR10(root='cifar10/data', train=False,
                                         download=True, transform=raw_transform))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=8)
    # problem mit shuffle true
    # brauchen eigenen sampler eventuell?
    # so? https://discuss.pytorch.org/t/index-concept-in-torch-utils-data-dataloader/72449/6
    # https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.__getitem__
    print("loaded data...")
    CLASS_NAMES = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    example_wise = True

    net = VGG_net(ghost_samples=0 if example_wise else 1).to_sequential()
    if example_wise:
        net.load_state_dict(torch.load(os.path.join(PATH, "cifar10.pt")))
    else:
        net.load_state_dict(torch.load(os.path.join(PATH, "batchwise_cifar10.pt")))
    net = net.to(device)
    net = net.eval()

    pbar = tqdm.tqdm(testloader)
    
    test_examples_indices = [2]
    test_examples_batch = torch.stack([testset[i][0] for i in test_examples_indices]).to(device)
    test_examples_predicted_probs, test_examples_predicted_labels = torch.max(F.softmax(net(test_examples_batch), dim=1), dim=1)
    test_examples_true_labels = torch.Tensor([testset[i][1] for i in test_examples_indices]).long().to(device)

    data = (test_examples_batch, test_examples_true_labels, torch.tensor(test_examples_indices))

    while True:
        # get the inputs; data is a list of [inputs, labels, index of batch]
        inputs, labels, ind = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        prob = torch.nn.functional.softmax(net(inputs[:1]).view(-1), dim=0)
        pred = prob.argmax()
        print(prob[pred])
        # if CLASS_NAMES[pred.item()] != "truck":
        #     continue
        if pred != labels[:1].item():
            print("missclassified")
            continue

        print(CLASS_NAMES[labels[:1].item()], CLASS_NAMES[pred.item()])
        contributions, preactivations, cosines, dot_products, norms, l, ids = explain(net, inputs[:1], history_file="/raid/pfahler/cifar10_batchwise_history.hdf5", history_folder="/raid/pfahler/tmp", dataset_labels=dataset_labels, example_wise=example_wise)
        # pickle.dump((contributions, preactivations, cosines, dot_products, norms, l, ids), open("tmp.pickle", "wb"))
        # (contributions, preactivations, cosines, dot_products, norms, l, ids) = pickle.load(open("tmp.pickle", "rb"))
        classes, weights = class_statistics(contributions, preactivations, cosines, norms, l)
        slides = ""
        raw_inputs, _, _ = raw_testset[ind[0]]
        print(raw_inputs)
        for p in preactivations:
            print(p.shape)
            print(torch.linalg.norm(p))
        with io.StringIO() as f:
            image(np.uint8(255 * (raw_inputs.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5))).write_html(f, include_plotlyjs=False, full_html=False)
            slides += f"<section><h2>Input</h2>{f.getvalue()}</section>"
        for i, layer in enumerate(list(classes.keys())):
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
            print((cosines[i] * norms[i]).sum(), torch.linalg.norm(preactivations[i]))
            for ii, norm, dot in zip(ids, norms[i], dots):
                most_influential[ii] += norm * dot#.clamp(min=0)
            print(torch.topk(most_influential, 100), dataset_labels[torch.topk(most_influential, 100).indices])
            most_influential = torch.topk(most_influential, 100).indices[:64]
            with io.StringIO() as f:
                fig = make_subplots(rows=8, cols=8)
                fig.add_trace(px.imshow(np.uint8(255 * (raw_inputs.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5))).data[0], row=1, col=1)
                for j, img in enumerate(most_influential[:63]):
                    j += 1
                    img = img.item()
                    img = trainset[img][0]
                    fig.add_trace(px.imshow(np.uint8(255 * (img.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5))).data[0], row= 1 + j // 8, col = 1 +  j % 8)
                fig.write_html(f)
                slides += f"<section>{f.getvalue()}</section>"
            slides += "</section>"

        with open(os.path.join(PATH, f"plots_{example_wise}.html"), "w") as output:
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

