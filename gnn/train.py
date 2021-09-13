from torch_geometric.nn import MessagePassing
import torch
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

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool
import os
PATH = os.path.dirname(__file__)
### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "mean")

        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(emb_dim, 2*emb_dim),
        #     # torch.nn.BatchNorm1d(2*emb_dim), #TODO: that one is going to suck...
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(2*emb_dim, emb_dim))
        # self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.squish = torch.nn.Sequential(torch.nn.Linear(2 * emb_dim, 1, bias=False), torch.nn.Tanh())
        self.edge_encoder = torch.nn.Linear(7, emb_dim, bias=False)

    def forward(self, args):
        x, edge_index, edge_attr, batch = args
        edge_embedding = self.edge_encoder(edge_attr)

        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        #self.mlp()

        return out, edge_index, edge_attr, batch

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
    # def message(self, x_i, x_j, edge_attr):
    #     xe = x_j + edge_attr
    #     x_agg = self.squish(torch.cat([x_i, xe], dim=-1))
    #     return F.relu(torch.matmul(x_agg.unsqueeze(-1), xe.unsqueeze(-1).transpose(-1, -2)).squeeze(-1).view(x_j.shape[0], -1))

    def update(self, aggr_out):
        return aggr_out

class WeirdEmbedding(torch.nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data
        x = torch.zeros_like(batch)
        return super().forward(x), *batched_data[1:]

class WeirdLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data
        return super().forward(x), *batched_data[1:]


class WeirdReLU(torch.nn.ReLU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data
        return super().forward(x), *batched_data[1:]

class WeirdGraphBatchNorm(torch.nn.Module):
    def __init__(self, *args, ghost_samples=0, **kwargs):
        super().__init__()
        self.ghost_samples = ghost_samples
        self.bn = nn.BatchNorm1d(*args, **kwargs)
        #TODO do running mean myself...fuck
    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data
        num_examples = batch.max()
        ghost_mask = batch > (num_examples - self.ghost_samples)
        if self.training:
            sample = x[torch.logical_not(ghost_mask)]
            ghost_sample = x[ghost_mask]
            y = self.bn(ghost_sample)
            mean = ghost_sample.mean(0, keepdim=True)#.detach() # experimental detach to disconnect ghost sample from model and remaining batch
            var =  ghost_sample.var(0, keepdim=True, unbiased=False)#.detach() # experimental detach to disconnect ghost sample from model and remaining batch
            sample_mean = sample.mean(0, keepdim=True)
            sample_var =  sample.var(0, keepdim=True, unbiased=False)#.detach() # experimental detach to disconnect ghost sample from model and remaining batch
            mean = mean + (sample_mean - mean) / (1 + ghost_sample.shape[0])
            # var = var + ((sample_mean - mean) ** 2 - var) / ghost_sample.shape[0]
            var = (ghost_sample.shape[0]) * var + sample_var + (sample_mean - mean) ** 2 * ghost_sample.shape[0]  / (ghost_sample.shape[0] + 1)
            var = var / (ghost_sample.shape[0] + 1)
            # print(mean[:,0].item(), self.bn.running_mean[0].item())
            tmp = (x - mean) / torch.sqrt(var + 1e-5)
            if self.bn.bias is not None:
                tmp * self.bn.weight + self.bn.bias
            return tmp, edge_index, edge_attr, batch
        else:
            return self.bn(x), edge_index, edge_attr, batch

class Model(torch.nn.Module):
    def __init__(self, num_classes=13, ghost_samples=1, emb_dim=300, num_layers=5):
        super().__init__()
        self.node_embedding = WeirdEmbedding(1, emb_dim)
        self.conv = torch.nn.Sequential(
            *sum([[
                GINConv(emb_dim),
                WeirdLinear(emb_dim * 1, 2 * emb_dim, bias=False),
                WeirdGraphBatchNorm(2 * emb_dim, ghost_samples=ghost_samples),
                WeirdReLU(),
                WeirdLinear(emb_dim * 2, emb_dim, bias=False),
                ] for i in range(num_layers)], [])
        )
        self.classifier = WeirdLinear(emb_dim, num_classes, bias=False)
    def to_sequential(self):
        return torch.nn.Sequential(self.node_embedding, *self.conv, self.classifier)
    def forward(self, batched_data):
        
        x = self.node_embedding(x, edge_index, edge_attr, batch)
        return self.conv(x, edge_index, edge_attr, batch)


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
    testloader = DataLoader(dataset[split_idx["test"]], batch_size=512, shuffle=False)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=ClassSampler(trainset, batch_size, True, ghost_samples), num_workers=28)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                         shuffle=False, num_workers=8)
    # problem mit shuffle true
    # brauchen eigenen sampler eventuell?
    # so? https://discuss.pytorch.org/t/index-concept-in-torch-utils-data-dataloader/72449/6
    # https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.__getitem__
    print("loaded data...")
    # CLASS_NAMES = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = Model(dataset.num_classes, ghost_samples=ghost_samples).to_sequential()
    # net.load_state_dict(torch.load("network10.pt"))
    net = net.to(device)

    
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    optimizer = WrappedOptimizer(torch.optim.SGD, history_file="/raid/gnn.hdf5")(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    # optimizer = WrappedOptimizer(torch.optim.SGD)(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 40, 120], gamma=0.1)
    loss = nn.CrossEntropyLoss(reduction="none")
    # loss = nn.BCEWithLogitsLoss()
    # weight = 1.0

    normalizer = sum([len(y) for x,y in sampler.classes_.items()]) / len(sampler.classes_)

    for e in range(50):
        # if e in [10,50,100]:
        #     weight *= 0.1
        print("starting epoch: " + str(e))
        correct = 0
        total = 0
        running_loss = 0
        net.train()
        pbar = tqdm.tqdm(trainloader)
        for data in pbar:
            # get the inputs; data is a list of [inputs, labels, index of batch]
            data, ids = data
            batched_data = data.cuda()
            # inputs, labels, ind = data
            # inputs, labels = inputs.to(device), labels.to(device)
            labels = batched_data.y.view(-1)
            # print(labels, labels.shape)
            if len(labels) < batch_size:
                continue
            one_class = labels[0].item()
            optimizer.zero_grad()
            # forward + backward + optimize
            x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
            # print("X", x, batched_data.x)
            outputs = net((x, edge_index, edge_attr, batch))[0]
            outputs = global_mean_pool(outputs, batch)
                        
            l = loss(outputs, labels)[:-ghost_samples].mean()
            # l = l * len(sampler.classes_[one_class]) / normalizer
            # print(len(sampler.classes_[one_class]) / normalizer)
            # l = loss(outputs, labels).mean()

            _, predicted = torch.max(outputs.data, 1)
            total += labels[:-ghost_samples].size(0) 
            correct += (predicted == labels)[:-ghost_samples].sum().item()
            running_loss += l.item() * labels[:-ghost_samples].size(0) 
            # total += labels.size(0) 
            # correct += (predicted == labels).sum().item()
            
            pbar.set_description(f"Loss: {running_loss / total :.4f} Accuracy: {1.0 * correct / total:.4f}\tgo do something else, stop staring")
            # l *= weight
            l.backward()
            optimizer.step()
            optimizer.archive(ids=ids, labels=labels)

        torch.save(net.state_dict(), os.path.join(PATH, "gnn.pt"))

    

        # torch.save(optimizer, "optimizer.pt")
        print("Training loss is: " + str(running_loss / total))
        train_acc = correct / total
        print("Training accuracy is: " + str(train_acc))
    
        correct = 0
        total = 0
        net.eval()
        if e % 5 == 4:
            with torch.no_grad():
                for data in testloader:
                    batched_data = data.cuda()
                    labels = batched_data.y.view(-1)
                    x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
                    outputs = net((x, edge_index, edge_attr, batch))[0]
                    outputs = global_mean_pool(outputs, batch)

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0) 
                    correct += (predicted == labels).sum().item()
            test_acc = correct/total
            print("Test accuracy is: " + str(test_acc))
        scheduler.step()
        net = net.to(device)
    # torch.save(net.cpu().state_dict(), "cifar_log.pt")   
    torch.save(net.state_dict(), os.path.join(PATH, "gnn.pt"))
    print("done with everything")

if __name__ == "__main__":
    main()
