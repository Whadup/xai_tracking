import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
except:
    pass
import tqdm
import h5py
import zipfile
import numpy as np
import itertools
import io
from copy import deepcopy
import torch_geometric
class ClassSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size, drop_last, ghost_samples, classes_per_batch=1):
        self.data_source = data_source
        self.batch_size = batch_size - ghost_samples
        self.ghost_samples = ghost_samples
        self.drop_last = drop_last
        self.generator = None
        self.classes_per_batch = classes_per_batch
        self.classes = {}
        self.random_stuff = []

        for i, d_y_j in enumerate(self.data_source):
            j = d_y_j[-1]
            if len(d_y_j) > 2:
                y = d_y_j[1]
            else:
                y = d_y_j[0].y #torch geometric
            if torch.is_tensor(y):
                y = y.item()
            self.random_stuff.append(y)
            if y not in self.classes:
                self.classes[y] = []
            self.classes[y].append(i)
        num_classes = len(self.classes)
        for y in self.classes:
            self.classes[y] = torch.LongTensor(self.classes[y])[torch.randperm(len(self.classes[y]))]
        self.classes_ = deepcopy(self.classes)
        self.random_stuff = torch.LongTensor(self.random_stuff)
        print({x:len(y) for x,y in self.classes_.items()})
        self.class_probs = np.array([len(self.classes_[x]) for x in sorted(list(self.classes_.keys()))], dtype=np.float64)
        self.class_probs /= self.class_probs.sum()
    def __iter__(self):
        self.classes = deepcopy(self.classes_)
        num_classes = len(self.classes)
        for y in self.classes:
            self.classes[y] = self.classes[y][torch.randperm(len(self.classes[y]))]#[:600]
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        while len(self.classes):
            batch = []
            for i in range(self.classes_per_batch):
                # cls = list(self.classes.keys())[torch.randint(high=len(self.classes), size=(1,)).item()]
                while True:
                    cls = np.random.choice(len(self.classes_), p=self.class_probs)
                    if cls in self.classes:
                        break
                batch += self.classes[cls][:self.batch_size // self.classes_per_batch].tolist()
                self.classes[cls] = self.classes[cls][self.batch_size // self.classes_per_batch:]
                if len(self.classes[cls]) < self.batch_size:
                    del self.classes[cls]
                if not len(self.classes):
                    break
            if self.ghost_samples // num_classes > 0:
                for y in self.classes:
                    amount = self.ghost_samples // num_classes
                    for x in torch.randint(high=len(self.classes[y]), size=(amount,)):
                        batch.append(x.item())
                    # for x in self.classes[y][:self.ghost_samples // num_classes]:
                    #     batch.append(x.item())
                    # self.classes[y] = self.classes[y][self.ghost_samples // num_classes:]
                self.classes = {x:y for x,y in self.classes.items() if len(y) > 0}
            else:
                for i in range(self.ghost_samples):
                    if len(self.classes) == 0:
                        break
                    # Sample class according to prior probabilities
                    while True:
                        cls = self.random_stuff[torch.randint(high=len(self.random_stuff), size=(1,))].item()
                        if cls in self.classes_:
                            break
                    # cls = list(self.classes.keys())[torch.randint(high=len(self.classes), size=(1,)).item()]
                    
                    # sample example from the class using all of the dataset (classes_ instead of classes)
                    for x in torch.randint(high=len(self.classes_[cls]), size=(1,)):
                        batch.append(self.classes_[cls][x].item())
                    # self.classes[cls] = self.classes[cls][1:]
                    # if len(self.classes[cls]) == 0:
                    #     del self.classes[cls]
                # if self.drop_last and len(self.classes[y]) < self.batch_size:
                #     del self.classes[y]
            # print(batch)
            yield batch

class WrappedDataset(torch.utils.data.Dataset):

    def __init__(self, ds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x = self.ds[index]
        if isinstance(x, torch_geometric.data.Data):
            return x, (index if torch.is_tensor(index) else torch.tensor(index))
        if torch.is_tensor(index):
            return (*x, index)
        return (*x, torch.tensor(index))

class WrappedOptimizer(torch.optim.Optimizer):
    def __init__(self, base_cls, history_file="/raid/history.hdf5"):
        self.internal_optimizer_cls = base_cls
        self.history_file = history_file
    def __call__(self, *args, **kwargs):
        history_file = self.history_file
        class _InternalOptimizer(self.internal_optimizer_cls):
            """
            Optimizer that logs all weight changes with one additional call to `archive(ids)` inserted after `step()`.
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_saver = {}
                tmp = {}
                for i,group in enumerate(self.param_groups):
                    for j,p in enumerate(group['params']):
                        self.weight_saver[p] = p.data.detach().clone()
                        tmp[f"group{i}.param{j}"] = p.data.detach().cpu().numpy()
                
                self.history =  h5py.File(history_file, "w")
                grp = self.history.create_group("iteration_0")
                for key in tmp:
                    ds = grp.create_dataset(key, data=tmp[key], compression="lzf")
                self.counter = 1

            # def __del__(self):
            #     self.history.close()

            @torch.no_grad()
            def step(self, *args):
                for group in self.param_groups:
                    for p in group['params']:
                        self.weight_saver[p].copy_(p.data)
                super().step(*args)

            @torch.no_grad()
            def archive(self, ids=None, labels=None):
                tmp = {}
                grp = self.history.create_group(f"iteration_{self.counter}")
                for i, group in enumerate(self.param_groups):
                    for j,p in enumerate(group['params']):
                        key = f"group{i}.param{j}"
                        ds = grp.create_dataset(key, data=torch.sub(p.data.detach(), self.weight_saver[p]).detach().cpu().numpy())#, compression="lzf")
                grp.create_dataset("ids", data=ids.detach().cpu().numpy())
                grp.create_dataset("labels", data=labels.detach().cpu().numpy())
                # with self.history.open(f"{self.counter}.pt", "w") as f:
                #     torch.save((tmp, ids, labels), f)
                self.counter += 1

        return _InternalOptimizer(*args, **kwargs)


class History(torch.utils.data.Dataset):
    def __init__(self, history_file="/raid/history.hdf5"):
        super().__init__()
        self.history_file = history_file
        self.history  = h5py.File(history_file, "r", rdcc_nbytes=1024**3)
        self.l = len(self.history.keys()) - 1
        self.workers = {}
        # self.history = {"default": zipfile.ZipFile(self.history_file, "r")}
    def reopen(self):
        self.history= h5py.File(self.history_file, "r")
        return self
    def __len__(self):
        return self.l
        # return (len(self.history["default"].namelist()) - 1)
    def __getitem__(self, ii):
        i = ii + 1
        wifo = torch.utils.data.get_worker_info()
        history = self.history
        # if wifo is not None:
        #     key = wifo.id
        #     if key not in self.workers:
        #         self.workers[key] = h5py.File(self.history_file, "r", swmr=False)
        #     history = self.workers[key]

        # with self.history[key].open(self.history["default"].namelist()[i], "r") as f:
        grp = history[f"iteration_{i}"]
        x = {}
        for key in grp:
            if key not in ("ids", "labels"):
                x[key] = grp[key][...]
        # print(x)
                # x[key] = torch.tensor(grp[key])
        if "ids" in grp:
            return x, grp["ids"][...], grp["labels"][...]
            return x, torch.tensor(grp["ids"]), torch.tensor(grp["labels"])
        else:
            return x, None, None
            # x = torch.load(f, map_location="cpu")
            # if len(x[1]) < 128:
            #     t = torch.zeros(128, dtype=x[1].dtype)
            #     t2 = torch.zeros(128, dtype=x[2].dtype)
            #     t[:len(x[1])] = x[1]
            #     t2[:len(x[2])] = x[2]
            #     return x[0], t, t2
            # return x

def block_diagonal(*arrs):
        bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
        if bad_args:
            raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args)
        shapes = torch.tensor([a.shape for a in arrs])
        i = []
        v = []
        r, c = 0, 0
        for k, (rr, cc) in enumerate(shapes):
            # TODO: Use arange(), repeat() and tile()
            first_index = torch.arange(r, r + rr, device=arrs[0].device)
            second_index = torch.arange(c, c + cc, device=arrs[0].device)
            index = torch.stack((first_index.tile((cc,1)).transpose(0,1).flatten(), second_index.repeat(rr)), dim=0)
            i += [index]
            v += [arrs[k].flatten()]
            r += rr
            c += cc
        out_shape = torch.sum(shapes, dim=0).tolist()

        if arrs[0].device == "cpu":
            out = torch.sparse.DoubleTensor(torch.cat(i, dim=1), torch.cat(v), out_shape)
        else:
            out = torch.cuda.sparse.DoubleTensor(torch.cat(i, dim=1).to(arrs[0].device), torch.cat(v), out_shape)
        return out

class SequentialWrapper(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.sequence = sequential
        self.lookup = {}
        for i,(n,p) in enumerate(self.sequence.named_parameters()):
            self.lookup[p] = f"group0.param{i}"
        

    def forward(self, x, weights_batch, weights_batch_size=None):

        def _get_first_weight(weights_batch, module):
            for _, p in enumerate(module.parameters()):
                break
            key = self.lookup[p]
            w = weights_batch[key]
            return w

        contributions = []
        preactivations = []
        for module in self.sequence:
            if isinstance(module, nn.Conv2d):
                w = _get_first_weight(weights_batch, module)
                contribution = F.conv2d(
                    x,
                    w.reshape(-1, *w.shape[2:]),
                    None,
                    module.stride,
                    module.padding,
                    module.dilation,
                    1
                )
                contribution = contribution.reshape(
                    contribution.shape[0], weights_batch_size, -1, *contribution.shape[2:]
                )
                contributions.append(contribution)
                x = module(x)
                preactivations.append(x)
            elif "WeirdLinear" in type(module).__name__: 
                x_data, edge_index, edge_attr, batch = x
                w = _get_first_weight(weights_batch, module)
                ww = w.reshape(-1, *w.shape[2:])
                contribution = torch.nn.functional.linear(x_data, ww)
                x = module(x)
                # do a fancy reshape
                # print(x_data.shape, contribution.shape)
                contribution = contribution.reshape(
                    1, weights_batch_size, -1, *contribution.shape[2:]
                )
                # print(contribution.shape)
                # asdfas


                contributions.append(contribution)
                preactivations.append(x[0].reshape(1, -1))
            elif "GINConv" in type(module).__name__:
                named_parameters = {x:y for x,y in module.named_parameters()}
                x_data, edge_index, edge_attr, batch = x
                w = weights_batch[self.lookup[named_parameters["edge_encoder.weight"]]]
                ww = w.reshape(-1, *w.shape[2:])
                contribution = torch.nn.functional.linear(edge_attr, ww)
                contribution = contribution.reshape(
                    1, weights_batch_size, -1, *contribution.shape[2:]
                )
                preac = module.edge_encoder(edge_attr)
                x = module(x)
                contributions.append(contribution)
                preactivations.append(preac.reshape(1, -1))
            elif isinstance(module, nn.Linear):
                w = _get_first_weight(weights_batch, module)
                ww = w.reshape(-1, *w.shape[2:])
                contribution = torch.nn.functional.linear(x, ww)
                x = module(x)

                contribution = contribution.reshape(
                    contribution.shape[0], weights_batch_size, -1, *contribution.shape[2:]
                )

                contributions.append(contribution)
                preactivations.append(x)
            elif isinstance(module, torchvision.models.resnet.Bottleneck):
                named_parameters = {x:y for x,y in module.named_parameters()}
                identity = x
                #Process this convolution
                w = weights_batch[self.lookup[named_parameters["conv1.weight"]]]
                contribution = F.conv2d(
                    x,
                    w.reshape(-1, *w.shape[2:]),
                    None,
                    module.conv1.stride,
                    module.conv1.padding,
                    module.conv1.dilation,
                    1
                )
                contributions.append(contribution.reshape(
                    contribution.shape[0], weights_batch_size, -1, *contribution.shape[2:]
                ))
                x = module.conv1(x)
                preactivations.append(x)
                x = module.bn1(x)
                x = module.relu(x)
                #and this convolution
                w = weights_batch[self.lookup[named_parameters["conv2.weight"]]]
                contribution = F.conv2d(
                    x,
                    w.reshape(-1, *w.shape[2:]),
                    None,
                    module.conv2.stride,
                    module.conv2.padding,
                    module.conv2.dilation,
                    1
                )
                contributions.append(contribution.reshape(
                    contribution.shape[0], weights_batch_size, -1, *contribution.shape[2:]
                ))
                x = module.conv2(x)
                preactivations.append(x)
                x = module.bn2(x)
                #and this convolution
                w = weights_batch[self.lookup[named_parameters["conv3.weight"]]]
                contribution = F.conv2d(
                    x,
                    w.reshape(-1, *w.shape[2:]),
                    None,
                    module.conv3.stride,
                    module.conv3.padding,
                    module.conv3.dilation,
                    1
                )
                contributions.append(contribution.reshape(
                    contribution.shape[0], weights_batch_size, -1, *contribution.shape[2:]
                ))
                x = module.conv3(x)
                preactivations.append(x)
                x = module.bn3(x)

                if module.downsample is not None:
                    identity = module.downsample(identity)

                x += identity
                x = module.relu(x)


            else:
                x = module(x)
        return x, contributions, preactivations


def explain(net, example, history_file="/raid/history.hdf5", batch_size=3, device="cuda"):
    wrapped = SequentialWrapper(net.eval()).eval()
    h = History(history_file=history_file)
    _, init, _ = wrapped(example, {k:torch.tensor(v).to(device).unsqueeze(0) for k,v in h[-1][0].items()}, 1)


    weights = torch.utils.data.DataLoader(h, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=False, pin_memory=True, prefetch_factor=3)
    print(len(weights))
    dots = None
    labels = None
    pre = None

    counter = 0
    with torch.no_grad():
        for w, i, l in tqdm.tqdm(weights, dynamic_ncols=True):
            counter +=1
            # if counter> 2048:
            #     break
            w = {k:v.to(device) for k,v in w.items()}
            actual_batchsize = len(next(iter(w.values())))
            outputs2, contribs, pre = wrapped(example, w, actual_batchsize)
            contribs = list([c.reshape(*c.shape[1:]) for c in contribs])
            if dots is None:
                labels = l
                ids = i
                preactivations = list([p - i.squeeze(0) for p, i in zip(pre, init)])
                dots = [
                    (c * p).sum(dim=list(range(1, len(p.shape)))).cpu() for c, p in zip(contribs, preactivations)
                ]
                c_norms = [
                    torch.sqrt((c * c).sum(dim=list(range(1, len(c.shape))))).cpu() for c in contribs
                ]
            else:
                new_dots = [
                    (c * p).sum(dim=list(range(1, len(p.shape)))).cpu() for c, p in zip(contribs, preactivations)
                ]
                new_c_norms = list([
                    torch.sqrt((c * c).sum(dim=list(range(1, len(c.shape))))).cpu() for c in contribs
                ])

                dots = [torch.cat((d1, d2), dim=0) for d1, d2 in zip(dots, new_dots)]
                c_norms = [torch.cat((d1, d2), dim=0) for d1, d2 in zip(c_norms, new_c_norms)]

                labels = torch.cat((labels, l), dim=0)
                ids = torch.cat((ids, i), dim=0)
    print(*[c.shape for c in preactivations])
    # contributions = list([c.reshape(*c.shape[1:]) for c in contributions]) #TODO: we don't actually need to store all of these, we can just compute the dot products and norms on the fly on the gpu and keep only those.
    # cont_pres = list([c.sum(dim=0, keepdim=True) for c in contributions])
    # diffs = list([(c-p.cpu()).norm() for c,p in zip(cont_pres, preactivations)])
    print(*[c.shape for c in preactivations])
    # print(*[c.shape for c in cont_pres])
    # print("DIFF", diffs, "should be small")
    # dot_products = list([
    #     (c * p.cpu()).sum(dim=list(range(1, len(p.shape)))) for c, p in zip(contributions, preactivations)
    # ])
    # c_norms = list([
    #     torch.sqrt((c * c).sum(dim=list(range(1, len(c.shape))))) for c in contributions
    # ])
    p_norms = list([
        torch.sqrt((p.cpu() * p.cpu()).sum(dim=list(range(1, len(p.shape))))) for p in preactivations
    ])
    cosines = list([
        d / (p * c + 1e-14) for d, p, c in zip(dots, p_norms, c_norms)
    ])
    print(*[c.shape for c in dots])

    # print(cosines)
    return None, preactivations, cosines, dots, c_norms, labels, ids


def class_statistics(contributions, preactivations, cosines, c_norms, labels):
    classes = {}
    weights = {}
    for i in range(len(cosines)):
        if f"layer{i}" not in classes:
            classes[f"layer{i}"] = {}
            weights[f"layer{i}"] = {}
        for cos, norm, y in zip(cosines[i], c_norms[i], labels):
            for l in y[:1]:
                l = l.item()
                if l not in classes[f"layer{i}"]:
                    classes[f"layer{i}"][l] = []
                    weights[f"layer{i}"][l] = []
                classes[f"layer{i}"][l].append(cos.item())
                weights[f"layer{i}"][l].append(norm.item())
    for layer in classes:
        for label in classes[layer]:
            classes[layer][label] = np.array(classes[layer][label])
            weights[layer][label] = np.array(weights[layer][label])
    return classes, weights