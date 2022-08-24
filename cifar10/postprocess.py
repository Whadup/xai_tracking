import torch
import numpy as np
import os
from timeit import default_timer as timer
from xai_tracking.nn import VGG_net, resnet50, resnet18
from xai_tracking.xai import blocksparse_conv2d, weights_conv2d
import tqdm
import multiprocessing
PATH = os.path.dirname(__file__)

def index_layers(model):
    tmp = {}
    for module in model.modules():
        for p in module.parameters():
            tmp[p] = module
    tmp2 = {}
    for j, p in enumerate(model.parameters()):
        tmp2[f"group{0}.param{j}"] = tmp[p]
    return tmp2


class HistoryV2(torch.utils.data.Dataset):
    def __init__(self, history_folder, files, parameter):
        super().__init__()
        self.history_folder = history_folder
        self.files = files
        self.parameter = parameter
        self.epochs = sorted(list(set([int(f.split(".")[-2]) for f in files if f.startswith(parameter) and int(f.split(".")[-2]) >= 1])))
        # self.history = {"default": zipfile.ZipFile(self.history_file, "r")}
    
    def __len__(self):
        return 50000
        # return (len(self.history["default"].namelist()) - 1)
    def __getitem__(self, i):
        a = []
        b = []
        agg = None
        for epoch in self.epochs:
            ids = np.load(os.path.join(self.history_folder, f"ids.{epoch}.npy"))
            if i in ids.tolist():
                deltas = [f for f in self.files if f.startswith(f"{self.parameter}.") and f".{epoch}." in f]
                if len(deltas) == 1:
                    # print(os.path.join(self.history_folder, deltas[0]))
                    W = np.load(os.path.join(self.history_folder, deltas[0]), mmap_mode="r")
                    for ii in np.argwhere(ids == i).reshape(-1):
                        if agg is None:
                            agg = np.copy(W[ii])
                        else:
                            agg += W[ii] 
                    del W
                        # a.append(np.copy(W[ii]))
                        
                elif len(deltas) == 2:
                    if ".U." in deltas[0]:
                        U, V = deltas
                    else:
                        V, U = deltas
                    start = timer()
                    U, V = np.load(os.path.join(self.history_folder,U), mmap_mode="r"), np.load(os.path.join(self.history_folder, V), mmap_mode="r")
                   
                    for ii in np.argwhere(ids == i).reshape(-1):
                        a.append(np.copy(U[ii]))
                        b.append(np.copy(V[ii]))
                    end = timer()
                    # print("loaded in ", end - start)
        # print(a, b)
        if len(b):
            return np.array([i]), np.array(a), np.array(b)
        return np.array([i]), agg[np.newaxis,...]

class History(torch.utils.data.Dataset):
    def __init__(self, history_folder, files, parameter):
        super().__init__()
        self.history_folder = history_folder
        self.files = files
        self.parameter = parameter
        self.epochs = list(set([int(f.split(".")[-2]) for f in files if f.startswith(parameter) and int(f.split(".")[-2]) >= 0]))
        # self.history = {"default": zipfile.ZipFile(self.history_file, "r")}
    
    def __len__(self):
        return len(self.epochs)
        # return (len(self.history["default"].namelist()) - 1)
    def __getitem__(self, i):
        epoch = self.epochs[i]
        deltas = [f for f in self.files if f.startswith(f"{self.parameter}.") and f".{epoch}." in f]
        ids = np.load(os.path.join(self.history_folder, f"ids.{epoch}.npy"))
        if len(deltas) == 1:
            W = np.load(os.path.join(self.history_folder, deltas[0]))
            return W, ids
        elif len(deltas) == 2:
            if ".U." in deltas[0]:
                U, V = deltas
            else:
                V, U = deltas
            start = timer()
            U, V = np.load(os.path.join(self.history_folder,U)), np.load(os.path.join(self.history_folder, V))
            end = timer()
            # print("loaded in ", end - start)
            return U, V, ids
        else:
            print(deltas)

def aggregate_by_examples_v2(model, history_folder, num_examples, parameter=None):
    files = [f for f in os.listdir(history_folder) if not f.startswith("ids") and not ".aggregate" in f]
    parameters = set([".".join(f.split(".")[:2]) for f in files])
    parameter_to_layer = index_layers(model)
    print(parameters)
    parameters = sorted(list(parameters))[1:4]
    if parameter is not None:
        parameters = [parameter]
    print(parameters)

    batch_size = 1
    for parameter in parameters: #["group0.param4"]: #:
        layer = parameter_to_layer[parameter]
        module = layer
        g = np.memmap(os.path.join("/ceph/tmp", f"{parameter}.aggregate.npy"), shape=(num_examples, *module.weight.shape), mode="w+", dtype=np.float32)
        loader = torch.utils.data.DataLoader(HistoryV2(history_folder, files, parameter), num_workers=100, pin_memory=False, batch_size=None, persistent_workers=True, shuffle=False)
        for data in tqdm.tqdm(loader):
            if len(data) == 2:
                i, w = data
                print(i, w.shape)
                # w = w.cuda()
                g[i] = w[0] #.sum(0).cpu()
            else:
                i, U, V = data
                print(i, U.shape, V.shape)
                U = U.to("cuda", non_blocking=False)
                V = V.to("cuda", non_blocking=False)
                g[i] = weights_conv2d(module.weight.shape, U, V, module.dilation, module.padding, module.stride, module.groups).cpu()
        g.flush()
        del loader
@torch.no_grad()
def aggregate_by_examples(model, history_folder, num_examples, parameter=None):
    files = [f for f in os.listdir(history_folder) if not f.startswith("ids")]
    parameters = set([".".join(f.split(".")[:2]) for f in files])
    parameter_to_layer = index_layers(model)
    if parameter is not None:
        parameters = [parameter]
    print(parameters)
    batch_size = 1
    for parameter in parameters: #["group0.param4"]: #:
        layer = parameter_to_layer[parameter]
        module = layer
        g = torch.empty((4096 * batch_size, *module.weight.shape), requires_grad=False).pin_memory().cuda()
        
        loader = torch.utils.data.DataLoader(History(history_folder, files, parameter), num_workers=40, pin_memory=True, batch_size=batch_size)
        print(layer)
        aggregate, jit = None, None
        for data in tqdm.tqdm(loader):
            if len(data) == 2:
                g, ids = data
                ids = ids.long().view(-1)
                g = g.view(-1, *g.shape[2:])
                if aggregate is None:
                    aggregate = torch.zeros((num_examples, *g.shape[1:])).pin_memory()
                aggregate.index_add_(0, ids, g)
            else:
                U, V, ids = data
                # flatten batch dimension 0 that with stored batch dimension 1
                U = U.view(-1, *U.shape[2:])
                V = V.view(-1, *V.shape[2:])
                ids = ids.long().view(-1)

                start = timer()
                U = U.to("cuda", non_blocking=False)
                V = V.to("cuda", non_blocking=False)
                # U = U.transpose(0,1).to("cuda", non_blocking=False)
                # V = V.transpose(0,1).to("cuda", non_blocking=False)
                end = timer()
                print("Transferred in ", end - start)
                
                start = timer()
                g = blocksparse_conv2d(g, U, V, module.stride, module.padding, module.dilation, module.groups)
                end = timer()
                print("Convoluted in ", end - start)

                if aggregate is None:
                    aggregate = torch.zeros((num_examples, *g.shape[1:])).pin_memory()

                start = timer()
                gcpu = g[:len(ids)].to("cpu", non_blocking=False)
                end = timer()
                print("Transferred out ", end - start)
                start = timer()
                # for i, r in zip(ids, gcpu):
                #     aggregate[i] += r
                aggregate.index_add_(0, ids, gcpu)
                # aggregate.index_put_(0, ids, gcpu, accumulate=True)
                end = timer()
                print("Index_Added in ", end - start)
        total = aggregate.sum(axis=0)
        init = np.load(os.path.join(history_folder, f"{parameter}.-1.npy"))
        total += init
        # print(np.linalg.norm(gcpu.numpy()))
        print(np.linalg.norm(total - layer.weight.numpy()), np.linalg.norm(total), np.linalg.norm(layer.weight.numpy()))
        np.save(os.path.join(history_folder, f"{parameter}.aggregate.npy"), aggregate.numpy())
                # print("Factorized", U, V)




if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
    torch.use_deterministic_algorithms(True)
    net = VGG_net(ghost_samples=0).to_sequential()
    net.load_state_dict(torch.load(os.path.join(PATH, "cifar10.pt")))
    aggregate_by_examples_v2(net, "/raid/pfahler/cifar10_history", 50000, parameter="group0.param12")
    #12, 11, 10