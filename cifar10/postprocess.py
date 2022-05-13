import torch
import numpy as np
import os
from timeit import default_timer as timer
from xai_tracking.nn import VGG_net, resnet50, resnet18
from xai_tracking.xai import blocksparse_conv2d
import tqdm
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
        deltas = [f for f in self.files if f.startswith(self.parameter) and f".{epoch}." in f]
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

@torch.no_grad()
def aggregate_by_examples(model, history_folder, num_examples):
    files = [f for f in os.listdir(history_folder) if not f.startswith("ids")]
    parameters = set([".".join(f.split(".")[:2]) for f in files])
    parameter_to_layer = index_layers(model)
    print(parameters)
    for parameter in ["group0.param4"]: #parameters:
        layer = parameter_to_layer[parameter]
        module = layer
        g = torch.empty((4096, *module.weight.shape), requires_grad=False).pin_memory().cuda()
        
        loader = torch.utils.data.DataLoader(History(history_folder, files, parameter), num_workers=40, pin_memory=True, batch_size=None)
        print(layer)
        aggregate, jit = None, None
        for data in tqdm.tqdm(loader):
            if len(data) == 2:
                W, ids = data
            else:
                U, V, ids = data
                ids = ids.long()
                start = timer()
                U = U.transpose(0,1).to("cuda", non_blocking=True)
                V = V.transpose(0,1).to("cuda", non_blocking=True)
                end = timer()
                print("Transferred in ", end - start)
                # print(U.shape[1], U.shape, V.shape, layer.weight.shape)
                if jit is None:
                    class _tmp_(torch.nn.Module):
                        # batch_size : torch.jit.Final[int] = 4096
                        stride : torch.jit.Final[int] = module.stride
                        padding : torch.jit.Final[int] = module.padding
                        dilation : torch.jit.Final[int] = module.dilation
                        groups : torch.jit.Final[int] = module.groups
                        def forward(self, g, X, W, *args, **kwargs):
                            # print(self.stride, self.padding, self.dilation, self.groups)
                            # print(g[0, :g.shape[1], :g.shape[2], :g.shape[3], :g.shape[4]].shape)
                            # print(torch.numel(g) *1. / (torch.numel(X) + torch.numel(W)))
                            with torch.no_grad():
                                for i in range(X.shape[1]):
                                    g[i, :g.shape[1], :g.shape[2], :g.shape[3], :g.shape[4]] = torch.nn.functional.conv2d(X[:,i:i+1,...], W[:,i:i+1,...], None, self.dilation, self.padding, self.stride, self.groups).transpose(0,1)[:g.shape[1], :g.shape[2], :g.shape[3], :g.shape[4]]
                                return g
                    # print("jit compiling backward pass for per-example gradients")
                    # jit = _tmp_()
                    # jit = torch.jit.script(jit)
                    jit = blocksparse_conv2d
                    # jit = torch.jit.trace(_tmp_(), (g, U, V), check_trace=False)
                    # jit = torch.jit.optimize_for_inference(jit)
                start = timer()
                g = jit(g, U, V, module.stride, module.padding, module.dilation, module.groups)
                # print(g[0])
                print(g[:len(ids)].abs().max().item(), torch.linalg.norm(g[:len(ids)]).item())
                # print(blocksparse.blocksparse_conv2d(g, U, V, module.stride, module.padding, module.dilation, module.groups)[0])
                end = timer()
                print("Convoluted in ", end - start)
                # for i in tqdm.tqdm(range(U.shape[1])):
                #     g = torch.nn.functional.conv2d(U[:,i:i+1,...], V[:,i:i+1,...], None, layer.dilation, layer.padding, layer.stride, layer.groups).transpose(0,1)[:layer.weight.shape[0], :layer.weight.shape[1], :layer.weight.shape[2], :layer.weight.shape[3]]

                if aggregate is None:
                    aggregate = torch.zeros((num_examples, *g.shape[1:])).pin_memory()

                start = timer()
                gcpu = g[:len(ids)].to("cpu", non_blocking=False)
                end = timer()
                print("Transferred out ", end - start)
                start = timer()
                aggregate.index_add_(0, ids, gcpu)
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
    net = VGG_net(ghost_samples=0).to_sequential()
    net.load_state_dict(torch.load(os.path.join(PATH, "cifar10.pt")))
    aggregate_by_examples(net, "/ceph/debug", 50000)