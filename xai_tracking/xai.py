from multiprocessing import shared_memory, Queue, Process
import os
import multiprocessing
# multiprocessing.set_start_method("spawn")
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
except:
    pass
import tqdm
import h5py
import bcolz
import zipfile
import numpy as np
import itertools
import io
from copy import deepcopy

FULL_BUFFER = 64
HALF_BUFFER = FULL_BUFFER // 2

#import torch_geometric
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
        # if isinstance(x, torch_geometric.data.Data):
        #     return x, (index if torch.is_tensor(index) else torch.tensor(index))
        if torch.is_tensor(index):
            return (*x, index)
        return (*x, torch.tensor(index))

def probabilistic_coreset(X, epsilon, delta):
    k  = int(np.floor(3.5 * np.log(1.0 / delta)) + 1)
    print(k, int(np.ceil(4 * k / epsilon)))
    S = X[np.random.choice(len(X), int(np.ceil(4 * k / epsilon)))].reshape(k, -1, X.shape[1])
    print(S.shape)
    s = S.mean(axis=1)
    print(s.shape)
    D = np.linalg.norm(s[:, np.newaxis, :] - s[np.newaxis, ...], 2, axis=2)
    print(D.shape)
    i = D.sum(axis=1).argmin()
    true = X.sum(axis=0)
    est = S[i].sum(axis=0) * len(X) / len(S[i])
    print((true * est).sum() / (np.linalg.norm(true) * np.linalg.norm(est)))
    return S[i]

def store_fullrank(key=None, shape=None, shm=None, shm2=None, queue=None, lock=None):
    from vector_summarization_coreset.FastEpsCoreset import sparseEpsCoreset
    from vector_summarization_coreset.Utils import getNormalizedWeightedSet
    import vector_summarization_coreset.FrankWolfeCoreset as FWC

    shm = shared_memory.SharedMemory(name=shm)
    array = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

    shm2 = shared_memory.SharedMemory(name=shm2)
    ids = np.ndarray((shape[0], ), dtype=np.int32, buffer=shm2.buf)

    dataset = None
    i = 0
    j = 0
    while True:
        try:
            workload_start, workload_end = queue.get()
            try:
                data = array[workload_start:workload_end]
                batch_ids = ids[workload_start:workload_end]
                # data = data.reshape(HALF_BUFFER * 512, -1)
                with lock:
                    np.save(key + "." + str(j), data)
                    np.save(os.path.join(os.path.dirname(key), "ids") +  "." + str(j), batch_ids)
                j += 1
            except Exception as e:
                raise RuntimeError(e)
        except Exception as e:
            # print(e)
            shm.close()
            shm2.close()
            # print("quit worker process for", key)
            return

def store_lowrank(key=None, shape1=None, shape2=None, shm_u=None, shm_v=None, shm2=None, queue=None, lock=None):
    from vector_summarization_coreset.FastEpsCoreset import sparseEpsCoreset
    from vector_summarization_coreset.Utils import getNormalizedWeightedSet
    import vector_summarization_coreset.FrankWolfeCoreset as FWC

    shm_u = shared_memory.SharedMemory(name=shm_u)
    U = np.ndarray(shape1, dtype=np.float32, buffer=shm_u.buf)

    shm_v = shared_memory.SharedMemory(name=shm_v)
    V = np.ndarray(shape2, dtype=np.float32, buffer=shm_v.buf)

    shm2 = shared_memory.SharedMemory(name=shm2)
    ids = np.ndarray((shape1[0], ), dtype=np.int32, buffer=shm2.buf)

    dataset = None
    i = 0
    j = 0
    while True:
        try:
            workload_start, workload_end = queue.get()
            try:
                u = U[workload_start:workload_end]
                v = V[workload_start:workload_end]
                batch_ids = ids[workload_start:workload_end]
                # data = data.reshape(HALF_BUFFER * 512, -1)
                with lock:
                    np.save(key + ".U." + str(j), u)
                    np.save(key + ".V." + str(j), v)
                    np.save(os.path.join(os.path.dirname(key), "ids") +  "." + str(j), batch_ids)
                j += 1
            except Exception as e:
                raise RuntimeError(e)
        except Exception as e:
            # print("quit worker process for", key)
            shm_u.close()
            shm_v.close()
            shm2.close()
            return


second_device = torch.device("cuda:0")

class WrappedSGD(torch.optim.SGD):
    def __init__(self, *args, history_file="/raid/history.hdf5", mode="examplewise", **kwargs):
        if kwargs.pop("momentum", 0.0) > 0:
            raise RuntimeError("We do not support momentum at the moment.")
        if kwargs.pop("weight_decay", 0.0) > 0:
            raise RuntimeError("We do not support weight decay at the moment.")
        print("setting stupid default parameters, change this!")
        super().__init__(*args, **kwargs) # momentum=0.9, weight_decay=5e-4
        self.history_file = history_file
        if mode == "batchwise":
            self.gradient_per_example = False
        elif mode== "examplewise":
            self.gradient_per_example = True
        else:
            raise RuntimeError(f"Unknown mode '{mode}'. We support 'batchwise' and 'examplewise'.")
        self.weight_saver = {}
        tmp = {}
        for i,group in enumerate(self.param_groups):
            for j,p in enumerate(group['params']):
                self.weight_saver[p] = p.data.detach().clone()
                tmp[f"group{i}.param{j}"] = p.data.detach().cpu().numpy()
        
        # self.history =  h5py.File(history_file, "w") #, rdcc_nbytes=64*1024*1024*1024)
        # self.history_grp = self.history.create_group("history")
        self.history_file = history_file
        os.makedirs(history_file, exist_ok=True)
        self.shared_arrays = {}
        self.garbage = []
        self.queues = {}
        self.processes = {}
        self.jit = {}
        self.lock = multiprocessing.Lock()
        
        self.counter = 0
        self.hooked_up = False
        self.parameter_to_module = {}
        self.per_example_output_buffers = {}
        self.per_example_input_buffers = {}
        self.pinned_buffers = {}

    def __del__(self):
        self.done()

    def done(self):
        for module in self.processes:
            p = self.processes[module]
            q = self.queues[module]
            print("Terminate ", p.pid)
            q.put((self.flush_point, self.counter))
            q.put(None)
            p.join()
            q.close()
        self.processes = {}
        self.queues = {}
        for shm in self.garbage:
            shm.close()
            shm.unlink()
        self.shared_arrays = {}
        self.garbage = []
        # for q in self.queues:
        #     q.close()

    @torch.no_grad()
    def hookup(self, model):
        for module in model.modules():
            first_parameter = None
            for p in module.parameters():
                if first_parameter is None:
                    first_parameter = p
                self.parameter_to_module[p] = module
            if self.gradient_per_example and isinstance(module,(torch.nn.Conv2d, torch.nn.Linear)):
                def store_output(module, grad_input, grad_output):
                    self.per_example_output_buffers[module] = grad_output[0]
                def store_input(module, input, output):
                    self.per_example_input_buffers[module] = input[0]
                module.register_full_backward_hook(store_output)
                module.register_forward_hook(store_input)
        self.hooked_up = True

    @torch.no_grad()
    def setup_tracking(self, key, module, X, W):
        g = torch.empty((X.shape[1], *module.weight.shape), requires_grad=False).pin_memory().to(second_device)
        self.pinned_buffers[module] = g
        if torch.numel(g) < torch.numel(X) + torch.numel(W):
            class _tmp_(torch.nn.Module):
                batch_size : torch.jit.Final[int] = X.shape[1]
                stride : torch.jit.Final[int] = module.stride
                padding : torch.jit.Final[int] = module.padding
                dilation : torch.jit.Final[int] = module.dilation
                groups : torch.jit.Final[int] = module.groups
                def forward(self, g, X, W):
                    # print(self.stride, self.padding, self.dilation, self.groups)
                    # print(g[0, :g.shape[1], :g.shape[2], :g.shape[3], :g.shape[4]].shape)
                    # print(torch.numel(g) *1. / (torch.numel(X) + torch.numel(W)))
                    with torch.no_grad():
                        for i in range(self.batch_size):
                            g[i, :g.shape[1], :g.shape[2], :g.shape[3], :g.shape[4]] = torch.nn.functional.conv2d(X[:,i:i+1,...], W[:,i:i+1,...], None, self.dilation, self.padding, self.stride, self.groups).transpose(0,1)[:g.shape[1], :g.shape[2], :g.shape[3], :g.shape[4]]
                        return g
            print("jit compiling backward pass for per-example gradients")
            self.jit[module] = torch.jit.trace(_tmp_(), (g, X, W), check_trace=False)
            self.jit[module] = torch.jit.optimize_for_inference(self.jit[module])
            torch.cuda.empty_cache()
            existing_shm = shared_memory.SharedMemory(create=True, size=FULL_BUFFER * torch.numel(g) * 4)
            existing_shm2 = shared_memory.SharedMemory(create=True, size=len(g) * FULL_BUFFER * 4)
            self.garbage.append(existing_shm)
            self.garbage.append(existing_shm2)
            g_array = np.ndarray((X.shape[1] * FULL_BUFFER, *g.shape[1:]),  dtype=np.float32, buffer=existing_shm.buf)
            print(g_array.shape)
            id_array = np.ndarray((X.shape[1] * FULL_BUFFER, ),  dtype=np.int32, buffer=existing_shm2.buf)
            self.shared_arrays[module] = (g_array, id_array)
            self.queues[module] = Queue(1)
            self.processes[module] = Process(target=store_fullrank, args=(os.path.join(self.history_file, key), g_array.shape, existing_shm.name, existing_shm2.name, self.queues[module], self.lock))
            self.processes[module].start()
        else:
            existing_shm_u = shared_memory.SharedMemory(create=True, size=FULL_BUFFER * torch.numel(X) * 4)
            existing_shm_v = shared_memory.SharedMemory(create=True, size=FULL_BUFFER * torch.numel(W) * 4)
            existing_shm2 = shared_memory.SharedMemory(create=True, size=len(g) * FULL_BUFFER * 4)
            self.garbage.append(existing_shm_u)
            self.garbage.append(existing_shm_v)
            self.garbage.append(existing_shm2)
            u_array = np.ndarray((X.shape[1] * FULL_BUFFER, X.shape[0], *X.shape[2:]),  dtype=np.float32, buffer=existing_shm_u.buf)
            v_array = np.ndarray((W.shape[1] * FULL_BUFFER, W.shape[0], *W.shape[2:]),  dtype=np.float32, buffer=existing_shm_v.buf)
            print(u_array.shape, v_array.shape)

            id_array = np.ndarray((X.shape[1] * FULL_BUFFER, ),  dtype=np.int32, buffer=existing_shm2.buf)
            self.shared_arrays[module] = (u_array, v_array, id_array)
            self.queues[module] = Queue(1)
            self.processes[module] = Process(target=store_lowrank, args=(os.path.join(self.history_file, key), u_array.shape, v_array.shape, existing_shm_u.name, existing_shm_v.name, existing_shm2.name, self.queues[module], self.lock))
            self.processes[module].start()

    @torch.no_grad()
    def step(self, ids, labels, *args):
        if self.gradient_per_example and not self.hooked_up:
            raise RuntimeError("Optimizer has not hooked up with the model yet. Call `optimizer.hookup(model)`.")
        if not self.gradient_per_example:
            for group in self.param_groups:
                for p in group['params']:
                    self.weight_saver[p].copy_(p.data)
        super().step(*args)
        tmp = {}
        # if self.gradient_per_example:
        #     grp = []
        #     for i in range(len(ids)):
        #         grp.append(self.history.create_group(f"iteration_{self.counter}"))
        #         self.counter += 1
        # else:
        #     grp = self.history.create_group(f"iteration_{self.counter}")
        #     self.counter += 1

        for I, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            for j,p in enumerate(group['params']):
                key = f"group{I}.param{j}"
                if self.gradient_per_example:
                    module = self.parameter_to_module[p]
                    if module in self.per_example_input_buffers:
                        if isinstance(module, torch.nn.Conv2d):
                            # with torch.cuda.stream(torch.cuda.Stream()):
                            X = self.per_example_input_buffers[module].transpose(0, 1)#.to(second_device)
                            W = self.per_example_output_buffers[module].transpose(0, 1)#.to(second_device)
                            if module not in self.processes:
                                self.setup_tracking(key, module, X, W)
                            if len(self.shared_arrays[module]) == 2:
                                # full rank
                                g = self.pinned_buffers[module]
                                g[...] = self.jit[module](g, X, W)
                                g *= -lr
                                g_array, id_array = self.shared_arrays[module]
                                g_array[self.counter: self.counter + X.shape[1]] = g.to("cpu", non_blocking=True).numpy()[:]
                                id_array[self.counter: self.counter + X.shape[1]] = ids.to("cpu").numpy()[:]
                            elif len(self.shared_arrays[module]) == 3:
                                # low rank
                                u_array, v_array, id_array = self.shared_arrays[module]
                                u_array[self.counter: self.counter + X.shape[1]] = -lr * X.transpose(0, 1).to("cpu", non_blocking=True).numpy()[:]
                                v_array[self.counter: self.counter + X.shape[1]] = W.transpose(0, 1).to("cpu", non_blocking=True).numpy()[:]
                                id_array[self.counter: self.counter + X.shape[1]] = ids.to("cpu").numpy()[:]
                            if self.counter + X.shape[1] == X.shape[1] * HALF_BUFFER:
                                self.queues[module].put((0, self.counter + X.shape[1]))
                                self.flush_point = self.counter + X.shape[1]
                                # print("put(0)", key)
                            elif self.counter + X.shape[1] == X.shape[1] * FULL_BUFFER:
                                self.queues[module].put((X.shape[1] * HALF_BUFFER, self.counter + X.shape[1]))
                                self.flush_point = 0
                                # print("put(512*HALF_BUFFER)", key)
                            # print("success")
                            # self.history_grp[key].resize(self.counter + X.shape[1], 0)
                            # self.history_grp[key][self.counter: self.counter + X.shape[1]] = g.detach().to("cpu", non_blocking=True).numpy()
                                # lam = torch.linalg.svdvals(g.reshape(len(g), -1))
                                # print((lam / (lam + 1e-7)).sum(), lam[2 * int((lam / (lam + 1e-6)).sum()):].sum(), len(g))
                            # for example_grp, example_g in zip(grp, g):
                            #     example_grp.create_dataset(key, data=lr * example_g.detach().cpu().numpy())
                        elif isinstance(module, torch.nn.Linear):
                            X = self.per_example_input_buffers[module]#.transpose(0, 1)
                            W = self.per_example_output_buffers[module]#.transpose(0, 1)
                            # print(X.shape, W.shape, module.weight.shape)
                            # for example_grp, example_i in zip(grp, range(len(X))):
                            #     example_grp.create_dataset(key+".U", data=lr * X[example_i].detach().cpu().numpy())
                            #     example_grp.create_dataset(key+".V", data=W[example_i].detach().cpu().numpy())
                else:
                    ds = grp.create_dataset(key, data=torch.sub(p.data.detach(), self.weight_saver[p]).detach().cpu().numpy())#, compression="lzf")
        if self.gradient_per_example:
            # for i, example_grp in enumerate(grp):
            #     example_grp.create_dataset("ids", data=ids[i:i+1].detach().cpu().numpy())
            #     example_grp.create_dataset("labels", data=labels[i:i+1].detach().cpu().numpy())
            self.counter += len(ids)
            if self.counter + len(ids) > len(ids) * FULL_BUFFER: #next buffer does not fit anymore.
                self.counter = 0
            

        else:
            self.counter += 1
            # grp.create_dataset("ids", data=ids.detach().cpu().numpy())
            # grp.create_dataset("labels", data=labels.detach().cpu().numpy())
        # with self.history.open(f"{self.counter}.pt", "w") as f:
        #     torch.save((tmp, ids, labels), f)

class WrappedOptimizer(torch.optim.Optimizer):
    def __init__(self, base_cls, history_file="/raid/history.hdf5", mode="batchwise"):
        self.internal_optimizer_cls = base_cls
        self.history_file = history_file
        if mode != "batchwise":
            raise RuntimeError("General Optimizers only support batchwise mode. Consider using WrappedAdam or WrappedSGD.")
    def __call__(self, *args, **kwargs):
        history_file = self.history_file
        class _InternalOptimizer(self.internal_optimizer_cls):
            """
            Optimizer that logs all weight changes with one additional call to `archive(ids)` inserted after `step()`.
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                

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