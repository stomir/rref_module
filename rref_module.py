import torch
import torch.distributed.rpc as rpc
import fire #type: ignore
from typing import *

PyRRef = torch._C._distributed_rpc.PyRRef
one_module : Optional[torch.nn.Module] = None

def to_value(a, device = None):
    if type(a) is RRefTensor or type(a) is PyRRef:
        return a.to_here()
    if isinstance(a, tuple):
        a = tuple(map(to_value, a))
    if type(a) is dict:
        a = {k:to_value(v) for k,v in a.items()}
    if type(a) is torch.Tensor and device is not None:
        a = a.to(device)
    return a

def to_arg(a):
    if type(a) is RRefTensor:
        return a.rref
    if isinstance(a, tuple):
        a = tuple(map(to_arg, a))
    if type(a) is dict:
        a = {k:to_arg(v) for k,v in a.items()}
    return a

def r_getattr(a, name):
    return getattr(to_value(a), name)

def r_add(a, b):
    return to_value(a) + to_value(b)

def r_getitem(a, b):
    return to_value(a)[to_value(b)]

def r_tuplesize(a):
    a = to_value(a)
    if isinstance(a, tuple):
        return len(a)
    else:
        return 0
     
def remote(worker : rpc.WorkerInfo, func, *args):
    return RRefTensor(rpc.remote(worker, func, args), worker)

class RRefTensor:
    def __init__(self, rref : rpc.RRef, worker : rpc.WorkerInfo):
        self.rref = rref
        self.worker = worker
        self.tuple_size = None
        
    def to_here(self):
        return self.rref.to_here()
    
    def local_value(self):
        return self.rref.local_value()
    
    def __getattr__(self, name : str):
        if hasattr(torch.Tensor, name) and callable(getattr(torch.Tensor, name)):
            #method call
            return RemoteMethod(self, name)
        assert name in {'shape', 'device', 'dtype'}, f"weird attr name {name=}"
        return rpc.rpc_sync(self.worker, r_getattr, args=(self.rref, name))
    
    def __add__(self, other):
        return remote(self.worker, r_add, self.rref, to_arg(other))
    
    def __getitem__(self, idx):
        if type(idx) is int:
            if self.tuple_size is None:
                self.tuple_size = rpc.rpc_sync(self.worker, r_tuplesize, args=(self.rref,))
            if idx >= self.tuple_size:
                raise IndexError
        return remote(self.worker, r_getitem, self.rref, to_arg(idx))
    
    def __repr__(self):
        return f'r_tensor(..., worker={self.worker.name})'

def r_call_method(tensor : rpc.RRef, name : str, args, kwargs):
    args = to_value(args)
    kwargs = to_value(kwargs)
    return getattr(torch.Tensor, name)(to_value(tensor), *args, **kwargs)

class RemoteMethod(NamedTuple):
    tensor : RRefTensor
    name : str
        
    def __call__(self, *args, **kwargs):
        return remote(self.tensor.worker, r_call_method, self.tensor.rref, self.name, to_arg(args), to_arg(kwargs))
    
def r_forward(module, args, kwargs):
    return module.to_here().forward(*args, **kwargs)

def r_parameter_count(module):
    return len(list(module.to_here().module.parameters()))

def r_get_paramter_by_idx(module, idx):
    return list(module.to_here().module.parameters())[idx]

class Wrapper:
    def __init__(self, module, device, move = False):
        assert device is not None, f"none device {module=}"
        self.device = torch.device(device)
        self.module = module
        
        if move:
            self.module = module.to(device)

        for name, param in self.module.named_parameters():
            assert (param.device == self.device), f"wrong device {name=} {self.module=} {device=} {param.device=}"

    def localize(self, arg):
        if type(arg) is tuple:
            return tuple(map(self.localize, arg))
        if type(arg) is list:
            return list(map(self.localize, arg))
        if type(arg) is dict:
            return {k:self.localize(v) for k,v in arg.items()}
        if type(arg) is PyRRef or type(arg) is RRefTensor:
            arg = arg.to_here()
        if type(arg) is torch.Tensor:
            return arg.to(self.device) if self.device is not None else arg
        return arg
        
    def forward(self, *args, **kwargs):
        args = tuple(map(self.localize, args))
        kwargs = {k: self.localize(v) for k,v in kwargs.items()}
        return self.module(*args, **kwargs)

def remote_copy(module : torch.nn.Module, worker : Union[str, rpc.WorkerInfo], device : Optional[Union[torch.device, int, str]]):
    rref = rpc.remote(worker, Wrapper, args=(module, device, True))
    return RRefModule(rref, rpc.get_worker_info(worker))
    
class RRefModule(torch.nn.Module):
    def __init__(self, module_rref : rpc.RRef[Wrapper], worker : rpc.WorkerInfo, name : Optional[str] = None, device : Optional[int] = None):
        super().__init__()
        self.worker = worker
        self.rref = module_rref
        self.param_cache : Optional[List[rpc.RRef[torch.nn.Parameter]]] = None
        self.name = name
        self.device = device
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        
    def __call__(self, *args, **kwargs) -> RRefTensor:
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs) -> RRefTensor:
        args = tuple(map(to_arg, args))
        kwargs = {k : to_arg(a) for k,a in kwargs.items()}
        return RRefTensor(rpc.remote(self.worker, r_forward, args=(self.rref, args, kwargs)), self.worker)
    
    def rref_parameters(self) -> List[rpc.RRef[torch.nn.Parameter]]:
        if self.param_cache is None:
            n = rpc.remote(self.worker, r_parameter_count, args=(self.rref,)).to_here()
            self.param_cache = [rpc.remote(self.worker, r_get_paramter_by_idx, args=(self.rref, i)) for i in  range(n)]
        return self.param_cache
    
    def __repr__(self):
        name = f"{self.name}, " if self.name is not None else ""
        device = f", device={self.device}" if self.device is not None else ""
        return f'RRefM({name}worker={self.worker.name}{device})'

