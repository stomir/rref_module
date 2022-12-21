#from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig #type: ignore
import transformers #type: ignore
from accelerate import init_empty_weights, infer_auto_device_map  #type: ignore
from accelerate.utils import set_module_tensor_to_device, get_balanced_memory #type: ignore
from transformers.utils import cached_file, WEIGHTS_INDEX_NAME #type: ignore
from transformers.utils.hub import get_checkpoint_shard_files #type: ignore
from transformers.modeling_utils import load_state_dict, get_disk_only_shard_files #type: ignore
import accelerate
import torch
from .rref_module import Wrapper, RRefModule
import torch.distributed.rpc as rpc
from typing import *
import warnings
import os

def r_partially_load(mname : str, device_map: Dict[str, Union[str,int]], cached_file_kwargs : Dict = {}) -> Dict[str, torch.nn.Module]:
    resolved_archive_file = cached_file(mname, WEIGHTS_INDEX_NAME, _raise_exceptions_for_missing_entries=False, **cached_file_kwargs)
    if resolved_archive_file is None:
        warnings.warn("Loading a non-sharded model (or could not resolve archive file for some other reason)")
        device_map = {s : i if i != 'disk' else 'cpu' for s,i in device_map.items()}
        model = transformers.AutoModelForCausalLM.from_pretrained(mname, device_map=device_map)
    else:
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(mname, resolved_archive_file, **cached_file_kwargs)
        folder = os.path.sep.join(resolved_archive_file[0].split(os.path.sep)[:-1])
        
        disk_only_shard_files = get_disk_only_shard_files(device_map = device_map, sharded_metadata = sharded_metadata)
        
        config = transformers.AutoConfig.from_pretrained(mname)
        with init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_config(config)
        
        for shard_file in resolved_archive_file:
            if shard_file in disk_only_shard_files: continue
            
            state_dict = load_state_dict(shard_file)
    
            for param_name, value in state_dict.items():
                weight_name = param_name
                while len(weight_name) > 0 and weight_name not in device_map:
                    weight_name = ".".join(weight_name.split(".")[:-1])
                if weight_name == "0" or device_map[weight_name] == "disk":
                    continue
                device = device_map[weight_name]
                
                set_module_tensor_to_device(model, param_name, device, value=value)
                                   
            del state_dict
        
    ret : Dict[str, torch.nn.Module] = {}
    for submodule, device in device_map.items():
        if device == 'disk': continue
        ret[submodule] = model.get_submodule(submodule)
    
    for submodule, model in ret.items():
        for param in model.parameters():
            assert not param.is_meta, f"meta parameter left: {submodule=} {param=}"
    
    return ret

def r_wait(partial_load : rpc.RRef):
    partial_load.to_here()
    
def r_get_part(partial_load : rpc.RRef, submodule_name : str, device : torch.device) -> Optional[Wrapper]:
    module : Dict[str, torch.nn.Module] = partial_load.local_value()
    return Wrapper(module[submodule_name], device=device)

def set_submodule(module : torch.nn.Module, submodule_name : str, new_module) -> None:
    path = submodule_name.split('.')
    parent = module.get_submodule('.'.join(path[:-1]))
    parent.__setattr__(path[-1], new_module)
    
def from_pretrained(pretrained_model_name_or_path : str, 
                    worker_devices : Dict[str, Dict[int, str]],
                    no_split_module_classes : Optional[List[str]] = None,
                    wait : bool = False,
                    balanced : bool = False,
                    ):
    config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path)
    with init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_config(config)
        
    worker_devices2 : Dict[rpc.WorkerInfo, Dict[int,str]] = {rpc.get_worker_info(wi): d for wi,d in worker_devices.items()}
    
    all_devices : List[Tuple[rpc.WorkerInfo,int]] = [(worker,dev) for worker, devices in worker_devices2.items() for dev in devices.keys()]
    
    pseudo_idxs : Dict[int, Tuple[rpc.WorkerInfo,int]] = {i: dev_tuple for i, dev_tuple in enumerate(all_devices)}
    
    max_memory = {i: worker_devices2[worker][dev] for i, (worker,dev) in pseudo_idxs.items()}
    
    if balanced:
        max_memory = get_balanced_memory(model, max_memory, no_split_module_classes = no_split_module_classes)
        
    device_map = infer_auto_device_map(model, max_memory = max_memory, no_split_module_classes = no_split_module_classes)
    
    assert all([type(dev) is int for dev in device_map.values()]), f"the model did not fit into GPUs"
    
    parts : Dict[rpc.WorkerInfo, rpc.RRef] = {}
    
    for worker in worker_devices2:
        local_device_map = {submodule : pseudo_idxs[i][1] if pseudo_idxs[i][0] == worker else "disk" for submodule, i in device_map.items()}
        parts[worker] = rpc.remote(worker, r_partially_load, args=(pretrained_model_name_or_path, local_device_map))
   
    if wait:
        for worker in worker_devices2:
            rpc.rpc_sync(worker, r_wait, args=(parts[worker],))
        
    for submodule, pseudo_idx in device_map.items():
        worker, device = pseudo_idxs[pseudo_idx]
        rref = rpc.remote(worker, r_get_part, args=(parts[worker], submodule, device))
        if submodule == '':
            model = RRefModule(rref, worker, name=type(model.get_submodule(submodule)).__name__, device=device)
        else:
            set_submodule(model, submodule, RRefModule(rref, worker, name=type(model.get_submodule(submodule)).__name__, device=device))
        
    return model
