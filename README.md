# rref_module

A small library for running [`torch`](https://pytorch.org/) models on multiple nodes with multiple GPUs based on [`torch.distributed.rpc`](https://pytorch.org/docs/stable/rpc.html)

Includes a tool for loading [HuggingFace](https://huggingface.co/) models, but only tested on [facebook's OPT](https://huggingface.co/facebook/opt-30b).

## Example using `rref_module.remote_copy`
```python
import torch
import torch.distributed.rpc as rpc
import fire #type: ignore
from rref_module import remote_copy
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd

def main(rank : int, world_size : int, gpus : int = 4):
    gpus = 4
    options = rpc.TensorPipeRpcBackendOptions(
        rpc_timeout=1000000,
        num_worker_threads=128,
        device_maps={f"worker{i}" : {j : j for j in range(gpus)} for i in range(world_size)}
        )

    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)
    if rank == 0:
        with dist_autograd.context() as context_id:
            n = 5000
            t = torch.rand((1, n))

            module = torch.nn.Linear(n,n)
            module2 = torch.nn.Linear(n,n, bias=False)
            rmodule = remote_copy(module, 'worker1', device=1)
            rmodule2 = remote_copy(module2, 'worker2', device=2)
            x = rmodule(t)
            for _ in range(5):
                x = rmodule2(x)
                x = rmodule(x)
            y = x.to_here()
            loss = y.sum()
            dist_autograd.backward(context_id, [loss])
           
            dist_optim = DistributedOptimizer(
                torch.optim.SGD,
                rmodule.rref_parameters() + rmodule2.rref_parameters(),
                lr=0.05,
            )
            dist_optim.step(context_id)
    rpc.shutdown()
    
if __name__ == "__main__":
    fire.Fire(main)
```

## Examples using `rref_module.from_pretrained`

```python
import torch
import rref_module
import torch.distributed.rpc as rpc
import transformers #type: ignore
import fire #type: ignore
from typing import *

def main(rank : int, world_size : int, gpus : int = 4):
    worker_devices = {f'worker{i}' : {j : '64gb' for j in range(gpus)} for i in range(1,world_size)}
        # I'm lying about the size and it fits, when with real size it puts data on disk
        # probably huggingface assumes I will be using entire context
    
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=512,
        rpc_timeout=1000000,
        device_maps={worker : {j : j for j in range(len(devices))} for worker, devices in worker_devices.items()}, 
    )
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)
    
    if rank == 0:
        model_name = "facebook/galactica-30b"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = rref_module.from_pretrained(model_name, wait = True, balanced = False,
                    worker_devices = worker_devices, no_split_module_classes=["OPTDecoderLayer"])
        print(model)
        
        text = "Mike is taller than Sally.\nQuestion: Is Sally taller than Mike?\nAnswer: "
        input_ids = tokenizer(text, return_tensors='pt').input_ids

        for _ in range(50):
            model_kwargs = {'input_ids':input_ids}
            
            outputs = model(
                    **model_kwargs,
                    output_attentions=True,
                    output_hidden_states=True,
                )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            #this is changed from default HF implementation
            #note: calling `argmax` method instread of `torch.argmax` (so it's executed remotely)
            next_tokens = next_token_logits.argmax(dim = -1).to_here().cpu()
            
            print(f'{next_tokens=}')
            
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs)
            
        print('generated: ', tokenizer.decode(input_ids[0]))
        
    rpc.shutdown()
    
if __name__ == "__main__":
    fire.Fire(main)
```