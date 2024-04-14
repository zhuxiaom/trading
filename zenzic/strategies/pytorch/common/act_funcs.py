import torch

def get(name: str, **kwargs):
    act_tbl = {
        'relu': torch.nn.ReLU(**kwargs),
        'gelu': torch.nn.GELU(**kwargs),
        'tanh': torch.nn.Tanh(**kwargs),
        'elu': torch.nn.ELU(**kwargs),
    }

    assert name in act_tbl.keys(), f"Invalid activation function name {name}!"
    return act_tbl[name]
