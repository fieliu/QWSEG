import math

import torch
import torch.nn as nn


class LoRALinear(nn.Linear):

    def __init__(self, in_features, out_features, rank=4, alpha=1.0,
                 dropout=0.0, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        original_out = super().forward(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return original_out + lora_out


def apply_lora_to_model(model, rank=4, alpha=1.0, dropout=0.0,
                        target_modules=None):
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj', 'proj']

    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        should_apply = any(t in name for t in target_modules)
        if not should_apply:
            continue

        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        replacements.append((parent_name, child_name, module))

    for parent_name, child_name, module in replacements:
        parent = model
        for attr in parent_name.split('.'):
            if attr:
                parent = getattr(parent, attr)

        lora_layer = LoRALinear(
            module.in_features,
            module.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=module.bias is not None)
        lora_layer.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            lora_layer.bias.data.copy_(module.bias.data)
        setattr(parent, child_name, lora_layer)

    return model


def get_lora_params(model):
    lora_params = []
    non_lora_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        else:
            non_lora_params.append(param)
    return lora_params, non_lora_params


def freeze_non_lora_params(model):
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def merge_lora_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_weight = (module.lora_B.data @ module.lora_A.data) * module.scaling
            module.weight.data += lora_weight.t()
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()
