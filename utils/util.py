
from utils.config import  Config
import torch.nn as nn
from typing import Tuple, List
import torch
from transformers import AdamW, get_linear_schedule_with_warmup, BartForConditionalGeneration
import json
import math

def read_data(file:str) -> List:
    with open(file, "r", encoding='utf-8') as read_file:
        data = json.load(read_file)
    return data


def get_optimizers(config: Config, model: nn.Module, num_training_steps: int, weight_decay:float = 0.01,
                   warmup_step: int = -1, eps:float = 1e-8) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    # no_decay = ["bias", "LayerNorm.weight", 'LayerNorm.bias']
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=eps)
                      # , correct_bias=False)
    warmup_step = warmup_step if warmup_step >= 0 else int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps
    )
    return optimizer, scheduler


EPSILON = 1e-6


def clip_gradients(model, gradient_clip_val, device):
    # this code is a modification of torch.nn.utils.clip_grad_norm_
    # with TPU support based on https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md
    if gradient_clip_val > 0:
        parameters = model.parameters()
        max_norm = float(gradient_clip_val)
        norm_type = float(2.0)
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        if norm_type == math.inf:
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            device = parameters[0].device
            total_norm = torch.zeros([], device=device if parameters else None)
            for p in parameters:
                param_norm = p.grad.data.pow(norm_type).sum()
                total_norm.add_(param_norm)
            total_norm = (total_norm ** (1. / norm_type))
        eps = EPSILON
        clip_coef = torch.tensor(max_norm, device=device) / (total_norm + eps)
        for p in parameters:
            p.grad.data.mul_(torch.where(clip_coef < 1, clip_coef, torch.tensor(1., device=device)))



def write_data(file:str, data) -> None:
    with open(file, "w", encoding="utf-8") as write_file:
        # json_results = json.dumps(results)
        # print(json_results)
        json.dump(data, write_file, ensure_ascii=False, indent=4)



class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.model = module  # that I actually define.

    def forward(self, x):
        return self.module(x)

def extract_module(model_file: str = "model_files/mtl/model.m"):
    dev = torch.device('cpu')
    model = BartForConditionalGeneration.from_pretrained('bart-base').to(dev)
    model = WrappedModel(model) ## because we trained with multi -GPU
    model.load_state_dict(torch.load(model_file, map_location=dev))
    torch.save(model.model.state_dict(), f"model_files/mtl/mtl_model.m")
