# ----------optimizer_scheduler---------#
from optimizer_scheduler.optimizer import D2SGDBuilder
import torch

def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer = D2SGDBuilder.build(model, cfg)
    return optimizer

# ----------optimizer_scheduler---------#