import torch
from typing import List
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler

#---------------------------------------scheduler---------------------------------------------#
class WarmupMultiStepLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        """
        Multi Step LR with warmup

        Args:
            optimizer (torch.optim.Optimizer): optimizer_scheduler used.
            milestones (list[Int]): a list of increasing integers.
            gamma (float): gamma
            warmup_factor (float): lr = warmup_factor * base_lr
            warmup_iters (int): iters to warmup
            warmup_method (str): warmup method in ["constant", "linear", "burnin"]
            last_epoch(int):  The index of last epoch. Default: -1.
        """
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}",
                milestones,
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor)
        return [
            base_lr * warmup_factor * self.gamma**bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()

def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int,
                               warmup_factor: float) -> float:

    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    elif method == "burnin":
        return (iter / warmup_iters)**4
    else:
        raise ValueError("Unknown warmup method: {}".format(method))

#---------------------------------------scheduler---------------------------------------------#