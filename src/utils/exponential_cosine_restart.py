import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.optim import Optimizer
from functools import partial

class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1,
                    eta_min=0, last_epoch=-1, verbose=False, decay=5):
        super().__init__(optimizer, T_0, T_mult=T_mult,
                            eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)
        self.decay = decay
        self.initial_lrs = self.base_lrs
    
    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0
            
            self.base_lrs = [initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs]

        super().step(epoch)
        
        
from transformers import get_cosine_schedule_with_warmup
from math import cos, pi

def get_combined_schedule(optimizer, warmup_steps, total_steps, num_restarts, cosine_init_lr, min_lr, decay_rate):
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    steps_per_cycle = (total_steps - warmup_steps) // num_restarts

    def get_lr(current_step):
        if current_step < warmup_steps:
            return cosine_init_lr * (current_step / warmup_steps)
        else:
            cycle_progress = (current_step % steps_per_cycle) / steps_per_cycle
            current_lr = min_lr + 0.5 * (cosine_init_lr - min_lr) * (1 + cos(pi * cycle_progress))
            if current_step % steps_per_cycle == 0:
                cosine_init_lr *= decay_rate
            return current_lr

    return get_lr


def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int, restart_decay: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0
    
    restart_step = num_training_steps // num_cycles
    cycle = (current_step - num_warmup_steps) // restart_step
    cycle_progress = (current_step - num_warmup_steps) % restart_step / restart_step
    cosine_lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycle_progress)))
    
    restart_lr = restart_decay ** cycle
    # final_lr = cosine_lr * restart_lr
    # if final
    return cosine_lr * restart_lr

def get_cosine_with_hard_restarts_schedule_with_warmupx(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1, restart_decay: float = 1.0
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        restart_decay=restart_decay,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)
