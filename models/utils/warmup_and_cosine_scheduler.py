#  ---------------------------------------------------------------
#  Copyright (c) 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
#  Licensed under the MIT License
#  ---------------------------------------------------------------
#


import math
import numpy as np
from logging import getLogger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = getLogger()


class WarmupAndCosineScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, start_warmup_lr, warmup_iters, base_lr, final_lr, total_iters):
        warmup_lr_schedule = np.linspace(start_warmup_lr, base_lr, warmup_iters)
        iters = np.arange(total_iters - warmup_iters + 1)
        cosine_lr_schedule = []
        for t in iters:
            tmp_lr = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + math.cos(math.pi * t / (total_iters - warmup_iters)))
            cosine_lr_schedule.append(tmp_lr)
        cosine_lr_schedule = np.array(cosine_lr_schedule)

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch=-1, verbose=False)
        logger.info("Building LR scheduler done.")

    def get_lr(self):
        if self.last_epoch <= self.total_iters:
            i = self.last_epoch
        elif self.last_epoch < 0:
            i = 0
        else:
            i = self.last_epoch
            logger.warning("WarmupAndCosineScheduler: iter overflow: {}/{}".format(self.last_epoch,
                                                                                   self.total_iters))

        return [base_lr * self.lr_schedule[i] for base_lr in self.base_lrs]

    # def get_lr(self):
    #
    #     if self.last_epoch == 0 or self.last_epoch > self.total_iters:
    #         return [group["lr"] for group in self.optimizer.param_groups]
    #
    #     decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
    #     return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.lr_schedule)), self.lr_schedule)
        plt.savefig('WarmupAndCosineScheduler.png')
        plt.close()
