import torch.nn as nn


class Haar(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = 2
        self.w = 2
        self.multiplier = 4
        self.factor = 2
        self.kernels = [[[0.5,0.5],[0.5,0.5]],
                        [[0.5,-0.5],[0.5,-0.5]],
                        [[0.5,0.5],[-0.5,-0.5]],
                        [[0.5,-0.5],[-0.5,0.5]]]