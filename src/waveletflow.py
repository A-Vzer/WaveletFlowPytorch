import torch.nn as nn
from src.dwt.dwt import Dwt
from src.dwt.wavelets import Haar
from src.nf.glow import Glow
import math
import torch
import numpy as np

class WaveletFlow(nn.Module):
    def __init__(self, cf, cond_net, partial_level=-1):
        super().__init__()
        self.n_levels = cf.nLevels
        self.base_level = cf.baseLevel
        self.partial_level = partial_level
        self.wavelet = Haar()
        self.dwt = Dwt(wavelet=self.wavelet)
        self.conditioning_network = cond_net
        
        if partial_level == -1 or partial_level == self.base_level:
            base_size = 2 ** self.base_level
            cf.K = cf.stepsPerResolution[partial_level]
            cf.L = cf.stepsPerResolution_L[partial_level]
            shape = (cf.imShape[0], 1, 1)
            self.base_flow = Glow(cf, shape, False)
        else:
            self.base_flow = None
        
        start_flow_padding = [None] * self.base_level
        self.sub_flows = start_flow_padding + [self.base_flow]
        
        for level in range(self.base_level + 1, self.n_levels + 1):
            if partial_level != -1 and partial_level != level:
                self.sub_flows.append(None)
            else:
                h = 2**(level-1)
                w = 2**(level-1)
                cf.K = cf.stepsPerResolution[level-1]
                cf.L = cf.stepsPerResolution_L[level-1]
                shape = (cf.imShape[0] * 3, h, w)
                self.sub_flows.append(Glow(cf, shape, cf.conditional))

        self.sub_flows = nn.ModuleList(self.sub_flows)

    def forward(self, x, partial_level=-1):

        latents = []
        low_freq = x 
        for level in range(self.n_levels, self.base_level-1, -1):
            if level == partial_level or partial_level == -1:
                if level == self.base_level:
                    flow = self.base_flow
                    conditioning = None
                    res = flow.forward(dwt_components['low'], conditioning=conditioning)
                else:
                    dwt_components = self.dwt.forward(low_freq)
                    low_freq = dwt_components['low']
                    conditioning = self.conditioning_network.encoder_list[level](low_freq)
                    flow = self.sub_flows[level]
                    res = flow.forward(dwt_components['high'], conditioning=conditioning)

                latents.append(res["latent"])
                b, c, h, w = low_freq.shape
                res["likelihood"] -= (c * h * w * torch.log(torch.tensor(0.5)) * (self.n_levels - level)) /  (math.log(2.0) * c * h * w)
                x = torch.abs(dwt_components['high'])
                if partial_level != -1:
                    break 
            
            else:
                if self.partial_level <= 8 and level > 8:
                    pass
                else:
                    dwt_components = self.dwt.forward(low_freq)
                    low_freq = dwt_components['low']
                latents.append(None)

        return {"latent":latents, "likelihood":res["likelihood"]}

