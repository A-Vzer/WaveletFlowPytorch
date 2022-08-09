import torch.nn as nn
from src.dwt.dwt import Dwt
from src.dwt.wavelets import Haar
from src.nf.glow.model2 import Glow
import math
import torch
import numpy as np

class WaveletFlow(nn.Module):
    def __init__(self, cf, cond_net, partial_level=-1):
        super().__init__()

        self.n_steps = cf.stepsPerResolution
        self.n_steps_L = cf.stepsPerResolution_L
        self.n_levels = cf.nLevels
        self.base_level = cf.baseLevel
        self.partial_level = partial_level
        self.conditional = cf.conditional
        self.wavelet = Haar()
        self.dwt = Dwt(wavelet=self.wavelet)
        self.conditioning_network = cond_net
        self.cf = cf

        if partial_level == -1 or partial_level == self.base_level:
            base_size = 2 ** self.base_level
            self.cf.K = self.n_steps[0]
            self.cf.L = self.n_steps_L[0]
            self.base_flow = Glow(self.cf, (cf.imShape[0] * 3, 1, 1), False)
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
                self.cf.K = self.n_steps[level-1]
                self.cf.L = self.n_steps_L[level-1]
                self.sub_flows.append(Glow(self.cf, (cf.imShape[0] * 3, h, w), cf.conditional))

        self.sub_flows = nn.ModuleList(self.sub_flows)

    def forward(self, x, partial_level=-1):

        latents = []
        bpd = 0
        low_freq = x 
        for level in range(self.n_levels, self.base_level-1, -1):
            if level == partial_level or partial_level == -1:
                if level == self.base_level:
                    flow = self.base_flow
                    conditioning = None
                    latent, logdet, _ = flow.forward(dwt_components['high'], conditioning=conditioning)
                else:
                    dwt_components = self.dwt.forward(low_freq)
                    low_freq = dwt_components['low']
                    if self.conditional:
                        conditioning = self.conditioning_network.encoder_list[level](low_freq)
                    else:
                        conditioning = None
                    flow = self.sub_flows[level]
                    latent, logdet, _ = flow.forward(dwt_components['high'], conditioning=conditioning)

                latents.append(latent)
                b, c, h, w = low_freq.shape
                correction =  c * h * w * torch.log(torch.tensor(0.5)) * (self.n_levels - level)
                bpd += -(logdet + correction) / (math.log(2.0) * c * h * w)
                x = torch.abs(dwt_components['high'])
                if not self.training:
                    bpd -=  torch.sum(torch.mean(torch.abs(x), dim=(2,3)), dim=1)
                if partial_level != -1:
                    break # stop of we are doing partial
            
            else:
                # decompose base
                if self.partial_level <= 8 and level > 8:
                    pass
                else:
                    dwt_components = self.dwt.forward(low_freq)
                    low_freq = dwt_components['low']
                latents.append(None)

        return latents, bpd


    def inverse(self, latents):
        ldj = 0

        base, ldj_b = self.base_flow.forward(reverse=True)
        ldj += ldj_b
        reconstructions = []
        details = []

        for level in range(self.base_level+1 ,self.n_levels+1):
            low_fr = reconstructions[-1]
            flow = self.sub_flows[level]
            conditioning = self.conditioning_network.encoder_list[level](low_fr)
            high_freq, ldj = flow.forward(z, conditioning=conditioning, reverse=True)
            rec, ldj_haar = self.dwt.inverse(high_freq, low_fr)
            ldj += ldj
            reconstructions.append(rec)
            details.append(high_freq)
        
        return reconstructions, ldj
