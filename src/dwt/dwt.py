from importlib import machinery
import mailcap
from operator import inv
import torch
import torch.nn as nn

class Dwt(nn.Module):
    def __init__(self, wavelet):
        super().__init__()
        self.wavelet = wavelet
        self.kernel = None
        self.inv_kernel = None
        self.f = self.wavelet.factor
        self.m = self.wavelet.multiplier

    def forward(self, x):
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
        H_w = self.wavelet.h
        W_w = self.wavelet.w
        high_freq = []
        low_freq = []
        ldj = 0

        assert H % H_w == 0 and W % W_w == 0, '({},{}) not dividing by {} nicely'.format(H, W, self.f)
        forward_kernel = self.make_forward_kernel(C).to(x.get_device())
        y = nn.functional.conv2d(x, forward_kernel, None, (2,2), 'valid')
        for i in range(C):
            low_freq.append(y[:, i*self.m:i*self.m+1, : ,:])
            high_freq.append(y[:, i*self.m+1:i*self.m+self.m, : ,:])
        
        high_freq = torch.cat(high_freq, dim=1)
        low_freq = torch.cat(low_freq, dim=1)

        components = {"low": low_freq, "high": high_freq}

        return components

    def make_forward_kernel(self, C):
        if self.kernel is not None:
            return self.kernel
        
        H_w = self.wavelet.h
        W_w = self.wavelet.w
        k = self.wavelet.kernels

        kernel = torch.zeros((C*self.m, C, H_w, W_w))
        
        for i in range(C):
            for j in range(self.m):
                kernel[i*self.m+j, i, :, :] = torch.tensor(k[j])
        
        self.kernel = kernel
        return kernel
        
