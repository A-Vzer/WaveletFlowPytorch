from importlib import machinery
import mailcap
from operator import inv
import torch
import torch.nn as nn

class Dwt(nn.Module):
    def __init__(self, wavelet, compensate=True):
        super().__init__()
        self.wavelet = wavelet
        self.kernel = None
        self.inv_kernel = None
        self.f = self.wavelet.factor
        self.m = self.wavelet.multiplier
        self.compensate = compensate

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
        
        if self.compensate:
            low_shape = low_freq.shape
            c = low_shape[1]
            h = low_shape[2]
            w = low_shape[3]
            low_freq = low_freq * 0.5

        components = {"low": low_freq, "high": high_freq}

        return components
    
    def inverse(self, high_freq, low_freq):
        reconstruction = []
        ldj = 0

        if self.compensate:
            low_freq = low_freq * 2.0
            low_shape = low_freq.shape
            c = low_shape[1]
            h = low_shape[2]
            w = low_shape[3]
            ldj = torch.log(0.5) * c * h * w

        for i in range(low_freq.shape[1]):
            reconstruction.append(low_freq[:, i:i+1, :, :])
            reconstruction.append(high_freq[:, i*3:i*3+3, :, :])

        y = torch.cat(reconstruction, dim=1)
        C = y.shape[1]
        H = y.shape[2]
        W = y.shape[3]

        assert C >= self.m and C % self.m == 0, '({}) channels must be divisible by {}'.format(C, self.f)

        x = nn.functional.conv2d(y, self.make_inverse_kernel(C // self.m), (1,1), 'same')
        x = torch.reshape(x, (-1, C // (self.f**2), self.f, self.f, H, W))
        x = torch.transpose(x, (0, 1, 4, 2, 5, 3))
        x = torch.reshape(x, (-1, C / (self.f**2), H * self.f, W * self.f))
        
        components = {"rec": x, "logdet": ldj}

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

    def make_inverse_kernel(self, C):
        if self.inv_kernel is not None:
            return self.inv_kernel

        k = self.wavelet.kernels
        inv_kernel = []
        for i in range(C):
            front_padding = [0.0,0.0,0.0,0.0]*i
            back_padding = [0.0,0.0,0.0,0.0]*(C-i-1)

            row = front_padding + k[0] + back_padding
            inv_kernel.append(row)
            row = front_padding + k[1] + back_padding
            inv_kernel.append(row)
            row = front_padding + k[2] + back_padding
            inv_kernel.append(row)
            row = front_padding + k[3] + back_padding
            inv_kernel.append(row)

        inv_kernel = torch.transpose(torch.tensor(inv_kernel),0,1)
        inv_kernel = inv_kernel.unsqueeze(-1).unsqueeze(-1)

        self.inv_kernel = inv_kernel

        return self.inv_kernel