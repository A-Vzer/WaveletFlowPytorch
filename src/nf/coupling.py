import torch
import torch.nn as nn
from utilities import split_feature
from nf.convnet import ConvNet
from nf.layers import SqueezeLayer
import numpy as np


class Affine(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, conditional):
        super().__init__()
        
        if out_channels % 2 != 0:
            out_channels += 1
            
        if conditional:
            self.block = ConvNet(in_channels // 2 + in_channels // 3, out_channels, hidden_channels)
        else:
            self.block = ConvNet(in_channels // 2, out_channels, hidden_channels)

    def get_param(self, x, conditioning):
        z1, z2 = split_feature(x, "split")
        if conditioning is not None:
            z1_c = torch.cat([z1, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = torch.sigmoid(s + 2.0)
        return s, t, z1, z2

    def forward(self, x, logdet, conditioning=None, reverse=False):
        
        s, t, z1, z2 = self.get_param(x, conditioning)
        if reverse:
            s = torch.sigmoid(s + 2.0)
            z2 = z2 / s
            z2 = z2 - t
            logdet = -torch.sum(torch.log(s), dim=[1, 2, 3]) + logdet
        else:
            s, t, z1, z2 = self.get_param(x, conditioning)
            z2 = z2 + t
            z2 = z2 * s
            logdet = torch.sum(torch.log(s), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)
        return z, logdet

class RadialMask(nn.Module):
    def __init__(self, in_channels, out_channels, h, w, hidden_channels, idx, conditional=True):
        super().__init__()
        if conditional:
            self.block = ConvNet(in_channels + 1, out_channels * 2, hidden_channels)
        else:
            self.block = ConvNet(in_channels, out_channels * 2, hidden_channels)

        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

        center = (int(w/2), int(h/2))
        radius = min(center[0], center[1], w - center[0], w - center[1])
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius // 2
        idx = (idx + 1) % 2

        if idx == 0:
            self.mask_in = nn.Parameter(torch.tensor(mask, dtype=torch.float), requires_grad=False)
            self.mask_out = nn.Parameter(torch.tensor(~mask, dtype=torch.float), requires_grad=False)
        elif idx == 1:
            self.mask_in = nn.Parameter(torch.tensor(~mask, dtype=torch.float), requires_grad=False)
            self.mask_out = nn.Parameter(torch.tensor(mask, dtype=torch.float), requires_grad=False)

    def get_param(self, x, conditioning):
        z1 = x * self.mask_in
        z2 = x * (1 - self.mask_in)
        if conditioning is not None:
            z1_c = torch.cat([z1, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * self.mask_out
        t = t * self.mask_out
        return s, t, z1, z2

    def forward(self, x, logdet, conditioning=None, reverse=False):
        s, t, z1, z2 = self.get_param(x, conditioning)
        exp_s = s.exp()
        if reverse:
            z = x * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet
        else:
            z2 = (z2 + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        z = z1 * self.mask_in + z2 * (1 - self.mask_in)
        return z, logdet

class HorizontalChain(nn.Module):
    def __init__(self, in_channels, out_channels, h, w, hidden_channels, idx, conditional=True):
        super().__init__()
        if conditional:
            self.block = ConvNet(in_channels + 1, out_channels * 2, hidden_channels)
        else:
            self.block = ConvNet(in_channels, out_channels * 2, hidden_channels)

        self.rescale = nn.utils.weight_norm(Rescale(in_channels))
        split_h = h // 8
        idx = (idx + 1) % 8
        self.mask_in = nn.Parameter(torch.zeros((1, in_channels, h, w)), requires_grad=False)
        self.mask_out = nn.Parameter(torch.zeros((1, in_channels, h, w)), requires_grad=False)
        if idx != 7:
            self.mask_in[:, :, idx*split_h:(idx+1)*split_h, :] = 1
            self.mask_out[:, :, (idx+1)*split_h:(idx+2)*split_h, :] = 1
        else:
            self.mask_in[:, :, idx*split_h:(idx+1)*split_h, :] = 1
            self.mask_out[:, :, 0:split_h, :] = 1

    def get_param(self, x, conditioning):
        z1 = x * self.mask_in
        z2 = x * (1 - self.mask_in)
        if conditioning is not None:
            z1_c = torch.cat([z1, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * self.mask_out
        t = t * self.mask_out
        return s, t, z1, z2

    def forward(self, x, logdet, conditioning=None, reverse=False):
        s, t, z1, z2 = self.get_param(x, conditioning)
        exp_s = s.exp()
        if reverse:
            z = x * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet
        else:
            z2 = (z2 + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        z = z1 * self.mask_in + z2 * (1 - self.mask_in)
        return z, logdet

class CycleMask(nn.Module):
    def __init__(self, in_channels, out_channels, h, w, hidden_channels, idx, conditional=True):
        super().__init__()
        if conditional:
            self.block = ConvNet(in_channels + 1, out_channels * 2, hidden_channels)
        else:
            self.block = ConvNet(in_channels, out_channels * 2, hidden_channels)

        self.rescale = nn.utils.weight_norm(Rescale(in_channels))
        split_h = h // 2
        split_w = w // 2
        idx = (idx + 1) % 4

        self.mask_in = nn.Parameter(torch.zeros((1, in_channels, h, w)), requires_grad=False)
        self.mask_out = nn.Parameter(torch.zeros((1, in_channels, h, w)), requires_grad=False)
        if idx == 0:
            self.mask_in[:, :, :split_h, :split_w] = 1
            self.mask_out[:, :, :split_h, split_w:] = 1
        elif idx == 1:
            self.mask_in[:, :, :split_h, split_w:] = 1
            self.mask_out[:, :, split_h:, split_w:] = 1
        elif idx == 2:
            self.mask_in[:, :, split_h:, split_w:] = 1
            self.mask_out[:, :, split_h:, :split_w] = 1
        elif idx == 3:
            self.mask_in[:, :, split_h:, :split_w] = 1
            self.mask_out[:, :, :split_h, :split_w] = 1

    def get_param(self, x, conditioning):
        z1 = x * self.mask_in
        z2 = x * (1 - self.mask_in)
        if conditioning is not None:
            z1_c = torch.cat([z1, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * self.mask_out
        t = t * self.mask_out
        return s, t, z1, z2

    def forward(self, x, logdet, conditioning=None, reverse=False):
        s, t, z1, z2 = self.get_param(x, conditioning)
        exp_s = s.exp()
        if reverse:
            z = x * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet
        else:
            z2 = (z2 + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        z = z1 * self.mask_in + z2 * (1 - self.mask_in)
        return z, logdet

class Checkerboard(nn.Module):
    def __init__(self, in_channels, out_channels, H, W, hidden_channels, conditional=True):
        super().__init__()
        if conditional:
            self.block = ConvNet(in_channels + 1, out_channels * 2, hidden_channels)
        else:
            self.block = ConvNet(in_channels, out_channels * 2, hidden_channels)

        checkerboard = [[((i % 2) + j) % 2 for j in range(W)] for i in range(H)]
        self.mask = torch.tensor(checkerboard, requires_grad=False).view(1, 1, H, W)
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

    def get_param(self, x, conditioning):
        self.mask = self.mask.to(x.get_device())
        z1 = x * self.mask
        if conditioning is not None:
            z1_c = torch.cat([z1, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        return s, t

    def forward(self, x, logdet, conditioning=None, reverse=False):
        s, t = self.get_param(x, conditioning)
        exp_s = s.exp()
        if reverse:
            z = x * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet
        else:
            z = (x + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        return z, logdet



class Checkerboard3D(nn.Module):
    def __init__(self, in_channels, out_channels, H, W, hidden_channels, conditional=True):
        super().__init__()
        if conditional:
            self.block = ConvNet(in_channels + 1, out_channels * 2, hidden_channels)
        else:
            self.block = ConvNet(in_channels, out_channels * 2, hidden_channels)

        checkerboard = np.indices((in_channels, H, W)).sum(axis=0) % 2
        self.mask = torch.tensor(checkerboard, requires_grad=False).view(1, in_channels, H, W)
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))

    def get_param(self, x, conditioning):
        self.mask = self.mask.to(x.get_device())
        z1 = x * self.mask
        if conditioning is not None:
            z1_c = torch.cat([z1, conditioning], dim=1)
            h = self.block(z1_c)
        else:
            h = self.block(z1)
        s, t = split_feature(h, "cross")
        s = self.rescale(torch.tanh(s))
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        return s, t

    def forward(self, x, logdet, conditioning=None, reverse=False):
        s, t = self.get_param(x, conditioning)
        exp_s = s.exp()
        if reverse:
            z = x * exp_s - t
            logdet = -torch.sum(s, dim=[1, 2, 3]) + logdet + logdet
        else:
            z = (x + t) * exp_s
            logdet = torch.sum(s, dim=[1, 2, 3]) + logdet
        return z, logdet

class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x