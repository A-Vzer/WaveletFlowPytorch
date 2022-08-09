import os, sys
sys.path.append(os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import torch
import torch.nn as nn
from utilities import split_feature, uniform_binning_correction
from nf.coupling import *
from nf.layers import *
import numpy as np

class FlowStep(nn.Module):
    def __init__(self, params, C, H, W, idx, conditional):
        super().__init__()
        hidden_channels = params.hiddenChannels
        self.flow_permutation = params.perm
        self.flow_coupling = params.coupling

        self.actnorm = ActNorm2d(C, params.actNormScale)

        # 2. permute
        if self.flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(C, LU_decomposed=params.LU)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif self.flow_permutation == "shuffle":
            self.shuffle = Permute2d(C, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (self.shuffle(z, rev), logdet,)
        else:
            self.reverse = Permute2d(C, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (self.reverse(z, rev), logdet,)

        # 3. coupling
        if params.coupling == 'affine':
            self.coupling = Affine(C, C, hidden_channels, conditional)
        elif params.coupling == 'checker':
            self.coupling = Checkerboard(C, C, H, W, hidden_channels, conditional)
        elif params.coupling == 'checker3d':
            self.coupling = Checkerboard3D(C, C, H, W, hidden_channels, conditional)
        elif params.coupling == 'cycle':
            self.coupling = CycleMask(C, C, H, W, hidden_channels, idx, conditional)
        elif params.coupling == 'radial':
            self.coupling = RadialMask(C, C, H, W, hidden_channels, idx, conditional)
        elif params.coupling == 'horizontal':
            self.coupling = HorizontalChain(C, C, H, W, hidden_channels, idx, conditional)
    
    def forward(self, input, conditioning, logdet, reverse=False):
        if not reverse:
            return self.normal_flow(input, conditioning, logdet)
        else:
            return self.reverse_flow(input, conditioning, logdet)

    def normal_flow(self, input, conditioning, logdet):
        # assert input.size(1) % 2 == 0
        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet)
        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)
        # 3. coupling
        z, logdet = self.coupling(z, logdet, conditioning)
        return z, logdet

    def reverse_flow(self, input, conditioning, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z, logdet = self.coupling(input, conditioning, logdet, reverse=True)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(self, params, shape, conditional):
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = params.K
        self.L = params.L
        C, H, W = shape

        for idx in range(self.K):
            self.layers.append(FlowStep(params, C, H, W, idx, conditional))
            self.output_shapes.append([-1, C, H, W])

    def forward(self, input, conditioning, logdet=None, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, conditioning, temperature)
        else:
            return self.encode(input, conditioning, logdet)

    def encode(self, z, conditioning, logdet):
        for layer, shape in zip(self.layers, self.output_shapes):
            if isinstance(layer, SqueezeLayer):
                z, logdet = layer(z, logdet, reverse=False)
            elif isinstance(layer, FlowStep):
                z, logdet = layer(z, conditioning, logdet, reverse=False)
            else:
                z, logdet = layer(z, logdet, reverse=False)
            
        return z, logdet

    def decode(self, z, conditioning, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, conditioning, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, conditioning, logdet=0, reverse=True)
        return z, logdet


class Glow(nn.Module):
    def __init__(self, params, shape, conditional=False):
        super().__init__()
        self.flow = FlowNet(params, shape, conditional)
        self.y_classes = params.y_classes
        self.y_condition = params.y_condition
        self.learn_top = params.y_learn_top
        
        # learned prior
        if self.learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        if self.y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = LinearZeros(self.y_classes, 2 * C)
            self.project_class = LinearZeros(C, self.y_classes)

        self.register_buffer("prior_h", torch.zeros(
            [1, self.flow.output_shapes[-1][1] * 2, self.flow.output_shapes[-1][2],
                self.flow.output_shapes[-1][3], ]), )

    def prior(self, data, y_onehot=None):
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            # Hardcoded a batch size of 32 here
            h = self.prior_h.repeat(32, 1, 1, 1)

        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            h += yp.view(h.shape[0], channels, 1, 1)
        return split_feature(h, "split")

    def forward(self, x=None, conditioning=None, y_onehot=None, z=None, temperature=None, reverse=False):
        if reverse:
            return self.reverse_flow(z, conditioning, y_onehot, temperature)
        else:
            return self.normal_flow(x, conditioning, y_onehot)

    def normal_flow(self, x, conditioning, y_onehot):
        b, c, h, w = x.shape

        x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, conditioning, logdet=logdet, reverse=False)
        mean, logs = self.prior(x, y_onehot)
        objective += gaussian_likelihood(mean, logs, z)
        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        # bpd = (-objective) / (math.log(2.0) * c * h * w)
        return z, objective, y_logits

    def reverse_flow(self, z, conditioning, y_onehot, temperature):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z, y_onehot)
                z = gaussian_sample(mean, logs, temperature)
            x, logdet = self.flow(z, temperature=temperature, reverse=True)
        return x, logdet

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True


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
