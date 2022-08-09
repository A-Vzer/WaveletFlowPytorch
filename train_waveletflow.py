import os
import torch
import random
import numpy as np
from data_loader import ISIC
from importlib.machinery import SourceFileLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from src.waveletflow import WaveletFlow
from src.conditioning_network import Conditioning_network
import argparse

seed = 786
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda")

def main():
    cf = SourceFileLoader('cf', f'isic.py').load_module()
    p = SourceFileLoader('cf', f'config.py').load_module()
    dataset = ISIC(cf, benign=True, test=False, gray=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True)
    model = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=p_level).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    lowest = 1e7
    patience = 0
    model.train()

    while True:
        ep_loss = []
        for idx, x in enumerate(loader):
            x = x.to(device)
            optimizer.zero_grad()
            z, bpd = model(x, partial_level=p_level)
            loss = torch.mean(bpd)
            loss.backward()
            optimizer.step()
            loss_ = loss.detach().cpu().numpy()
            ep_loss.append(loss_)
        avg_loss = np.round_(np.mean(ep_loss), 2)
        if lowest > avg_loss:    
            lowest = avg_loss
            print(lowest, end = "\r")
            torch.save(model.state_dict(), f'saves/wf-{p.coupling}/wf-conditional-{p.conditional}-{p_level}-gray-{p.stepsPerResolution[p_level]}.pt')
            patience = 0
        else:
            patience += 1
            print(f"Patience: {patience}", end = "\r")
        if patience == 10:
            break

parser = argparse.ArgumentParser()
parser.add_argument('--level', type=int, default=0, help='train level')
args = parser.parse_args()
p_level = args.level
main()