import os
import torch
import random
import numpy as np
from data_loader import ISIC
from importlib.machinery import SourceFileLoader
from torch.utils.data import DataLoader
import torch.optim as optim
from src.nf.glow import Glow
from src.waveletflow import WaveletFlow
from src.conditioning_network import Conditioning_network
import argparse
import time

seed = 786
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.autograd.set_detect_anomaly(True)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
device = torch.device("cuda")
torch.cuda.empty_cache()



def main():
    fName = time.strftime("%Y%m%d_%H_%M")
    if not os.path.exists("saves/"):
        os.makedirs("saves/")
   
    cf = SourceFileLoader('cf', f'{args.data}.py').load_module()
    dataset = ISIC(cf, benign=True, test=False, gray=cf.grayscale)
    loader = DataLoader(dataset, batch_size=cf.batch_size, shuffle=False, pin_memory=True)
    if args.model == "glow":
        p = SourceFileLoader('cf', 'config_glow.py').load_module()
        model = Glow(p).to(device)
    elif args.model == "waveletflow":
        p = SourceFileLoader('cf', 'config_waveletflow.py').load_module()
        model = WaveletFlow(cf=p, cond_net=Conditioning_network(), partial_level=p_level).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    
    lowest = 1e7
    patience = 0
    model.train()
    ep = 1

    while True:
        ep_loss = []
        for idx, x in enumerate(loader):
            print(f"Epoch: {ep} Progress:      {round((idx * 100) / (len(loader)), 4)}% Likelihood:      {lowest} Patience:      {patience}" , end="\r")
            x = x.to(device)
            optimizer.zero_grad()
            res_dict = model(x, partial_level=p_level)
            loss = torch.mean(res_dict["likelihood"])
            loss.backward()
            optimizer.step()
            loss_ = loss.detach().cpu().numpy()
            ep_loss.append(loss_)
        avg_loss = round(float(np.mean(ep_loss)), 2)
        if lowest > avg_loss:    
            lowest = avg_loss
            torch.save(model.state_dict(), f'saves/{fName}-{args.model}-{args.data}-{args.level}-test.pt')
            patience = 0
        else:
            patience += 1
        ep += 1
        if patience == 10:
            break
        break
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="glow", help='train level')
parser.add_argument('--level', type=int, default=-1, help='train level')
parser.add_argument('--data', type=str, default="isic", help='train level')
args = parser.parse_args()
p_level = args.level
main()