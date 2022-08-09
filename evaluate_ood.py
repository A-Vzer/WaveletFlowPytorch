import os
import torch
import random
import wandb
import pickle
import numpy as np
from data_loader import ISIC
from importlib.machinery import SourceFileLoader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from src.nf.glow.model2 import Glow
from src.waveletflow import WaveletFlow
from src.conditioning_network import Conditioning_network
from sklearn.decomposition import PCA


device = torch.device("cuda")
bs = 64
cf = SourceFileLoader('cf', f'isic-wf.py').load_module()
steps = 16

params = SourceFileLoader('cf', f'config_waveletflow.py').load_module()
params.stepsPerResolution = [16] * 8
params.coupling = 'affine'
dataset_b_train = ISIC(cf, benign=True, test=False, gray=True)
dataset_b_test = ISIC(cf, benign=True, test=True, gray=True)
dataset_b_test_c = ISIC(cf, benign=True, test=True, gray=False, standardize=False)
dataset_m = ISIC(cf, benign=False, gray=True)
dataset_m_c = ISIC(cf, benign=False, gray=False, standardize=False)
dataset_b_train.cf.augmentation = False
dataset_b_test.cf.augmentation = False
dataset_m.cf.augmentation = False
loader_b_train = DataLoader(dataset_b_train, batch_size=bs, shuffle=False, pin_memory=True, num_workers=8)
loader_b_test = DataLoader(dataset_b_test, batch_size=bs, shuffle=False, pin_memory=True, num_workers=8)
loader_m = DataLoader(dataset_m, batch_size=bs, shuffle=False, pin_memory=True, num_workers=8)
n_ims_b_train = len(loader_b_train) * bs
n_ims_b_test = len(loader_b_test) * bs
n_ims_m = len(loader_m) * bs
likelihoods_b_train = np.zeros((8, n_ims_b_train))
likelihoods_b_test = np.zeros((8, n_ims_b_test))
likelihoods_m = np.zeros((8, n_ims_m))
rocs = []

for p in range(0,8):
    print(p)
    model = WaveletFlow(cf=params, cond_net=Conditioning_network(), partial_level=p).to(device)
    model.load_state_dict(torch.load(f'saves/wf-{params.coupling}/wf-conditional-True-{p}-gray-{params.stepsPerResolution[0]}.pt'))
    for flow in model.sub_flows:
        if flow is not None:
            flow.set_actnorm_init()
    model.eval()
    
    with torch.no_grad():
        for idx, x in enumerate(loader_b_train):
            print((idx / len(loader_b_train)) * 100, end="\r")
            x = x.to(device)
            z, bpd = model(x, partial_level=p)
            try:
                likelihoods_b_train[p, idx*bs:(idx+1)*bs] = bpd.detach().cpu().numpy()
            except:
                bpds = bpd.detach().cpu().numpy()
                likelihoods_b_train[p, -bpds.shape[0]:] = bpds
            

        for idx, x in enumerate(loader_b_test):
            print((idx / len(loader_b_test)) * 100, end="\r")
            x = x.to(device)
            z, bpd = model(x, partial_level=p)
            try:
                likelihoods_b_test[p, idx*bs:(idx+1)*bs] = bpd.detach().cpu().numpy()
            except:
                bpds = bpd.detach().cpu().numpy()
                likelihoods_b_test[p, -bpds.shape[0]:] = bpds

        for idx, x in enumerate(loader_m):
            print((idx / len(loader_m)) * 100, end="\r")
            x = x.to(device)
            z, bpd = model(x, partial_level=p)
            try:
                likelihoods_m[p, idx*bs:(idx+1)*bs] = bpd.detach().cpu().numpy()
            except:
                bpds = bpd.detach().cpu().numpy()
                likelihoods_m[p, -bpds.shape[0]:] = bpds

with open(f'likelihoods_b_train-wf.pickle', 'wb') as handle:
    pickle.dump(likelihoods_b_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'likelihoods_b_test-wf.pickle', 'wb') as handle:
    pickle.dump(likelihoods_b_test, handle, protocol=pickle.HIGHEST_PROTOCOL)   

with open(f'likelihoods_m-wf.pickle', 'wb') as handle:
    pickle.dump(likelihoods_m, handle, protocol=pickle.HIGHEST_PROTOCOL)   

with open(f'likelihoods_b_train-wf.pickle', 'rb') as handle:
    likelihoods_b_train = pickle.load(handle)

with open(f'likelihoods_b_test-wf.pickle', 'rb') as handle:
    likelihoods_b_test = pickle.load(handle)

with open(f'likelihoods_m-wf.pickle', 'rb') as handle:
    likelihoods_m = pickle.load(handle)

ranges = [1,1,1,1,1,1,1,1]

fig, ax = plt.subplots(2,8, figsize=(8, 2.3))

for i in range(0,8):
    likelihoods_b_train_ = likelihoods_b_train[i][(likelihoods_b_train[i] >= ranges[i])]
    likelihoods_b_test_ = likelihoods_b_test[i][(likelihoods_b_test[i] >= ranges[i])]
    likelihoods_m_ = likelihoods_m[i][(likelihoods_m[i] >= ranges[i])]

    ax[0,i].hist(likelihoods_b_train_, 30, density=True, color='tab:blue', alpha=0.8, histtype='bar', ec='black', linewidth=0.5, label="Benign train")
    ax[0,i].hist(likelihoods_m_, 30, density=True, color='tab:red', alpha=0.8, histtype='bar', ec='black', linewidth=0.5, label="Malignant")
    ax[0,i].hist(likelihoods_b_test_, 30, density=True, color='tab:green', alpha=0.8, histtype='bar', ec='black', linewidth=0.5, label="Benign test")
    ax[0,i].set_yticks([])
    ax[0,i].set_xticks([])
    ax[0,i].annotate(f'Level\n{i}', xy=(45, 40), xycoords='axes points', size=6, ha='right', va='center')

    labels_positive = [1] * len(likelihoods_b_test_)
    labels_negative = [0] * len(likelihoods_m_)
    test_labels = labels_positive + labels_negative
    test_scores = np.concatenate((-likelihoods_b_test_, -likelihoods_m_))
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    ax[1, i].plot(fpr, tpr, color='darkred', label='ROC %0.2f' % auc(fpr, tpr), linewidth=1)
    x = np.linspace(0, 1, 3)
    ax[1, i].plot(x, x, 'k--', linewidth=1)
    ax[1, i].set_xlim([0.0, 1.0])
    ax[1, i].set_ylim([0.0, 1.0])
    ax[1, i].set_yticks([])
    ax[1, i].set_xticks([])
    ax[1, i].annotate(f'ROC:\n{round(auc(fpr, tpr), 2)}', xy=(36, 10), xycoords='axes points',
            size=6, ha='center', va='center',
            bbox=dict(boxstyle='round', fc='w'))

order = [0,2,1]
handles, labels = ax[0, 7].get_legend_handles_labels()
fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="center", ncol=3, fontsize=10)
fig.subplots_adjust(hspace=0.6) 
fig.text(0.5125, 0.95, "Negative likelihood", ha="center", va="center")
fig.text(0.5125, 0.04, "1-Specificity", ha="center", va="center")
fig.text(0.11, 0.74, "Frequency", ha="center", va="center", rotation=90)
fig.text(0.11, 0.26, "Sensitivity", ha="center", va="center", rotation=90)
plt.savefig(f"pics/histroc/wf-cond-{params.coupling}-level-{params.stepsPerResolution[0]}.png", dpi=600, bbox_inches='tight')

likelihoods_b_train = np.mean(likelihoods_b_train[3:], axis=0)
likelihoods_b_test = np.mean(likelihoods_b_test[3:], axis=0)
likelihoods_m = np.mean(likelihoods_m[3:], axis=0)

likelihoods_b_train = likelihoods_b_train[(likelihoods_b_train >= 1)]
likelihoods_b_test = likelihoods_b_test[(likelihoods_b_test >= 1)]
likelihoods_m = likelihoods_m[(likelihoods_m >= 1)]

d_b = {k: v for v, k in enumerate(list(likelihoods_b_test))}
d_list_b = np.sort(likelihoods_b_test)
x_b = np.array([0, 50, 200, 260, 285, 298, 299]).astype(np.int32)

d_m = {k: v for v, k in enumerate(list(likelihoods_m))}
d_list_m = np.sort(likelihoods_m)
x_m = np.array([0, 50, 300, 700, 1050, 1180, 1195]).astype(np.int32)


fig, ax = plt.subplots(figsize=(9,4))
for idx, i in enumerate(d_list_b[x_b]):
    newax = ax.inset_axes([i, 0.25, 1, 1], transform=ax.transData)
    newax.imshow(np.transpose(dataset_b_test_c[d_b[i]], (1,2,0)))
    newax.set_xticks([])
    newax.set_yticks([])
    for spine in newax.spines.values():
        spine.set_edgecolor('darkgreen')
    plt.axvline(i + 0.5, 0, 0.7, color='darkgreen', linestyle='--', zorder=0)
for idx, i in enumerate(d_list_m[x_m]):
    newax = ax.inset_axes([i, 0.05, 1, 1], transform=ax.transData)
    newax.imshow(np.transpose(dataset_m_c[d_m[i]], (1,2,0)))
    newax.set_xticks([])
    newax.set_yticks([])
    for spine in newax.spines.values():
        spine.set_edgecolor('darkred')
    plt.axvline(i + 0.5, 0, 0.5, color='darkred', linestyle='--', zorder=0)

ax.hist(likelihoods_b_train, 50, density=True, color='tab:blue', alpha=0.8, histtype='bar', ec='black', label="Benign train")
ax.hist(likelihoods_m, 50, density=True, color='tab:red', alpha=0.8, histtype='bar', ec='black', label="Malignant")
ax.hist(likelihoods_b_test, 50, density=True, color='tab:green', alpha=0.8, histtype='bar', ec='black', label="Benign test")
ax.set_ylim([0.0, 1.0])
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_xlabel("Averaged negative likelihood", fontsize=10)
ax.set_ylabel("Relative Frequency", fontsize=10)
ax.set_xlim([11,24])
ax.set_xticks([])
ax.set_yticks([])
order = [0,2,1]
handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=3, fontsize=12)
fig.subplots_adjust(wspace=0.025) 
plt.savefig(f"pics/wf-cond-{params.coupling}-wide-avg-{steps}.png", dpi=600, bbox_inches='tight')

fig, ax = plt.subplots(1,2, figsize=(9,4))
ax[0].hist(likelihoods_b_train, 50, density=True, color='tab:blue', alpha=0.8, histtype='bar', ec='black', label="Benign train")
ax[0].hist(likelihoods_m, 50, density=True, color='tab:red', alpha=0.8, histtype='bar', ec='black', label="Malignant")
ax[0].hist(likelihoods_b_test, 50, density=True, color='tab:green', alpha=0.8, histtype='bar', ec='black', label="Benign test")
# ax[0].set_ylim([0.0, 1.0])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].tick_params(axis='both', which='major', labelsize=8)
ax[0].set_xlabel("Averaged negative likelihood", fontsize=10)
ax[0].set_ylabel("Relative Frequency", fontsize=10)
order = [0,2,1]
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)

labels_positive = [1] * len(likelihoods_b_test)
labels_negative = [0] * len(likelihoods_m)
test_labels = labels_positive + labels_negative
test_scores = np.concatenate((-likelihoods_b_test, -likelihoods_m))
fpr, tpr, _ = roc_curve(test_labels, test_scores)

labels_positive2 = [1] * len(likelihoods_b_train)
test_labels2 = labels_positive2 + labels_negative
test_scores2 = np.concatenate((-likelihoods_b_train, -likelihoods_m))
fpr2, tpr2, _ = roc_curve(test_labels2, test_scores2)

ax[1].plot(fpr2, tpr2, color='navy', label='ROC train (area: %0.2f)' % auc(fpr2, tpr2), linewidth=4)
ax[1].plot(fpr, tpr, color='darkred', label='ROC test (area: %0.2f)' % auc(fpr, tpr), linewidth=4)
x = np.linspace(0, 1, 3)
ax[1].plot(x, x, 'k--', linewidth=3, label='No skill')
ax[1].set_xlim([0.0, 1.0])
ax[1].set_ylim([0.0, 1.0])
ax[1].yaxis.tick_right()
ax[1].tick_params(axis='both', which='major', labelsize=8)
ax[1].set_xlabel('1-Specificity', fontsize=10)
ax[1].set_ylabel('Sensitivity', fontsize=10)
ax[1].yaxis.set_label_position("right")
ax[1].legend(loc="lower right", fontsize=10)
fig.subplots_adjust(wspace=0.025) 
plt.savefig(f"pics/wf-cond-{params.coupling}-avg-{params.stepsPerResolution[0]}.png", dpi=600, bbox_inches='tight')
