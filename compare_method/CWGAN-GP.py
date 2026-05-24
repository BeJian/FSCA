# -*- coding: utf-8 -*-

import os
import time
import pickle
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad as torch_grad


LabN_size = 7
select_number = 50
generate_num = 1000 - select_number
Zn_size = 10
lr_g = 0.0001
lr_D = 0.0001
train_epoch = 10000
D_updates_per_epoch = 4
DA_name = 'AHU'

PROJECT_ROOT = Path(__file__).resolve().parents[1]
root = PROJECT_ROOT / "generate_data" / "CWGAN-GP" / DA_name / f"N{select_number}"
hist_dir = 'train_hist/'

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


scaler = MinMaxScaler()
label_encoder = LabelEncoder()

def load_original_data(D_name: str) -> pd.DataFrame:
    if D_name == 'chiller':
        path = PROJECT_ROOT / "datasets" / "tabular" / "chiller" / "Train" / f"chiller_L1_Train_{select_number}.csv"
        return pd.read_csv(path, sep=',', header='infer')
    if D_name == 'AHU':
        path = PROJECT_ROOT / "datasets" / "tabular" / "AHU" / "train" / f"train_{select_number}.csv"
        return pd.read_csv(path, sep=',', header='infer')
    raise ValueError(f"Unknown dataset name: {D_name}")


def load_select_data(data: pd.DataFrame, select_num: int, D_name: str):

    if 'Y' in data.columns:
        label_col = 'Y'
    elif 'fault type' in data.columns:
        label_col = 'fault type'
    else:
        raise ValueError("Cannot find label column 'Y' or 'fault type' in data.")

    if D_name == 'chiller':
        class_list = ['cf12', 'eo14', 'fwc10', 'fwe10', 'nc1', 'rl10', 'ro10']
    elif D_name == 'AHU':
        class_list = [f'F{k}' for k in range(1, LabN_size + 1)]
    else:
        raise ValueError(f"Unknown DA_name: {D_name}")

    parts = []
    for cls in class_list:
        parts.append(data[data[label_col] == cls].sample(n=select_num, replace=False))
    sdata = pd.concat(parts, ignore_index=True)

    feature_cols = [c for c in sdata.columns if c != label_col]
    X = sdata[feature_cols].copy().values
    labels_raw = sdata[[label_col]].values.reshape(-1, 1)

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels_raw.ravel())
    class_names = le.classes_.tolist()


    X_norm = scaler.fit_transform(X)

    y_placeholder = np.ones((X_norm.shape[0], 1)).ravel()
    return (X_norm, y_encoded), y_placeholder, feature_cols, class_names

def to_one_hot(y_int: np.ndarray, num_classes: int) -> np.ndarray:
    y_int = y_int.astype(int)
    return np.eye(num_classes, dtype=np.float32)[y_int]

def G_labels(select_num: int, noise_dim: int, save_name: bool = False):

    labels_int = np.concatenate([np.full((select_num,), c, dtype=int) for c in range(LabN_size)], axis=0)
    y_onehot = to_one_hot(labels_int, LabN_size).astype(np.float32)
    z = np.random.uniform(-1.0, 1.0, size=(select_num * LabN_size, noise_dim)).astype(np.float32)

    if save_name:
        os.makedirs(root, exist_ok=True)
        np.savetxt(os.path.join(root, f"labels_{generate_num}.csv"), labels_int.reshape(-1, 1), delimiter=",", fmt='%d')
    return y_onehot, z

class Generator(nn.Module):
    def __init__(self, noise_dim: int, y_dim: int, x_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + y_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, x_dim)
        )

    def forward(self, z, y_onehot):
        inp = torch.cat([z, y_onehot], dim=1)
        return self.net(inp)

class Discriminator(nn.Module):
    def __init__(self, x_dim: int, y_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x, y_onehot):
        inp = torch.cat([x, y_onehot], dim=1)
        return self.net(inp)


train_hist = {
    'D_losses': [],
    'G_losses': [],
    'per_epoch_ptimes': [],
}

def show_train_hist(hist, show=False, save=False):
    x_axis = range(len(hist['D_losses']))
    plt.figure()
    plt.plot(x_axis, hist['D_losses'], label='D_loss')
    plt.plot(x_axis, hist['G_losses'], label='G_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(root, hist_dir)
    os.makedirs(path, exist_ok=True)

    pd.DataFrame(hist).to_csv(os.path.join(path, 'train_loss.csv'), index=False)

    if save:
        plt.savefig(os.path.join(path, 'train_hist.png'))
    if show:
        plt.show()
    else:
        plt.close()

def gradient_penalty(discriminator, real_x, fake_x, cond_y):

    batch_size = real_x.size(0)
    epsilon = torch.rand(batch_size, 1, device=device).expand_as(real_x)

    inter_x = epsilon * real_x + (1 - epsilon) * fake_x
    inter_x.requires_grad_(True)

    d_inter = discriminator(inter_x, cond_y)
    gradients = torch_grad(
        outputs=d_inter.sum(),
        inputs=inter_x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    grad_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    gp = 10.0 * torch.relu(grad_norm - 1.0).mean()
    return gp


os.makedirs(root, exist_ok=True)
os.makedirs(os.path.join(root, hist_dir), exist_ok=True)


print('loading data ...')
dataset_df = load_original_data(DA_name)

(x_np, y_int_np), _placeholder, feature_cols, class_names = load_select_data(dataset_df, select_number, DA_name)
M_size = select_number * LabN_size
G_size = select_number * LabN_size

N_size = x_np.shape[1]

y_oh_np = to_one_hot(y_int_np, LabN_size).astype(np.float32)

x_real = torch.tensor(x_np, dtype=torch.float32, device=device)           # (M_size, N_size)
y_real_oh = torch.tensor(y_oh_np, dtype=torch.float32, device=device)     # (M_size, LabN_size)


G = Generator(noise_dim=Zn_size, y_dim=LabN_size, x_dim=N_size).to(device)
D = Discriminator(x_dim=N_size, y_dim=LabN_size).to(device)

opt_D = optim.RMSprop(D.parameters(), lr=lr_D)
opt_G = optim.RMSprop(G.parameters(), lr=lr_g)

print('training start!')
start_time = time.time()

for epoch in range(train_epoch):
    epoch_start = time.time()

    D_losses_epoch = []
    for _ in range(D_updates_per_epoch):
        z_np = np.random.uniform(-1.0, 1.0, size=(G_size, Zn_size)).astype(np.float32)
        z = torch.tensor(z_np, dtype=torch.float32, device=device)

        fake = G(z, y_real_oh)
        D_real = D(x_real, y_real_oh).mean()
        D_fake = D(fake.detach(), y_real_oh).mean()
        gp = gradient_penalty(D, x_real, fake.detach(), y_real_oh)

        D_loss = -(D_real - D_fake - gp)

        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        D_losses_epoch.append(D_loss.item())


    G_y_oh_np, z_np = G_labels(select_number, Zn_size, save_name=False)
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    G_y_oh = torch.tensor(G_y_oh_np, dtype=torch.float32, device=device)

    fake_for_G = G(z, G_y_oh)
    D_fake_for_G = D(fake_for_G, y_real_oh).mean()  # Still use the real label condition to match the original version.
    G_loss = -D_fake_for_G

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()


    per_epoch_ptime = time.time() - epoch_start
    train_hist['D_losses'].append(np.mean(D_losses_epoch))
    train_hist['G_losses'].append(G_loss.item())
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    if (epoch + 1) % 50 == 0 or epoch == 0:
        print('[%d/%d] - ptime: %.2f  mloss_d: %.4f  mloss_g: %.4f' %
              (epoch + 1, train_epoch, per_epoch_ptime, np.mean(D_losses_epoch), G_loss.item()))

    if epoch == train_epoch - 1:

        G_y_oh_np, z_np = G_labels(generate_num, Zn_size, save_name=True)  # Save labels_{generate_num}.csv with integer classes.
        z = torch.tensor(z_np, dtype=torch.float32, device=device)
        G_y_oh = torch.tensor(G_y_oh_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            gen_scaled = G(z, G_y_oh).cpu().numpy()


        gen_unscaled = scaler.inverse_transform(gen_scaled)

        gen_labels = np.concatenate([np.full((generate_num,), cls, dtype=object) for cls in class_names], axis=0)

        gen_df = pd.DataFrame(gen_unscaled, columns=feature_cols)
        gen_df['Y'] = gen_labels

        gen_name = f"G_data_{generate_num}.csv"
        gen_out_path = os.path.join(root, gen_name)
        gen_df.to_csv(gen_out_path, index=False)
        print('Saved generated:', gen_out_path)


        combined_df = pd.concat([dataset_df, gen_df], ignore_index=True)
        combined_name = f"OA_{DA_name}_{select_number}.csv"
        combined_out_path = os.path.join(root, combined_name)
        combined_df.to_csv(combined_out_path, index=False)
        print('Saved combined :', combined_out_path)


total_ptime = time.time() - start_time
print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f'
      % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")

with open(os.path.join(root, hist_dir, 'train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, show=True, save=True)
