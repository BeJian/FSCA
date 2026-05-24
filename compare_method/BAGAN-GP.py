# -*- coding: utf-8 -*-
import argparse
import os
import csv
import math
from pathlib import Path
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import grad

import torch
import torch.nn as nn
import torch.nn.functional as F

Au_num_ber = 960
num = 40
PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_dir = PROJECT_ROOT / "dataset" / "RP_1312" / "image" / f"m={num}" / "FAG_images_AHU_Summer_Train_40"
loss_dir = PROJECT_ROOT / "generate_data" / "BAGAN-GP" / "AHU" / f"N{num}" / "Runs"
data_aug_dir = PROJECT_ROOT / "generate_data" / "BAGAN-GP" / "AHU" / f"N{num}" / "Augmented"

# ==================== Hyperparameters ====================
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.9)
parser.add_argument("--n_cpu", type=int, default=0)
parser.add_argument("--latent_dim", type=int, default=128)
parser.add_argument("--n_classes", type=int, default=7, help="Overridden by the actual number of dataset classes")
parser.add_argument("--img_size", type=int, default=64)
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--sample_interval", type=int, default=500)
parser.add_argument("--lambda_gp", type=float, default=10.0)
parser.add_argument("--n_critic", type=int, default=1)
# Added for the paper architecture
parser.add_argument("--pretrain_epochs", type=int, default=50, help="Number of pretraining epochs")
parser.add_argument("--cond_dim", type=int, default=128, help="Embedding dimension")
opt = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def weights_init_normal(m):
    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "BatchNorm2d" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        c_ = max(1, in_dim // 8)
        self.query_conv = nn.Conv2d(in_dim, c_, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_dim, c_, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W
        q = self.query_conv(x).view(B, -1, N).transpose(1, 2)  # (B, N, C')
        k = self.key_conv(x).view(B, -1, N)                    # (B, C', N)
        attn = torch.bmm(q, k)                                 # (B, N, N)
        attn = F.softmax(attn / math.sqrt(k.size(1) + 1e-8), dim=-1)
        v = self.value_conv(x).view(B, C, N)                   # (B, C, N)
        out = torch.bmm(v, attn).view(B, C, H, W)              # (B, C, H, W)
        return self.gamma * out + x


transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(opt.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

actual_n_classes = len(dataset.classes)
if opt.n_classes != actual_n_classes:
    print(f"[INFO] Provided n_classes={opt.n_classes} differs from the dataset value {actual_n_classes}; using {actual_n_classes}.")
opt.n_classes = actual_n_classes
print("dataset.classes:", dataset.classes)
print("dataset.class_to_idx:", dataset.class_to_idx)
print("Using n_classes =", opt.n_classes)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, img_size=64, latent_dim=128, attn_at=("32", "16")):
        super().__init__()
        self.img_size = img_size
        downs = int(math.log2(img_size)) - 2
        assert downs >= 2
        ch = [64, 128, 256, 512, 512]
        blocks, in_c, fmap = [], in_channels, img_size
        self.attn_layers = nn.ModuleDict()
        for i in range(downs):
            out_c = ch[i] if i < len(ch) else ch[-1]
            blocks.append(nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            in_c = out_c
            fmap //= 2
            if str(fmap) in attn_at:
                self.attn_layers[str(fmap)] = SelfAttention(in_c)

        self.blocks = nn.ModuleList(blocks)
        self.to_lat = nn.Sequential(
            nn.Conv2d(in_c, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        fmap = self.img_size
        h = x
        for blk in self.blocks:
            h = blk(h)
            fmap //= 2
            key = str(fmap)
            if key in self.attn_layers:
                h = self.attn_layers[key](h)
        h = self.to_lat(h).view(h.size(0), -1)
        z = self.fc_mu(h)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=128, out_channels=3, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        ups = int(math.log2(img_size)) - 2
        assert ups >= 1
        in_ch = latent_dim + cond_dim

        layers = []
        # 1×1 -> 4×4
        layers += [
            nn.ConvTranspose2d(in_ch, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        mid_ch = [256, 128, 64, 32, 16]
        c_in = 512
        for c_out in mid_ch[: max(0, ups - 1)]:
            layers += [
                nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            c_in = c_out

        layers += [
            nn.ConvTranspose2d(c_in, out_channels, 4, 2, 1, bias=True),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z, y_emb):
        zy = torch.cat([z, y_emb], dim=1).view(z.size(0), -1, 1, 1)
        return self.net(zy)

class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, cond_dim, channels, img_size):
        super().__init__()
        self.embed = nn.Embedding(n_classes, cond_dim)
        self.decoder = Decoder(latent_dim, cond_dim, channels, img_size)

    def forward(self, z, labels):
        y_emb = self.embed(labels)
        return self.decoder(z, y_emb)

class Discriminator(nn.Module):
    def __init__(self, n_classes, channels, img_size, cond_dim=128):
        super().__init__()
        downs = int(math.log2(img_size)) - 4
        assert downs >= 1
        out_seq = [32, 64, 128, 256][:downs]
        in_c = channels
        blocks = []
        for oc in out_seq:
            blocks.append(nn.Sequential(
                nn.Conv2d(in_c, oc, 4, 2, 1, bias=False),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            in_c = oc
        self.blocks = nn.ModuleList(blocks)

        self.attn = SelfAttention(in_c)

        self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.max_pool = nn.AdaptiveMaxPool2d((8, 8))
        fc_in = in_c * 8 * 8 * 2
        self.fc1 = nn.Linear(fc_in, fc_in // 2)
        self.fc2 = nn.Linear(fc_in // 2, 128)
        self.fc_out = nn.Linear(128, 1)

        self.class_embed = nn.Embedding(n_classes, 128)

    def forward(self, x, labels):
        h = x
        for blk in self.blocks:
            h = blk(h)
        h = self.attn(h)
        ha, hm = self.avg_pool(h), self.max_pool(h)
        hf = torch.cat([ha.flatten(1), hm.flatten(1)], dim=1)
        hf = F.leaky_relu(self.fc1(hf), 0.2, inplace=True)
        feat = F.leaky_relu(self.fc2(hf), 0.2, inplace=True)  # φ(x)
        out_uncond = self.fc_out(feat).squeeze(1)
        proj = torch.sum(self.class_embed(labels) * feat, dim=1)
        logits = out_uncond + proj
        return logits

class AutoEncoder(nn.Module):
    def __init__(self, in_channels, img_size, latent_dim, n_classes, cond_dim):
        super().__init__()
        self.encoder = Encoder(in_channels, img_size, latent_dim, attn_at=("32", "16"))
        self.embed = nn.Embedding(n_classes, cond_dim)
        self.decoder = Decoder(latent_dim, cond_dim, in_channels, img_size)

    def forward(self, x, labels):
        z = self.encoder(x)
        y_emb = self.embed(labels)
        recon = self.decoder(z, y_emb)
        return recon, z

autoencoder = AutoEncoder(
    in_channels=opt.channels, img_size=opt.img_size,
    latent_dim=opt.latent_dim, n_classes=opt.n_classes, cond_dim=opt.cond_dim
).to(device)

generator = Generator(
    latent_dim=opt.latent_dim, n_classes=opt.n_classes, cond_dim=opt.cond_dim,
    channels=opt.channels, img_size=opt.img_size
).to(device)

discriminator = Discriminator(
    n_classes=opt.n_classes, channels=opt.channels, img_size=opt.img_size, cond_dim=opt.cond_dim
).to(device)

autoencoder.apply(weights_init_normal)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_AE = torch.optim.Adam(autoencoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G  = torch.optim.Adam(generator.parameters(),  lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D  = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

recon_criterion = nn.L1Loss().to(device)
bce_logits = nn.BCEWithLogitsLoss().to(device)


def compute_gradient_penalty(D, real_samples, fake_samples, real_labels):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    inter = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    logits = D(inter, real_labels)
    ones = torch.ones_like(logits, device=real_samples.device)
    grads = grad(
        outputs=logits, inputs=inter, grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return gp

@torch.no_grad()
def sample_image(n_row, batches_done):
    n_row = min(n_row, opt.n_classes)
    z = torch.randn(n_row * n_row, opt.latent_dim, device=device)
    labels = (torch.arange(n_row, device=device).repeat_interleave(n_row)) % opt.n_classes
    gen_imgs = generator(z, labels)
    sample_dir = os.path.join(loss_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    save_image(gen_imgs, os.path.join(sample_dir, f"{batches_done}.png"), nrow=n_row, normalize=True)


csv_save_path = os.path.join(loss_dir, 'loss.csv')
os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
loss_csv = open(csv_save_path, "w", newline="", encoding="utf-8")
writer = csv.writer(loss_csv)
writer.writerow(["epoch", "d_loss", "d_acc(%)", "g_loss"])


model_weights_dir = os.path.join(loss_dir, 'weight')
os.makedirs(model_weights_dir, exist_ok=True)
gen_weight_path   = os.path.join(model_weights_dir, "generator_final.pth")
disc_weight_path  = os.path.join(model_weights_dir, "discriminator_final.pth")
ae_enc_weight_path= os.path.join(model_weights_dir, "ae_encoder_final.pth")
ae_dec_weight_path= os.path.join(model_weights_dir, "ae_decoder_final.pth")

def safe_load(model, path):
    if not os.path.exists(path):
        return False
    try:
        state = torch.load(path, map_location=device)
        model.load_state_dict(state, strict=True)
        return True
    except Exception as e:
        print(f"[INFO] Could not load weights {os.path.basename(path)}: {e}; training from scratch.")
        return False

loaded_g = safe_load(generator, gen_weight_path)
loaded_d = safe_load(discriminator, disc_weight_path)


def pretrain_autoencoder():
    autoencoder.train()
    for ep in range(opt.pretrain_epochs):
        tot, n = 0.0, 0
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer_AE.zero_grad()
            recons, _ = autoencoder(imgs, labels)
            loss = recon_criterion(recons, imgs)
            loss.backward()
            optimizer_AE.step()
            tot += loss.item(); n += 1
        print(f"[AE] Epoch {ep+1}/{opt.pretrain_epochs}  Recon L1: {tot/max(1,n):.6f}")
    generator.decoder.load_state_dict(autoencoder.decoder.state_dict())


if loaded_g and loaded_d:
    print("Loaded GAN weights; skipping training.")
else:
    if opt.pretrain_epochs > 0:
        pretrain_autoencoder()
        torch.save(autoencoder.encoder.state_dict(), ae_enc_weight_path)
        torch.save(autoencoder.decoder.state_dict(), ae_dec_weight_path)

    batches_done = 0
    step_d = 0
    for epoch in range(opt.n_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        batch_count  = 0

        for i, (imgs, real_labels) in enumerate(dataloader):
            if imgs.size(0) == 0:
                continue
            imgs = imgs.to(device)
            real_labels = real_labels.to(device)
            bsz = imgs.size(0)


            optimizer_D.zero_grad()


            z = torch.randn(bsz, opt.latent_dim, device=device)
            fake_labels = torch.randint(0, opt.n_classes, (bsz,), device=device)
            fake_imgs = generator(z, fake_labels).detach()


            wrong_labels = torch.randint(0, opt.n_classes, (bsz,), device=device)
            wrong_labels = (wrong_labels + (wrong_labels == real_labels).long()) % opt.n_classes


            logit_real_true = discriminator(imgs, real_labels)
            logit_fake      = discriminator(fake_imgs, fake_labels)
            logit_real_wrong= discriminator(imgs, wrong_labels)


            valid = torch.ones_like(logit_real_true, device=device)
            fake  = torch.zeros_like(logit_fake,      device=device)

            d_real_true = bce_logits(logit_real_true, valid)
            d_fake      = bce_logits(logit_fake,      fake)
            d_real_wrong= bce_logits(logit_real_wrong,fake)


            gp = compute_gradient_penalty(discriminator, imgs.data, fake_imgs.data, real_labels)

            d_loss = d_real_true + d_fake + d_real_wrong + opt.lambda_gp * gp
            d_loss.backward()
            optimizer_D.step()
            step_d += 1


            g_loss_val = 0.0
            if step_d % opt.n_critic == 0:
                optimizer_G.zero_grad()
                z = torch.randn(bsz, opt.latent_dim, device=device)
                gen_labels = torch.randint(0, opt.n_classes, (bsz,), device=device)
                gen_imgs = generator(z, gen_labels)
                logit_gen = discriminator(gen_imgs, gen_labels)
                g_loss = bce_logits(logit_gen, valid)  # Encourage generated samples to be classified as real.
                g_loss.backward()
                optimizer_G.step()
                g_loss_val = g_loss.item()


            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss_val
            batch_count  += 1

            print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D: {d_loss.item():.6f} (GP {gp.item():.6f})] [G: {g_loss_val:.6f}]")

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=opt.n_classes, batches_done=batches_done)


        if batch_count > 0:
            avg_d_loss = epoch_d_loss / batch_count
            avg_g_loss = epoch_g_loss / batch_count
        else:
            avg_d_loss = avg_g_loss = 0.0
        writer.writerow([epoch, avg_d_loss, 0.0, avg_g_loss])
        print(f"Epoch {epoch} finished: Avg D loss: {avg_d_loss:.6f}, Avg G loss: {avg_g_loss:.6f}")

    loss_csv.close()
    torch.save(generator.state_dict(), gen_weight_path)
    torch.save(discriminator.state_dict(), disc_weight_path)

print("dataset.class_to_idx:", dataset.class_to_idx)
normal_class_idx = dataset.class_to_idx.get("Normal", None)

aug_root = data_aug_dir
os.makedirs(aug_root, exist_ok=True)

generator.eval()
with torch.no_grad():
    for class_name in dataset.classes:
        if normal_class_idx is not None and class_name == "Normal":
            continue
        class_idx = dataset.class_to_idx[class_name]
        class_folder = os.path.join(aug_root, class_name)
        os.makedirs(class_folder, exist_ok=True)

        print(f"[{class_name}] Generating {Au_num_ber} images...")
        remain = Au_num_ber
        step = max(1, min(64, Au_num_ber))
        img_id = 1
        while remain > 0:
            cur = min(step, remain)
            z = torch.randn(cur, opt.latent_dim, device=device)
            y = torch.full((cur,), class_idx, dtype=torch.long, device=device)
            gen_imgs = generator(z, y).cpu()
            for k in range(cur):
                save_path = os.path.join(class_folder, f"{class_name}-{img_id}.png")
                save_image(gen_imgs[k:k+1], save_path, normalize=True)
                img_id += 1
            remain -= cur
