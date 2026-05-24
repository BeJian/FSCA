# -*- coding: utf-8 -*-
import os
import time as t
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import utils
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import datasets

Au_num_ber = 980
num = 20
PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_dir = PROJECT_ROOT / "dataset" / "RP_1312" / "image" / f"m={num}" / "FAG_images_AHU_Summer_Trian_20"
loss_dir = PROJECT_ROOT / "generate_data" / "DCGAN" / "AHU" / f"N{num}" / "Runs"
data_aug_dir = PROJECT_ROOT / "generate_data" / "DCGAN" / "AHU" / f"N{num}" / "Augmented"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--channels", type=int, default=3)
parser.add_argument("--img_size", type=int, default=64, help="64x64")
args, _ = parser.parse_known_args()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

print("[INFO] Building ImageFolder dataset...")
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
print(f"[INFO] Classes: {full_dataset.classes}")
print(f"[INFO] Total samples: {len(full_dataset)}")

if len(full_dataset) == 0:
    raise RuntimeError(
        f"[ERROR] Dataset is empty: {data_dir}\n"
    )

os.makedirs(loss_dir, exist_ok=True)

class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),  # 1→4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 4→8
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),   # 8→16
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # 16→32
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, channels, kernel_size=4, stride=2, padding=1)  # 32→64
        )  # → (C, 64, 64)
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.main_module = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=4, stride=2, padding=1),   # 64→32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),        # 32→16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),        # 16→8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),       # 8→4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )  # → (1024, 4, 4)

        self.output = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0),         # 4→1
            nn.Sigmoid()
        )  # → (1, 1, 1)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


class DCGAN(object):
    def __init__(self, args, weight_dir: str, sample_dir: str):
        print("DCGAN model initalization.")
        self.G = Generator(args.channels).to(device)
        self.D = Discriminator(args.channels).to(device)
        self.C = args.channels
        self.img_size = args.img_size  # 64

        self.loss = nn.BCELoss().to(device)

        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0001, betas=(0.5, 0.999))

        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.number_of_images = 10

        self.weight_dir = weight_dir
        os.makedirs(self.weight_dir, exist_ok=True)
        self.gen_weight_path = os.path.join(self.weight_dir, "generator_final.pth")
        self.disc_weight_path = os.path.join(self.weight_dir, "discriminator_final.pth")

        self.sample_dir = sample_dir
        os.makedirs(self.sample_dir, exist_ok=True)

    def train(self, train_loader):
        self.t_begin = t.time()
        generator_iter = 0

        for epoch in range(self.epochs):
            self.epoch_start_time = t.time()

            for i, (images, _) in enumerate(train_loader):
                bsz = images.size(0)
                images = images.to(device, non_blocking=True)

                z = torch.randn(bsz, 100, 1, 1, device=device)
                real_labels = torch.ones(bsz, device=device)
                fake_labels = torch.zeros(bsz, device=device)

                outputs = self.D(images)               # [bsz,1,1,1]
                d_loss_real = self.loss(outputs.view(-1), real_labels)

                z = torch.randn(bsz, 100, 1, 1, device=device)
                fake_images = self.G(z)
                outputs = self.D(fake_images)          # [bsz,1,1,1]
                d_loss_fake = self.loss(outputs.view(-1), fake_labels)

                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad(set_to_none=True)
                d_loss.backward()
                self.d_optimizer.step()

                z = torch.randn(bsz, 100, 1, 1, device=device)
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                g_loss = self.loss(outputs.view(-1), real_labels)

                self.D.zero_grad(set_to_none=True)
                self.G.zero_grad(set_to_none=True)
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1

                if generator_iter % 1000 == 0:
                    print('Epoch-{}'.format(epoch + 1))
                    self.save_model()

                    with torch.no_grad():
                        z = torch.randn(800, 100, 1, 1, device=device)
                        samples = self.G(z).mul(0.5).add(0.5)
                        samples = samples.data[:64].cpu()
                        grid = utils.make_grid(samples)
                        utils.save_image(grid, os.path.join(self.sample_dir, f"{str(generator_iter).zfill(6)}.png"))

                    elapsed = t.time() - self.t_begin
                    print("Generator iter: {}".format(generator_iter))
                    print("Time {}".format(elapsed))

                if ((i + 1) % 100) == 0:
                    total_batches = (len(train_loader.dataset) + self.batch_size - 1) // self.batch_size
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), total_batches, d_loss.item(), g_loss.item()))

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        self.save_model()

    def evaluate(self, bsz: int = None):
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            if bsz is None:
                bsz = self.batch_size
            z = torch.randn(bsz, 100, 1, 1, device=device)
            samples = self.G(z).mul(0.5).add(0.5).data.cpu()
            grid = utils.make_grid(samples)
            out_path = os.path.join(self.sample_dir, 'dcgan_eval.png')
            print(f"Grid saved to '{out_path}'.")
            utils.save_image(grid, out_path)

    def to_np(self, x):
        return x.detach().cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), self.gen_weight_path)
        torch.save(self.D.state_dict(), self.disc_weight_path)
        print(f'[SAVED] {self.gen_weight_path} & {self.disc_weight_path}')

    def load_model(self):
        self.D.load_state_dict(torch.load(self.disc_weight_path, map_location=device))
        self.G.load_state_dict(torch.load(self.gen_weight_path, map_location=device))
        self.D.to(device)
        self.G.to(device)
        print('Loaded generator from {}.'.format(self.gen_weight_path))
        print('Loaded discriminator from {}.'.format(self.disc_weight_path))

    def generate_latent_walk(self, number):
        os.makedirs(self.sample_dir, exist_ok=True)
        number_int = 10
        with torch.no_grad():
            z1 = torch.randn(1, 100, 1, 1, device=device)
            z2 = torch.randn(1, 100, 1, 1, device=device)
            images_list = []
            alpha = 1.0 / float(number_int + 1)
            for _ in range(1, number_int + 1):
                z_intp = z1 * alpha + z2 * (1.0 - alpha)
                alpha += alpha
                fake_im = self.G(z_intp).mul(0.5).add(0.5)  # denormalize
                images_list.append(fake_im.view(self.C, self.img_size, self.img_size).detach().cpu())
            grid = utils.make_grid(images_list, nrow=number_int)
            out_path = os.path.join(self.sample_dir, f'interpolated_{str(number).zfill(3)}.png')
            utils.save_image(grid, out_path)
            print(f"Saved interpolated images to {out_path}.")


if __name__ == "__main__":
    classes = full_dataset.classes
    class_to_idx = full_dataset.class_to_idx
    print("dataset.class_to_idx:", class_to_idx)

    for class_name in classes:
        print("\n" + "="*80)
        print(f"[CLASS] {class_name} ")
        target_idx = class_to_idx[class_name]
        indices = [i for i, (_, y) in enumerate(full_dataset.samples) if y == target_idx]
        if len(indices) == 0:
            print(f"[WARN] Class {class_name} has no samples; skipping.")
            continue

        subset = Subset(full_dataset, indices)
        train_loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda")
        )

        print(f"[INFO] Class {class_name} sample count: {len(subset)}")
        _first_batch = next(iter(train_loader))
        print(f"[INFO] First batch for {class_name}: images.shape={_first_batch[0].shape}, labels.shape={_first_batch[1].shape}")

        weight_dir_cls = os.path.join(loss_dir, "weight", class_name)
        sample_dir_cls = os.path.join(loss_dir, "samples", class_name)

        model = DCGAN(args, weight_dir=weight_dir_cls, sample_dir=sample_dir_cls)
        model.train(train_loader)
        model.evaluate(bsz=args.batch_size)

        print(f"[AUG] Generating {Au_num_ber} augmented samples for class {class_name}...")
        aug_root = os.path.join(data_aug_dir, class_name)
        os.makedirs(aug_root, exist_ok=True)

        model.G.eval()
        done = 0
        gen_batch = 64
        with torch.no_grad():
            while done < Au_num_ber:
                cur = min(gen_batch, Au_num_ber - done)
                z = torch.randn(cur, 100, 1, 1, device=device)
                gen_imgs = model.G(z).cpu()  # [-1,1]
                for k in range(cur):
                    save_path = os.path.join(aug_root, f"{class_name}-{done + k + 1}.png")
                    # normalize=True maps [-1, 1] to [0, 1].
                    utils.save_image(gen_imgs[k], save_path, normalize=True)
                done += cur
        print(f"[AUG] Class {class_name} saved to: {aug_root}")

    print("\n[INFO] Training and augmented sample export completed for all classes.")
