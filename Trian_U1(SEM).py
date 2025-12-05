import os
import inspect
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
from torchvision.transforms import Compose, Lambda, ToPILImage
from PIL import Image
from collections import defaultdict
from sklearn.metrics.pairwise import rbf_kernel

from utils.networkHelper import *
from noisePredictModels.Unet.UNet1 import Unet
from utils.trainNetworkHelper import SimpleDiffusionTrainer
from diffusionModels.simpleDiffusion.guided_Diffusion import DiffusionModel

custom_dataset_path = r"\dataset(O20)\FAG_images_AHU_Summer_Train_20\F"
image_size = 64
channels = 1
batch_size = 4
epoches = 10000
LR = 1e-4
NUM = 1000
timesteps = 1000
schedule_name = "linear_beta_schedule"
device = "cuda" if torch.cuda.is_available() else "cpu"
num_samples_per_label = 1000
U = 'U1'
image_gener_size = 64

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=channels),
    transforms.ToTensor()])
reverse_transform = Compose([
    Lambda(lambda t: t.permute(1, 2, 0)),
    Lambda(lambda t: t * 255.0),
    Lambda(lambda t: t.numpy().astype(np.uint8)),
    ToPILImage(),
])


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform   = transform
        self.image_paths = []
        self.labels      = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                label = int(folder.split('-')[0])
            except Exception as e:
                print(f" '{folder}': {e}")
                continue
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(folder_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img  = torchvision.io.read_image(path)
        img  = torchvision.transforms.functional.to_pil_image(img)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return img, label

custom_data = CustomDataset(custom_dataset_path, transform=transform)
data_loader = torch.utils.data.DataLoader( custom_data, batch_size=batch_size, shuffle=True, num_workers=0 )

dim_mults    = (1, 2)
denoise_model = Unet(dim=image_size, channels=channels, dim_mults=dim_mults, cond_dim=1)
DDPM         = DiffusionModel(schedule_name, timesteps, 1e-4, 0.02, denoise_model).to(device)
optimizer    = Adam(DDPM.parameters(), lr=LR)

root_path = f"./saved_train_models_{epoches}_{U}_{LR}_se2(resnet_block_groups=8)_Summer(20)_U1SEM"
setting = ( f"imageSize{image_size}_batch{batch_size}_channels{channels}_"
    f"dimMults{dim_mults}_timeSteps{timesteps}_{schedule_name}")
saved_path = os.path.join(root_path, setting)
os.makedirs(saved_path, exist_ok=True)
checkpoint_path = os.path.join(saved_path, "checkpoint.pth")


def save_checkpoint(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)

def load_checkpoint(path):
    if not os.path.isfile(path):
        return None
    load_kwargs = {"map_location": device}
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        load_kwargs["weights_only"] = True
    return torch.load(path, **load_kwargs)

start_epoch = 0
ckpt = load_checkpoint(checkpoint_path)
if ckpt:
    DDPM.load_state_dict(ckpt["model_state_dict"])
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as e:
        print(f"{e}，optimizer")
    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"{start_epoch} ")

one_epoch_trainer = SimpleDiffusionTrainer(epoches=1,train_loader=data_loader,optimizer=optimizer,device=device,timesteps=timesteps)

for epoch in range(start_epoch, epoches):
    DDPM = one_epoch_trainer(DDPM)
    save_checkpoint({ "epoch": epoch,"model_state_dict": DDPM.state_dict(),"optimizer_state_dict": optimizer.state_dict()}, checkpoint_path)

final_model_path = os.path.join(saved_path, "final_model.pth")
torch.save(DDPM.state_dict(), final_model_path)
print(f"final_model_path：{final_model_path}")

label_folder_map = {
    1: "F2",
    2: "F3",
    3: "F4",
    4: "F5",
    5: "F6",
    6: "F7",
    7: "F8"
}

output_base = f"./Generate_{epoches}_{U}_{LR}_se2(resnet_block_groups=8)_Summer(20)_U1SEM"

labels_to_generate = [1, 2, 3, 4, 5, 6, 7]

for label in labels_to_generate:
    gen_condition = torch.full((num_samples_per_label, 1), float(label), device=device)
    samples = DDPM(
        mode="generate",
        image_size=image_size,
        batch_size=num_samples_per_label,
        channels=channels,
        condition=gen_condition
    )

    if label not in label_folder_map:
        print(f"{label} no in label_folder_map")
        continue
    subfolder_name = label_folder_map[label]
    output_folder = os.path.join(output_base, subfolder_name)
    os.makedirs(output_folder, exist_ok=True)

    for i, sample_data in enumerate(samples[-1]):
        generate_image = sample_data.reshape(channels, image_gener_size, image_gener_size)
        print(f"{label}， {i + 1} fig：shape = {generate_image.shape}")
        g_image = reverse_transform(torch.from_numpy(generate_image))
        save_name = f"generated_image_label_{label}_{i + 1}.png"
        save_path = os.path.join(output_folder, save_name)
        g_image.save(save_path)
    print(f"{label} save: {output_folder}")

