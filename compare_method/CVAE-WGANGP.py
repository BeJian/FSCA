import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Data loading and preprocessing
dataset_name = 'AHU'
num = 50
EP = 500 

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_train_csv(dataset_name, num):
    if dataset_name == 'chiller':
        return PROJECT_ROOT / "datasets" / "tabular" / "chiller" / "Train" / f"chiller_L1_Train_{num}.csv"
    if dataset_name == 'AHU':
        return PROJECT_ROOT / "datasets" / "tabular" / "AHU" / "train" / f"train_{num}.csv"
    raise ValueError(f"Unknown dataset name: {dataset_name}")

data = pd.read_csv(get_train_csv(dataset_name, num))

if dataset_name == 'chiller':
    select_features = ['TEO','FWC','FWE','TCA','TO_sump','TO_feed','PO_feed','VC','VE','TWI','Y']
    data = data[select_features]
    fault_data = data[data['Y'] != 'normal']
    num_g = 1000-num
elif dataset_name == 'AHU':
    fault_data = data[data['Y'] != 'F1']
    num_g = 1000-num

x = fault_data.iloc[:, :-1].values

y_dummies = pd.get_dummies(fault_data.iloc[:, -1])
labels = y_dummies.values
class_values = y_dummies.columns.tolist()

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.float32)

# Data loader
batch_size = 32
tensor_dataset = TensorDataset(x_tensor, y_tensor)
dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

# Model definition (Conditional VAE-GAN)
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2))
        self.mean = nn.Linear(128, latent_dim)
        self.logstd = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, input_dim))

    def reparametrize(self, mean, logstd):
        eps = torch.randn(mean.size()).to(mean.device)
        return mean + eps * torch.exp(logstd)

    def forward(self, x, labels):
        enc_input = torch.cat([x, labels], dim=1)
        h = self.encoder(enc_input)
        mean, logstd = self.mean(h), self.logstd(h)
        z = self.reparametrize(mean, logstd)
        dec_input = torch.cat([z, labels], dim=1)
        recon = self.decoder(dec_input)
        return recon, mean, logstd

class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + num_classes, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1))

    def forward(self, x, labels):
        return self.model(torch.cat([x, labels], dim=1)).squeeze()

# Model initialization
input_dim, latent_dim, num_classes = x.shape[1], 32, labels.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = ConditionalVAE(input_dim, latent_dim, num_classes).to(device)
D = ConditionalDiscriminator(input_dim, num_classes).to(device)

opt_vae = optim.Adam(vae.parameters(), lr=1e-4)
opt_D = optim.Adam(D.parameters(), lr=1e-4)

# Loss function
def vae_loss(recon, x, mean, logstd):
    mse = nn.MSELoss()(recon, x)
    kld = -0.5 * torch.mean(1 + 2 * logstd - mean.pow(2) - (2 * logstd).exp())
    return mse + kld

# WGAN-GP parameters
lambda_gp = 10

# Train model (WGAN-GP)
for epoch in range(EP):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        recon, _, _ = vae(batch_x, batch_y)

        # Gradient penalty
        alpha = torch.rand(batch_x.size(0), 1).to(device)
        interpolates = (alpha * batch_x + (1 - alpha) * recon.detach()).requires_grad_(True)
        disc_interpolates = D(interpolates, batch_y)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True, retain_graph=True)[0]
        gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        loss_D = D(recon.detach(), batch_y).mean() - D(batch_x, batch_y).mean() + gradient_penalty
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        recon, mean, logstd = vae(batch_x, batch_y)
        loss_vae = vae_loss(recon, batch_x, mean, logstd) - D(recon, batch_y).mean()
        opt_vae.zero_grad()
        loss_vae.backward()
        opt_vae.step()

    print(f'Epoch[{epoch+1}], Loss D: {loss_D.item():.4f}, Loss VAE: {loss_vae.item():.4f}')


# Data generation function
def generate_data(vae, num_samples, class_label, scaler):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        labels = torch.zeros(num_samples, num_classes).to(device)
        labels[:, class_label] = 1
        generated = vae.decoder(torch.cat([z, labels], dim=1)).cpu().numpy()
        return scaler.inverse_transform(generated)

# Generate and save data
output_dir = PROJECT_ROOT / "generate_data" / "CVAE-WGANGP" / dataset_name / f"N{num}"
os.makedirs(output_dir, exist_ok=True)
i=11
# for i in range(10):
if i == 11:
    all_generated = []
    for c in range(num_classes):
        gen_samples = generate_data(vae, num_g, c, scaler)
        df = pd.DataFrame(gen_samples, columns=fault_data.columns[:-1])
        df['Y'] = class_values[c]
        all_generated.append(df)

    generated_df = pd.concat(all_generated, ignore_index=True)
    gen_path = output_dir / f'CVAE-WGANGP_{dataset_name}_{num}_{i+1}.csv'
    generated_df.to_csv(gen_path, index=False)

    combined_df = pd.concat([data, generated_df], ignore_index=True)
    combined_path = output_dir / f'OA_{dataset_name}_{num}_{i+1}.csv'
    combined_df.to_csv(combined_path, index=False)

    print(f'Saved generated data: {gen_path}')
    print(f'Saved combined (original + generated): {combined_path}')
