import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        positions = np.arange(timesteps)[:, np.newaxis]  # Shape: (timesteps, 1)
        dimensions = np.arange(self.embedding_dim)[
            np.newaxis, :
        ]  # Shape: (1, embedding_dim)

        # Compute angles using sine for even indices and cosine for odd indices
        angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / self.embedding_dim)
        angle_rads = positions * angle_rates

        pos_encoding = np.zeros_like(angle_rads)
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return pos_encoding


if __name__ == "__main__":
    embedding_dim = 128  # Match embedding dimension from the reference
    timesteps = 1000  # Number of rows (timesteps)

    # Generate Embeddings
    embedding = SinusoidalPositionalEmbedding(embedding_dim)
    embeddings = embedding(timesteps)  # Shape: [timesteps, embedding_dim]

    # Plot Heatmap
    plt.figure(figsize=(6, 4))
    plt.imshow(embeddings, aspect="auto", cmap="coolwarm")
    cbar = plt.colorbar()
    cbar.outline.set_visible(False)
    cbar.set_ticks(np.linspace(-1.0, 1.0, 5))
    plt.title("Sinusoidal Timestep Embedding (dim={0})".format(embedding_dim), fontsize=14)
    plt.xlabel("Embedding Dimensions")
    plt.ylabel("Timesteps", labelpad=0)
    y_ticks = np.arange(0, timesteps + 1, 200)
    plt.yticks(y_ticks)
    plt.tight_layout()
    plt.savefig("Sinusoidal Embedding Heatmap.png", dpi=200, bbox_inches='tight')
    plt.show()
