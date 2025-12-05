import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# 假设我们有 7 个类别，每个类别的嵌入维度是 128
num_categories = 7
embedding_dim = 128

# 创建嵌入层
embedding_layer = torch.nn.Embedding(num_categories, embedding_dim)

# 获取嵌入矩阵
embedding_matrix = embedding_layer.weight.detach().numpy()  # shape: [7, 128]

# 绘制热图
plt.figure(figsize=(6, 4))
ax = sns.heatmap(
    embedding_matrix,
    cmap="coolwarm",
    cbar=True,
    xticklabels=20,  # 每隔 10 个维度显示一次标签
    yticklabels=[f"Label {i+1}" for i in range(num_categories)]
)

# 设置标题和坐标轴标签
plt.title(f"Label Embedding (dim=128)", fontsize=14)
plt.xlabel("Embedding Dimensions", fontsize=14)
plt.ylabel("Categories", fontsize=14)

# 调整横坐标刻度标签（旋转角度 & 间距）
ax.set_xticks(np.arange(0, embedding_dim, 20))  # 每隔10个维度显示
ax.set_xticklabels(np.arange(0, embedding_dim, 20), fontsize=14)

# 显示图形
plt.tight_layout()
plt.savefig("Label Embedding Heatmap.png", dpi=200, bbox_inches='tight')
plt.show()
