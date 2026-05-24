import os
import time
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import seaborn as sns

plt.rc('font', family='Times New Roman', style='normal', weight='light')
from sklearn.manifold import TSNE


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=8, in_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def resnet18(num_classes=8, in_channels=1):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)


def train_model(model, train_loader, val_loader, criterion, optimizer, out_dir, num_epochs=10, patience=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model_wts = None
    best_acc = 0.0
    best_epoch = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    stime = time.time()
    end_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_preds.double() / len(train_loader.dataset)

        scheduler.step()

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}, Train_Acc: {epoch_acc:.4f}')

        model.eval()
        val_loss = 0.0
        correct_val_preds = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val_preds += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_val_preds.double() / len(val_loader.dataset)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        print(f'------------- Validation Loss: {val_loss:.4f}, Val_Acc: {val_acc:.4f}')

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, os.path.join(out_dir, 'best_model.pth'))
            print('save best!')

        if epoch - best_epoch > patience:
            end_epoch = epoch
            print(f'early stopping at epoch {epoch}')
            break

    etime = time.time()
    alltime = etime - stime
    print('training time:', alltime)
    print(f'max_val_acc: {max(history["val_acc"]):.4f}')
    if best_model_wts:
        model.load_state_dict(best_model_wts)

    return model, history, alltime, end_epoch


def test_model(model, test_loader, out_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct_preds = 0
    all_preds = []
    all_labels = []
    stime = time.time()
    all_features = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            features = outputs
            all_features.append(features.cpu().numpy())
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    etime = time.time()
    alltime = etime - stime
    print('testing time:', alltime)
    import numpy as np
    test_acc = float(correct_preds.double() / len(test_loader.dataset))
    np.savetxt(os.path.join(out_dir, ' test_acc.txt'), [test_acc])

    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    np.savetxt(os.path.join(out_dir, 'weighted_f1.txt'),  [weighted_f1])

    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Weighted F1 Score: {weighted_f1:.4f}')
    report = classification_report(all_labels, all_preds, digits=4)
    print(report)

    import numpy as np
    plt.rc('font', family='Times New Roman', style='normal', weight='normal', size=16)
    all_features = np.concatenate(all_features, axis=0)
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    features_2d = tsne.fit_transform(all_features)
    mark = ['*', 'p', 's', 'v', '8', '^', 'h', 'o', 'd', '>', '4', 'x', '1', 'D', '<']
    plt.figure()
    num_classes = len(np.unique(all_labels))
    for i in range(num_classes):
        idx = np.array(all_labels) == i
        # Note: classes here use the class names from the test set
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=test_loader.dataset.classes[i], s=100,
                    alpha=0.8, marker=mark[i])
    plt.legend(fontsize=12)
    plt.xlabel("Component 1", weight='bold', fontsize=18)
    plt.ylabel("Component 2", weight='bold', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "t-SNE.svg"), dpi=600)

    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=13)
    cm = confusion_matrix(all_labels, all_preds)
    cm_ = cm.astype('float') / cm.sum(axis=1)[:, None] * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_, display_labels=test_loader.dataset.classes)
    fig, ax = plt.subplots()
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2f')
    plt.xticks(rotation=30, fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_xlabel('Predicted label', weight='bold', fontsize=18)
    ax.set_ylabel('True label', weight='bold', fontsize=18)
    plt.title('Confusion Matrix', weight='bold', fontsize=20)
    plt.savefig(os.path.join(out_dir, 'Confusion_matrix.svg'), bbox_inches='tight', dpi=600)
    plt.tight_layout()

    return alltime, report


def plot_training_history(history, out_dir):
    expected_keys = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    for key in expected_keys:
        if key not in history:
            history[key] = [None] * len(history['train_loss'])
    df_history = pd.DataFrame(history)
    df_history['epoch'] = range(1, len(df_history) + 1)
    csv_file_path = os.path.join(out_dir, 'training_history.csv')
    df_history.to_csv(csv_file_path, index=False)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 4))
    plt.figure()
    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=16)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs', size=18)
    plt.ylabel('Loss', size=18)
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'Loss.svg'), bbox_inches='tight', dpi=600)

    plt.figure()
    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=16)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs', size=18)
    plt.ylabel('Accuracy (%)', size=18)
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'ACC.svg'), bbox_inches='tight', dpi=600)

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

# Result output path
result_dir1 = 'results_FL1(U1)_10/RestNet18_result'
os.makedirs(result_dir1, exist_ok=True)

# Dataset path settings
train_data_dir = PROJECT_ROOT / "generate_data" / "DDPM-UNet" / "AHU" / "N10" / "Generate_10000_U1_0.0001_se2(resnet_block_groups=8)_Summer(10)_U1SEM"
test_data_dir = PROJECT_ROOT / "datasets" / "image" / "AHU" / "Test"

# Preprocessing parameters
channel = 1
norm = [0.5] if channel == 1 else [0.5, 0.5, 0.5]
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=channel),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm, std=norm),
])

# Load the training set and split 20% as the validation set
train_dataset_full = datasets.ImageFolder(root=train_data_dir, transform=transform)
class_to_idx = train_dataset_full.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
all_labels = [idx_to_class[idx] for idx in train_dataset_full.targets]
train_idx, val_idx = train_test_split(
    list(range(len(train_dataset_full))), test_size=0.2, stratify=all_labels, random_state=36)
train_dataset = Subset(train_dataset_full, train_idx)
val_dataset = Subset(train_dataset_full, val_idx)

# Load the independent test set
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
test_labels = [label for _, label in test_dataset]
label_counts = Counter(test_labels)
print("Test set class distribution:")
for label, count in label_counts.items():
    print(f"Class {label}: {count} samples")

# Create DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


result_dir = result_dir1
os.makedirs(result_dir, exist_ok=True)

epochs = 100
early_stop = 20

model = resnet18(num_classes=len(train_dataset_full.classes), in_channels=channel)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
summary(model, (channel, 64, 64))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model, history, train_time, epoch_ = train_model(model, train_loader, val_loader, criterion, optimizer,
                                                 result_dir, num_epochs=epochs, patience=early_stop)

test_time, reports = test_model(model, test_loader, result_dir)


with open(os.path.join(result_dir, "report.txt"), "w") as file:
    file.write(f"early stop epoch: {epoch_}\n")
    file.write(f"training time: {train_time}\n")
    file.write(f"testing time: {test_time}\n")
    file.write(reports)

plot_training_history(history, result_dir)

