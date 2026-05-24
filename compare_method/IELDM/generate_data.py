import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
from pathlib import Path
from IELDiffusion import XELDiffusionModel
import numpy as np
import warnings

warnings.filterwarnings('ignore')
label_encoder = LabelEncoder()

dataset = 'AHU'
method = 'vp'
model = 'xgb'
num = 50
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_CONFIG = {
    'AHU': {
        'train_path': PROJECT_ROOT / "datasets" / "tabular" / "AHU" / "train" / f"train_{num}.csv",
        'normal_label': 'F1',
    },
    'chiller': {
        'train_path': PROJECT_ROOT / "datasets" / "tabular" / "chiller" / "Train" / f"chiller_L1_Train_{num}.csv",
        'normal_label': 'normal',
    },
}

def get_dataset_config(dataset):
    if dataset not in DATASET_CONFIG:
        supported = ", ".join(DATASET_CONFIG)
        raise ValueError(f"Unknown dataset: {dataset}. Supported datasets: {supported}")
    return DATASET_CONFIG[dataset]

dataset_config = get_dataset_config(dataset)
data = pd.read_csv(dataset_config['train_path'])

normal_label = dataset_config['normal_label']
fault_data = data[data['Y'] != normal_label]
num_g = 1000 - num

X = fault_data.iloc[:, fault_data.columns != 'Y'].to_numpy()
y = fault_data.iloc[:, fault_data.columns == 'Y']
y = pd.DataFrame(label_encoder.fit_transform(y), columns=['Y']).to_numpy().ravel()

print('Training...')
start = time.time()
forest_model = XELDiffusionModel(X=X, label_y=y, n_t=50, model=model, duplicate_K=100,
                                    bin_indexes=[], cat_indexes=[], int_indexes=[], p_in_one=True,
                                    diffusion_type='vp', n_batch=0, gpu_hist=False, n_jobs=-1)
train_end = time.time()
print('Training.time: %.2f' % (train_end-start))
print('Generating...')
Xy_fake = forest_model.generate(num_g)
end = time.time()
output_root = PROJECT_ROOT / "generate_data" / "IELDM" / dataset / f"N{num}" / method
output_dir = output_root / model
output_dir.mkdir(parents=True, exist_ok=True)
file_path = output_dir / f"fake_{num}_cpu_{model}.csv"
np.savetxt(file_path, Xy_fake, delimiter=',')
print('Generation.time: %.2f' % (end - train_end))

time_dir = output_root / "time"
time_dir.mkdir(parents=True, exist_ok=True)
time_file = time_dir / f"time_{num}_cpu_{model}.txt"
with open(time_file, 'w') as file:
    file.write(f"Training time:\n{train_end-start}\n")
    file.write(f"Generation time:\n{end - train_end}\n")

