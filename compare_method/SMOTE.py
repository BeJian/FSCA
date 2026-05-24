import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler  # 0~1
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore

label_encoder = LabelEncoder()
scaler = MinMaxScaler()
class_num = 8
cc = 10
num_smote = 1000
PROJECT_ROOT = Path(__file__).resolve().parents[1]

np.random.seed(36)
features = ['TEO', 'FWC', 'FWE', 'TCA', 'TO_sump', 'TO_feed', 'PO_feed', 'VC', 'VE', 'TWI']


def load_select_train_data(data):
    F1 = data[data['Y'] == 'cf12' ].sample(n=cc, random_state=36)
    F2 = data[data['Y'] == 'eo14' ].sample(n=cc, random_state=36)
    F3 = data[data['Y'] == 'fwc10'].sample(n=cc, random_state=36)
    F4 = data[data['Y'] == 'fwe10'].sample(n=cc, random_state=36)
    F5 = data[data['Y'] == 'nc1' ].sample(n=cc, random_state=36)
    F6 = data[data['Y'] == 'rl10'].sample(n=cc, random_state=36)
    F7 = data[data['Y'] == 'ro10' ].sample(n=cc, random_state=36)
    Normal = data[data['Y'] == 'normal'].sample(n=1000, random_state=36)
    sdata = pd.concat([F1, F2, F3, F4, F5, F6, F7, Normal], ignore_index=True)

    # sdata_cleaned = replace_outliers_with_mean(sdata.copy(), features)
    X = sdata.iloc[:, data.columns != "Y"]
    # X = sdata[features]
    labels = sdata.iloc[:, data.columns == "Y"].values.ravel()

    global label_encoder
    label_encoder.fit(labels)
    labels = label_encoder.transform(labels)

    X_norm = scaler.fit_transform(X)

    return X, labels, X_norm


def load_select_test_data(data, select_num):
    size = select_num * class_num
    F1 = data[data['Y'] == 'cf12' ].sample(n=select_num)
    F2 = data[data['Y'] == 'eo14' ].sample(n=select_num)
    F3 = data[data['Y'] == 'fwc10'].sample(n=select_num)
    F4 = data[data['Y'] == 'fwe10' ].sample(n=select_num)
    F5 = data[data['Y'] == 'nc1' ].sample(n=select_num)
    F6 = data[data['Y'] == 'rl10'].sample(n=select_num)
    F7 = data[data['Y'] == 'ro10' ].sample(n=select_num)
    Normal = data[data['Y'] == 'normal'].sample(n=select_num)
    sdata = pd.concat([F1, F2, F3, F4, F5, F6, F7, Normal], ignore_index=True)

    X = sdata.iloc[:, data.columns != "Y"]
    # X = sdata[features]
    labels = sdata.iloc[:, data.columns == "Y"].values.ravel()

    global label_encoder
    labels = label_encoder.transform(labels)

    X_norm = scaler.transform(X)

    return X, labels, X_norm


# def replace_outliers_with_mean(df, cols):
#     """
#     Replace outliers in specified columns of a DataFrame with the mean of those columns.
#     An outlier is defined as a value with a Z-score greater than 3 or less than -3.
#
#     Parameters:
#     df (pd.DataFrame): The input DataFrame.
#     cols (list): List of column names to check for outliers.
#
#     Returns:
#     pd.DataFrame: DataFrame with outliers replaced by the mean.
#     """
#     for col in cols:
#         z_scores = zscore(df[col])
#         mean_value = df[col].mean()
#         df.loc[np.abs(z_scores) > 3, col] = mean_value
#     return df


real_data = pd.read_csv(PROJECT_ROOT / "datasets" / "tabular" / "chiller" / "Train" / f"chiller_L1_Train_{cc}.csv")
real_X, real_Y, _ = load_select_train_data(real_data)

class_counts = pd.Series(real_Y).value_counts().to_dict()

sampling_strategy = {cls: num_smote for cls in class_counts if class_counts[cls] < num_smote }

if sampling_strategy:
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=4, random_state=36)
    real_X_resampled, real_Y_resampled = smote.fit_resample(real_X, real_Y)
else:
    real_X_resampled, real_Y_resampled = real_X, real_Y

features = list(real_X_resampled.columns)

resampled_df = pd.DataFrame(real_X_resampled, columns=features)
resampled_labels = label_encoder.inverse_transform(real_Y_resampled)
resampled_df['Y'] = resampled_labels

category_order = ['cf12', 'eo14', 'fwc10', 'fwe10', 'nc1', 'rl10', 'ro10', 'normal']
resampled_df_sorted = resampled_df.sort_values(by='Y', key=lambda x: x.map({cat: i for i, cat in enumerate(category_order)})).reset_index(drop=True)

output_dir = PROJECT_ROOT / "generate_data" / "SMOTE" / "chiller" / f"N{cc}"
output_dir.mkdir(parents=True, exist_ok=True)
output_csv_path = output_dir / f'chiller_Train({cc}_smote_{num_smote})_FL1.csv'
resampled_df_sorted.to_csv(output_csv_path, index=False)

data_ = pd.read_csv(PROJECT_ROOT / "datasets" / "tabular" / "chiller" / "Test" / "chiller_L1_Test_300.csv", sep=',')  # Header row is present by default.
Xtest, ytest, X_norm = load_select_test_data(data_, 300)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=36),
    "Random Forest": RandomForestClassifier(random_state=36),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier(verbosity=-1),
}

output_file = output_dir / f'chiller_Train({cc}_smote_{num_smote})_{cc}.txt'

with open(output_file, 'w') as file:
    for name, model in models.items():
        model.fit(real_X_resampled, real_Y_resampled)
        y_pred = model.predict(Xtest)

        report = classification_report(ytest, y_pred, digits=4)

        cm = confusion_matrix(ytest, y_pred)

        print(f"{name} Report:\n{report}")
        file.write(f"{name} Report:\n{report}\n\n")



