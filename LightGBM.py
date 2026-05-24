import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent

def load_X_y(path):

    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"{path} must contain at least one feature column and one label column")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def main():
    # ====== 1. Set file paths ======
    train_path = PROJECT_ROOT / "generate_data" / "SMOTE" / "chiller" / "N10" / "chiller_Train(10_smote_1000)_FL1.csv"
    test_path  = PROJECT_ROOT / "datasets" / "tabular" / "chiller" / "Test" / "chiller_L1_Test_300.csv"

    # ====== 2. Load data ======
    X_train, y_train = load_X_y(train_path)
    X_test,  y_test  = load_X_y(test_path)

    # ====== 3. Encode labels ======
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    class_names = le.classes_
    print("Class mapping:", dict(enumerate(class_names)))

    # ====== 4. Optional: scale features ======
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Training set shape:", X_train.shape, "label distribution:", pd.Series(y_train_enc).value_counts().to_dict())
    print("Test set shape:", X_test.shape,  "label distribution:", pd.Series(y_test_enc).value_counts().to_dict())

    # ====== 5. Build and train the LightGBM multiclass model ======
    num_classes = len(class_names)
    model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=100,
        max_depth=10,
        learning_rate=0.1,
        n_estimators=100,
        objective='multiclass',
        num_class=num_classes,
        random_state=42
    )
    model.fit(X_train, y_train_enc)

    # ====== 6. Predict and evaluate ======
    y_pred_enc = model.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred_enc)
    f1  = f1_score(y_test_enc, y_pred_enc, average='macro')


    report = classification_report(y_test_enc, y_pred_enc, digits=4)
    print(report)
    cm  = confusion_matrix(y_test_enc, y_pred_enc, labels=range(num_classes))

    print(f"Test set accuracy (Accuracy): {acc:.4f}")
    print(f"Test set macro-average F1 (Macro-F1): {f1:.4f}")
    print("Confusion matrix:\n", cm)

    # ====== 7. Plot the confusion matrix heatmap ======
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
