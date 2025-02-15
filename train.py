import pandas as pd
import cupy as cp 
from utils.dataset import CARD_LOADER
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from tqdm import tqdm  # 進度條
from xgboost import DMatrix
from xgboost.callback import TrainingCallback
from xgboost.callback import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np  # 用於 CuPy -> NumPy 轉換


class TqdmCallback(TrainingCallback):

    def __init__(self, total_rounds):
        self.pbar = tqdm(total=total_rounds, desc="Training Progress")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False  # False 表示不停止訓練

    def __del__(self):
        self.pbar.close()

def train_model(X_train, y_train, X_val, y_val):


    # 將數據轉換為 DMatrix 格式，適用於 xgboost.train()
    dtrain = DMatrix(X_train, label=y_train)
    dval = DMatrix(X_val, label=y_val)

    # 設定參數
    params = {
        'booster':'gbtree',
        'objective': 'binary:logistic',  # 二元分類問題
        'eval_metric': 'logloss',  
        'learning_rate': 0.01,  # 學習率 
        'max_depth': 9,  # 樹的最大深度 
        'min_child_weight': 2,  
        'gamma': 0.1,  
        'subsample': 0.9,  
        'colsample_bytree': 0.5,  
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum()  ,  # 處理類別不平衡
        'tree_method': 'hist',  # 使用 GPU 訓練
        'random_state': 7,  
        'n_jobs': -1,  # 使用所有可用 CPU
        'gpu_id': 0,  
    }

    # 早停
    early_stopping = EarlyStopping(rounds=10, metric_name='logloss', data_name='eval')

    # 設定評估資料
    evals = [(dtrain, 'train'), (dval, 'eval')]

    # 訓練模型並顯示進度
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=50000,
        evals=evals,
        early_stopping_rounds=5000,
        verbose_eval=True,
        callbacks=[early_stopping]
    )

    return model

def evaluate_model(model, X_val, y_val, threshold=0.4):
    
    dval = DMatrix(X_val)

    y_probs = model.predict(dval)
    y_pred_threshold = (y_probs >= threshold).astype(int)

    y_val_numpy = cp.asnumpy(y_val)
    print("分類報告:")
    print(classification_report(y_val_numpy, y_pred_threshold))

    fpr_val, tpr_val, _ = roc_curve(y_val_numpy, y_probs)
    auc_score = auc(fpr_val, tpr_val)
    print(f"AUC = {auc_score:.4f}")

def predict_and_save(model, X_test, X_test_cp, X_val, output_file='output13.csv'):
    
    dtest = DMatrix(X_test_cp)
    dval = DMatrix(X_val)
    # 進行預測，並將概率轉換為 0 或 1
    y_test_pred = (model.predict(dtest) >= 0.4).astype(int)
    y_val_pred = (model.predict(dval) >= 0.4).astype(int)
    # 將預測結果存儲到 output.csv
    output_df = pd.DataFrame({'index': X_test.index, 'label': y_test_pred})
    output_df.to_csv(output_file, index=False)
    print(f"預測結果已儲存至 {output_file}")

    return y_val_pred 

def plot_confusion_matrix(y_train_full, y_train_pred, title="Confusion Matrix"):
    y_train_full_np = cp.asnumpy(y_train_full)
    y_train_pred_np = cp.asnumpy(y_train_pred)
    
    # 計算混淆矩陣
    cm = confusion_matrix(y_train_full_np, y_train_pred_np)

    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

    plt.title(title)
    plt.show()

   


def main():
    dataloader = CARD_LOADER()
    X_train_full, y_train_full, X_test = dataloader.get_processed_data()
    print(f"訓練資料大小: {X_train_full.shape}, 測試資料大小: {X_test.shape}")

    # 將數據分割成訓練集和驗證集
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=0)


    # 將數據轉換為 CuPy 陣列進行 GPU 加速訓練
    X_train = cp.array(X_train)
    y_train = cp.array(y_train)
    X_val = cp.array(X_val)
    y_val = cp.array(y_val)
    X_test_cp = cp.array(X_test)
    X_train_full_cp = cp.array(X_train_full)

    model = train_model(X_train, y_train, X_val, y_val)  # 使用驗證集進行訓練

    evaluate_model(model, X_val, y_val)
    Y = predict_and_save(model, X_test, X_test_cp, X_val)
    plot_confusion_matrix(y_val, Y)

if __name__ == "__main__":
    main()

