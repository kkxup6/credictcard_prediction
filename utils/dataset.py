import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm  # 匯入 tqdm

class CARD_LOADER:
    def __init__(self, train_path='train.csv', test_path='X_test.csv'):
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def load_data(self):
        """讀取訓練和測試資料。"""
        print("Loading data...")
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)

    def fill_missing_values(self):
        """填補缺失值。"""
        print("Filling missing values...")
        numeric_cols = [
            'locdt', 'loctm', 'contp', 'etymd', 'mcc', 'conam', 'ecfg',
            'insfg', 'iterm', 'bnsfg', 'flam1', 'stocn', 'scity',
            'ovrlt', 'flbmk', 'hcefg', 'csmcu', 'csmam', 'flg_3dsmk'
        ]
        categorical_cols = ['txkey', 'chid', 'cano', 'mchno', 'acqic']

        for col in tqdm(numeric_cols, desc="Processing numeric columns"):
            self.train_data[col] = self.train_data[col].fillna(self.train_data[col].median())

        for col in tqdm(categorical_cols, desc="Processing categorical columns"):
            mode_value = self.train_data[col].mode()[0]
            self.train_data[col] = self.train_data[col].fillna(mode_value)

    def encode_labels(self):
        """對類別型欄位進行標籤編碼和頻率編碼。"""
        print("Encoding labels...")

        label_cols = ['txkey', 'chid', 'cano', 'mchno', 'acqic']
        
        # 標籤編碼
        for col in tqdm(label_cols, desc="Label encoding"):
            self.label_encoder.fit(self.train_data[col])
            self.train_data[f'{col}_code'] = self.label_encoder.transform(self.train_data[col])
            # 對測試資料進行編碼，使用訓練資料的映射
            self.test_data[f'{col}_code'] = self.test_data[col].map(lambda x: self.label_encoder.transform([x])[0] if x in self.label_encoder.classes_ else -1)

        # 使用 Frequency Encoding 處理 ['contp', 'etymd']
        freq_cols = ['contp', 'etymd']
        
        for col in tqdm(freq_cols, desc="Frequency encoding"):
            freq_map = self.train_data[col].value_counts().to_dict()
            self.train_data[f'{col}_freq'] = self.train_data[col].map(freq_map)
            self.test_data[f'{col}_freq'] = self.test_data[col].map(freq_map).fillna(0)  # 對測試資料進行頻率編碼

    def drop_unused_columns(self):
        """移除不需要的欄位。"""
        print("Dropping unused columns...")
        self.train_data = self.train_data.drop(['stscd'], axis=1)

    def one_hot_encode_features(self, feature_columns):
        """對指定特徵欄位進行 One-Hot Encoding。"""
        print("Performing One-Hot Encoding...")
        return pd.get_dummies(self.train_data[feature_columns])

    def get_processed_data(self):
        """主流程：讀取、處理和回傳訓練和測試資料集。"""
        self.load_data()
        self.fill_missing_values()
        self.encode_labels()
        self.drop_unused_columns()

        features = [
            'txkey_code', 'locdt', 'loctm', 'chid_code', 'cano_code', 'contp_freq', 'etymd_freq',
            'mchno_code', 'acqic_code', 'mcc', 'conam', 'ecfg', 'insfg', 'iterm',
            'bnsfg', 'flam1', 'stocn', 'scity', 'ovrlt', 'flbmk', 'hcefg',
            'csmcu', 'csmam', 'flg_3dsmk'
        ]

        # 確認特徵是否存在
        missing_features = [col for col in features if col not in self.train_data.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
        else:
            print("All features are present.")
        
        # 提取特徵和標籤
        X_train_full = self.train_data[features]
        y_train_full = self.train_data['label']  # 確保標籤存在

        # 回傳訓練特徵、標籤與測試特徵
        return X_train_full, y_train_full, self.test_data[features]
        #return X_train_full, y_train_full, self.test_data[features], X_train_full.shape[1]