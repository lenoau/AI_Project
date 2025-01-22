import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. 데이터 로드
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 2. 시간 데이터 처리
def process_time_data(df):
    # event_time을 datetime 형식으로 변환
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # epc_code별로 시간 차이 계산
    df['time_diff'] = df.groupby('epc_code')['event_time'].diff().dt.total_seconds().fillna(0)
    
    # 시간의 순환적 특성 인코딩
    df['hour'] = df['event_time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

# 3. Label Encoding
def encode_categorical_data(df):
    # Label Encoding for hub_type and event_type
    le_hub = LabelEncoder()
    le_event = LabelEncoder()
    df['hub_type_encoded'] = le_hub.fit_transform(df['hub_type'])
    df['event_type_encoded'] = le_event.fit_transform(df['event_type'])
    return df

# 4. 시계열 데이터 구성
def create_sequences(df, seq_length):
    sequences = []
    for _, group in df.groupby('epc_code'):
        group_data = group[['time_diff', 'hub_type_encoded', 'event_type_encoded']].values
        for i in range(len(group_data) - seq_length + 1):
            sequences.append(group_data[i:i + seq_length])
    return np.array(sequences)

# 5. 데이터 정규화
def normalize_data(sequences):
    scaler = MinMaxScaler()
    for feature_idx in range(sequences.shape[2]):
        sequences[:, :, feature_idx] = scaler.fit_transform(sequences[:, :, feature_idx])
    return sequences

# 6. 데이터 분할
def split_data(sequences, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    X_train, X_temp = train_test_split(sequences, test_size=val_ratio + test_ratio, random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)
    return X_train, X_val, X_test

# 7. 전체 파이프라인 실행 함수
def preprocess_pipeline(file_path, seq_length=10):
    # 데이터 로드
    df = load_data(file_path)
    
    # 시간 데이터 처리
    df = process_time_data(df)

    # Label Encoding
    df = encode_categorical_data(df)

    # 시계열 데이터 생성
    sequences = create_sequences(df, seq_length)

    # 데이터 정규화
    sequences = normalize_data(sequences)

    # 데이터 분할
    X_train, X_val, X_test = split_data(sequences)

    return X_train, X_val, X_test

# 8. 실행
file_path = "./Deep_Learning/sampled_data.csv"
seq_length = 10
X_train, X_val, X_test = preprocess_pipeline(file_path, seq_length)

# 결과 확인
print("Preprocessing complete!")