import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 데이터 로드
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 2. 시간 데이터 처리: datetime 변환, 정렬, 시간 차 및 추가 피처 생성
def process_time_data(df):
    # event_time을 datetime 형식으로 변환 및 정렬
    df['event_time'] = pd.to_datetime(df['event_time'])
    df = df.sort_values(['epc_code', 'event_time'])
    
    # epc_code별 시간 차 계산 및 로그 변환 (분포 치우침 보정)
    df['time_diff'] = df.groupby('epc_code')['event_time'].diff().dt.total_seconds().fillna(0)
    df['log_time_diff'] = np.log1p(df['time_diff'])
    
    # 시간의 순환적 특성 인코딩: 시간 (시) 정보를 sine, cosine 값으로 변환
    df['hour'] = df['event_time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 추가 시간 피처: 요일과 주말 여부
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df

# 3. 범주형 데이터 인코딩: hub_type, event_type, 및 결합 피처 인코딩
def encode_categorical_data(df):
    le_hub = LabelEncoder()
    le_event = LabelEncoder()
    df['hub_type_encoded'] = le_hub.fit_transform(df['hub_type'])
    df['event_type_encoded'] = le_event.fit_transform(df['event_type'])
    
    df['hub_event_combined'] = df['hub_type'] + '_' + df['event_type']
    le_combined = LabelEncoder()
    df['hub_event_encoded'] = le_combined.fit_transform(df['hub_event_combined'])
    
    return df

# 4. 시계열 데이터 구성: 선택 피처를 기반으로 시퀀스 생성 (각 시퀀스에 해당 epc_code 그룹 정보도 반환)
def create_sequences(df, seq_length):
    # 사용할 피처 목록 (여기서는 log_time_diff, 시간의 sine/cosine, 요일, 주말 여부, 그리고 인코딩된 범주형 피처 사용)
    feature_cols = ['log_time_diff', 'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend',
                    'hub_type_encoded', 'event_type_encoded', 'hub_event_encoded']
    sequences = []
    groups = []  # 각 시퀀스가 속하는 그룹 (epc_code)
    
    # epc_code별 그룹화 후, 시간 순으로 정렬하여 슬라이딩 윈도우 방식으로 시퀀스 생성
    for epc, group in df.groupby('epc_code'):
        group = group.sort_values('event_time')
        group_data = group[feature_cols].values
        if len(group_data) < seq_length:
            continue  # 시퀀스 길이보다 짧은 경우 건너뜀
        for i in range(len(group_data) - seq_length + 1):
            sequences.append(group_data[i:i+seq_length])
            groups.append(epc)
    return np.array(sequences), groups


# 5. 그룹 기반 데이터 분할: 동일 그룹(epc_code)의 시퀀스가 훈련/검증/테스트에 섞이지 않도록 분할
def split_data_by_group(sequences, groups, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    groups = np.array(groups)
    # GroupShuffleSplit을 이용해 훈련 세트와 나머지(임시) 세트로 분할
    gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=42)
    train_idx, temp_idx = next(gss.split(sequences, groups=groups))
    X_train = sequences[train_idx]
    groups_temp = groups[temp_idx]
    
    # 임시 세트에서 unique한 그룹을 기준으로 검증과 테스트 세트로 나눔
    unique_temp_groups = np.unique(groups_temp)
    val_groups, test_groups = train_test_split(unique_temp_groups, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
    
    # 임시 인덱스 중 해당 그룹에 해당하는 인덱스를 선택
    val_idx = [idx for idx in temp_idx if groups[idx] in val_groups]
    test_idx = [idx for idx in temp_idx if groups[idx] in test_groups]
    
    X_val = sequences[val_idx]
    X_test = sequences[test_idx]
    return X_train, X_val, X_test

# 6. 데이터 정규화: 훈련 데이터에 피팅한 StandardScaler를 활용하여 전체 데이터를 정규화
def normalize_data(X_train, X_val, X_test):
    num_features = X_train.shape[2]
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, num_features)
    scaler.fit(X_train_flat)
    
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# 7. 전체 파이프라인 실행: 데이터 로드 → 전처리 → 시퀀스 구성 → 그룹 분할 → 정규화
def preprocess_pipeline(file_path, seq_length=10):
    df = load_data(file_path)
    df = process_time_data(df)
    df = encode_categorical_data(df)
    # 불필요한 컬럼 제거 (존재하지 않는 컬럼이 있을 경우 에러 방지)
    df = df.drop(columns=['product_serial', 'product_name'], errors='ignore')
    
    sequences, groups = create_sequences(df, seq_length)
    print("Total sequences created:", sequences.shape)
    
    X_train, X_val, X_test = split_data_by_group(sequences, groups, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    print("Train shape:", X_train.shape, "Val shape:", X_val.shape, "Test shape:", X_test.shape)
    
    X_train, X_val, X_test, scaler = normalize_data(X_train, X_val, X_test)
    return X_train, X_val, X_test, scaler