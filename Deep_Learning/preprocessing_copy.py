import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from spektral.layers import GCNConv

file_path = "./이상치데이터예제.csv"

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
    df = df.drop(columns=['event_time'])
    return df

# 3. Label Encoding
def encode_categorical_data(df):
    encoder = OneHotEncoder(sparse_output=False)
    hub_encoded = encoder.fit_transform(df[['hub_type']])
    event_encoded = encoder.fit_transform(df[['event_type']])
    df = df.drop(columns=['hub_type', 'event_type'])
    df = pd.concat([df, pd.DataFrame(hub_encoded, columns=[f'hub_{i}' for i in range(hub_encoded.shape[1])]), 
                        pd.DataFrame(event_encoded, columns=[f'event_{i}' for i in range(event_encoded.shape[1])])], axis=1)
    return df

# 4. 시계열 데이터 구성
def create_sequences(df, seq_length):
    sequences = []
    feature_columns = [col for col in df.columns if col not in ['epc_code']]  # epc_code 제외
    for _, group in df.groupby('epc_code'):
        group_data = group[feature_columns].values  # 기존 컬럼 대신 One-Hot 포함 전체 컬럼 사용
        for i in range(len(group_data) - seq_length + 1):
            sequences.append(group_data[i:i + seq_length])
    return np.array(sequences)

# 5. 데이터 정규화
def normalize_data(sequences):
    scaler = StandardScaler()
    for feature_idx in range(sequences.shape[2]):
        sequences[:, :, feature_idx] = scaler.fit_transform(sequences[:, :, feature_idx])
    return sequences

# 6. 데이터 분할
def split_data(sequences, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    X_train, X_temp = train_test_split(sequences, test_size=val_ratio + test_ratio, random_state=42)
    X_val, X_test = train_test_split(X_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)
    return X_train, X_val, X_test

def preprocess_pipeline(file_path, seq_length=10):
    df = load_data(file_path)
    df = process_time_data(df)
    df = encode_categorical_data(df)
    df = df.drop(columns=['product_serial', 'product_name'])
    sequences = create_sequences(df, seq_length)
    sequences = normalize_data(sequences)
    X_train, X_val, X_test = split_data(sequences)
    return X_train, X_val, X_test