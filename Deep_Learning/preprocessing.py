import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

file_path = "./이상치데이터예제.csv"

# 1. 데이터 로드
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 2. 시간 데이터 처리
def process_time_data(df):
    # event_time을 datetime 형식으로 변환
    df['event_time'] = pd.to_datetime(df['event_time'])
    
    # 날짜 정보(연, 월, 일) 추출
    df['year'] = df['event_time'].dt.year
    df['month'] = df['event_time'].dt.month
    df['day'] = df['event_time'].dt.day

    # epc_code별 event_time 차이 계산 (초 단위)
    df['time_diff'] = df.groupby('epc_code')['event_time'].diff().dt.total_seconds().fillna(0)
    
    # time_diff 값을 로그 변환하여 스케일 조정
    df['time_diff'] = np.log1p(df['time_diff'])

    # time_diff의 다양한 단위 변환 추가
    df['time_diff_minutes'] = df['time_diff'] / 60
    df['time_diff_hours'] = df['time_diff'] / 3600
    df['time_diff_days'] = df['time_diff'] / 86400

    # 시간(hour)의 주기적 특성 인코딩
    df['hour'] = df['event_time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # 월(month)의 주기적 특성 인코딩
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # 일(day)의 주기적 특성 인코딩
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # event_time, year, month, day, hour 컬럼 삭제 (불필요한 원본 값)
    df = df.drop(columns=['event_time', 'year', 'month', 'day', 'hour'], errors='ignore')
    return df

# 3. Label Encoding
def encode_categorical_data(df):
    # hub_type, event_type 라벨 인코딩
    le_hub = LabelEncoder()
    le_event = LabelEncoder()
    df['hub_type_encoded'] = le_hub.fit_transform(df['hub_type'])
    df['event_type_encoded'] = le_event.fit_transform(df['event_type'])

    # 허브 및 이벤트의 이전 상태 추가 (순서 학습 강화)
    df['hub_type_shifted'] = df.groupby('epc_code')['hub_type_encoded'].shift(1).fillna(-1)
    df['event_type_shifted'] = df.groupby('epc_code')['event_type_encoded'].shift(1).fillna(-1)

    # 허브 이동 횟수 추가 (허브 변경 감지)
    df['hub_transition'] = (df['hub_type_encoded'] != df['hub_type_encoded'].shift(1)).astype(int)

    df = df.drop(columns=['product_serial', 'product_name'])
    df = df.drop(columns=['hub_type', 'event_type'], errors='ignore')

    return df

# 4. 시계열 데이터 구성
def create_sequences(df, seq_length):
    sequences = []
    for _, group in df.groupby('epc_code'):
        group_data = group[['time_diff', 'time_diff_minutes', 'time_diff_hours', 'time_diff_days', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                            'day_sin', 'day_cos', 'hub_type_encoded', 'event_type_encoded', 'hub_type_shifted', 'event_type_shifted', 'hub_transition']].values
        for i in range(len(group_data) - seq_length + 1):
            sequences.append(group_data[i:i + seq_length])
    return np.array(sequences)

# 5. 데이터 정규화
def normalize_data(sequences):
    # MinMaxScaler를 사용할 열들 (시간 차이 관련 변수)
    minmax_scaler = MinMaxScaler()
    minmax_indices = [0, 1, 2, 3]  # time_diff, time_diff_minutes, time_diff_hours, time_diff_days

    # StandardScaler를 사용할 열들 (주기적 시간 변수)
    standard_scaler = StandardScaler()
    standard_indices = [4, 5, 6, 7, 8, 9]  # hour_sin, hour_cos, month_sin, month_cos, day_sin, day_cos

    # MinMaxScaler 적용
    sequences[:, :, minmax_indices] = minmax_scaler.fit_transform(
        sequences[:, :, minmax_indices].reshape(-1, len(minmax_indices))
    ).reshape(sequences.shape[0], sequences.shape[1], len(minmax_indices))

    # StandardScaler 적용
    sequences[:, :, standard_indices] = standard_scaler.fit_transform(
        sequences[:, :, standard_indices].reshape(-1, len(standard_indices))
    ).reshape(sequences.shape[0], sequences.shape[1], len(standard_indices))

    # hub_type_encoded, event_type_encoded 등 범주형 변수는 정규화 제외

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
    df = df.drop(columns=['product_serial', 'product_name'], errors='ignore')
    sequences = create_sequences(df, seq_length)
    sequences = normalize_data(sequences)
    return sequences