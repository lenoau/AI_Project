from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import uvicorn

# python -m uvicorn LSTM_copy:app --reload
# python -m uvicorn LSTM_copy:app --host 0.0.0.0 --port 8000 --reload

# FastAPI 앱 생성
app = FastAPI()

# 1. 저장된 모델과 임계치 불러오기
loaded_model = load_model('./LSTM_결과/임계치_0.4864.h5', custom_objects={'mae': tf.keras.losses.MeanSquaredError()})
loaded_model.compile()
with open('./LSTM_결과/임계치_0.4864.pkl', 'rb') as f:
    threshold = pickle.load(f)

# LabelEncoder 불러오기
with open('./LSTM_결과/event_type_encoder.pkl', 'rb') as f:
    le_event = pickle.load(f)

with open('./LSTM_결과/hub_type_encoder.pkl', 'rb') as f:
    le_hub = pickle.load(f)

# MinMaxScaler 불러오기
with open('./LSTM_결과/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 새로운 Label 처리
def safe_transform(le, values):
    unknown_values = set(values) - set(le.classes_)
    if unknown_values:
        print(f"Warning: 새로운 값 발견 {unknown_values}. 새로운 라벨로 추가합니다.")
        le.classes_ = np.append(le.classes_, list(unknown_values))
    return le.transform(values)

print("불러온 임계치:", threshold)

# 2. Pydantic 모델 정의
class EventRecord(BaseModel):
    epc_code: str
    product_serial: int
    product_name: Optional[str] = None
    hub_type: str
    event_type: str
    event_time: str  # ISO 형식 (예: "2025-01-21 17:54:00")

# 3. 시퀀스 생성 함수
def create_sequence(group: pd.DataFrame, max_seq_length: int, feature_columns: List[str]) -> np.ndarray:
    seq = group[feature_columns].values
    if len(seq) < max_seq_length:
        padding = np.zeros((max_seq_length - len(seq), len(feature_columns)))
        seq = np.vstack([seq, padding])
    else:
        seq = seq[:max_seq_length]
    return seq

# 4. FastAPI 엔드포인트: SpringBoot와 통신
@app.post("/detect_anomaly")
def predict(events: List[EventRecord]):
    df = pd.DataFrame([e.dict() for e in events])
    df['event_time'] = pd.to_datetime(df['event_time'])
    df = df.sort_values(by=['epc_code', 'product_serial', 'event_time'])
    df['time_delta'] = df.groupby(['epc_code', 'product_serial'])['event_time'].transform(lambda x: (x - x.min()).dt.total_seconds())
    df['event_type_enc'] = safe_transform(le_event, df['event_type'])
    df['hub_type_enc'] = safe_transform(le_hub, df['hub_type'])
    df['time_delta_scaled'] = scaler.transform(df[['time_delta']])

    max_seq_length = 10
    feature_columns = ['event_type_enc', 'hub_type_enc', 'time_delta_scaled']
    grouped = df.groupby(['epc_code', 'product_serial'])
    sequences = grouped.apply(lambda x: create_sequence(x, max_seq_length, feature_columns))

    if sequences.empty:
        return []

    X_input = np.stack(sequences.values)
    X_pred = loaded_model.predict(X_input)
    
    # 개별 이벤트 Reconstruction Error 계산
    reconstruction_errors = np.mean(np.square(X_pred - X_input), axis=2)  # 이벤트별 에러 계산

    # 임계치 적용하여 개별 이벤트 이상 여부 판단
    anomaly_flags = reconstruction_errors > threshold

    results = []
    for i, ((epc_code, product_serial), group) in enumerate(grouped):
        group_data = group.drop(columns=['time_delta', 'event_type_enc', 'hub_type_enc', 'time_delta_scaled']).copy()
        
        # 각 이벤트마다 이상 여부 적용
        group_data["is_anomaly"] = anomaly_flags[i]
        results.extend(group_data.to_dict(orient='records'))

    return sorted(results, key=lambda x: (x['epc_code'], x['product_serial'], x['event_time']))


@app.get("/")
async def read_root():
    return {"message": "LSTM모델"}
