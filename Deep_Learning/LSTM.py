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

# python -m uvicorn LSTM:app --reload

# FastAPI 앱 생성
app = FastAPI()

# 1. 저장된 모델과 임계치 불러오기
loaded_model = load_model('./LSTM_결과/임계치_0.4864.h5', custom_objects={'mae': tf.keras.losses.MeanSquaredError()})
with open('./LSTM_결과/임계치_0.4864.pkl', 'rb') as f:
    threshold = pickle.load(f)
print("불러온 임계치:", threshold)

# 2. Pydantic 모델 정의 (입력 및 출력 스키마)

class EventRecord(BaseModel):
    epc_code: str
    product_serial: int
    product_name: Optional[str] = None
    hub_type: str
    event_type: str
    event_time: str  # ISO 형식의 문자열 (예: "2025-01-21 17:54:00")

class PredictionResult(BaseModel):
    epc_code: str
    is_anomaly: bool

class PredictionResponse(BaseModel):
    results: List[PredictionResult]

# 2. 시퀀스 생성 함수 (학습 시와 동일)
def create_sequence(group: pd.DataFrame, max_seq_length: int, feature_columns: List[str]) -> np.ndarray:
    
   #그룹 데이터에서 지정된 feature_columns만 사용하여 시퀀스 배열을 생성합니다.
   #시퀀스 길이가 max_seq_length보다 작으면 0으로 패딩합니다.
   
    seq = group[feature_columns].values  # shape: (num_events, num_features)
    if len(seq) < max_seq_length:
        padding = np.zeros((max_seq_length - len(seq), len(feature_columns)))
        seq = np.vstack([seq, padding])
    else:
        seq = seq[:max_seq_length]
    return seq

# 3. FastAPI 엔드포인트: /predict
@app.post("/predict", response_model=PredictionResponse)
def predict(events: List[EventRecord]):
    
    # SpringBoot에서 JSON 형식으로 전달한 이벤트 데이터를 받아서
    # 전처리 후 저장된 LSTM Autoencoder 모델로 이상 탐지를 수행합니다.
    # 각 그룹(제품 단위: epc_code, product_serial)별로 이상 여부를 판단하여 반환합니다.
    
    # (1) JSON 데이터를 DataFrame으로 변환
    df = pd.DataFrame([e.dict() for e in events])
    
    # (2) 날짜 변환 및 정렬
    df['event_time'] = pd.to_datetime(df['event_time'])
    df = df.sort_values(by=['epc_code', 'product_serial', 'event_time'])
    
    # (3) 각 그룹별 첫 이벤트 기준으로 시간 차이(time_delta) 계산
    df['time_delta'] = df.groupby(['epc_code', 'product_serial'])['event_time'] \
                         .transform(lambda x: (x - x.min()).dt.total_seconds())
    
    # (4) 범주형 변수 인코딩 (학습 시 사용한 인코더로 변환)
    # 만약 입력 데이터에 학습 시 정의되지 않은 카테고리가 있다면 에러가 발생할 수 있음
    # 학습 데이터로 LabelEncoder 학습
    le_event = LabelEncoder()
    df['event_type_enc'] = le_event.fit_transform(df['event_type'])

    # 허브 타입(hub_type) 인코딩
    le_hub = LabelEncoder()
    df['hub_type_enc'] = le_hub.fit_transform(df['hub_type'])
        
    # (5) 수치형 변수 정규화 (time_delta)
    scaler = MinMaxScaler()
    df['time_delta_scaled'] = scaler.fit_transform(df[['time_delta']])
    
    print("전처리 후 데이터 샘플:")
    print(df[['epc_code', 'product_serial', 'event_time', 'time_delta', 
          'event_type', 'event_type_enc', 'hub_type', 'hub_type_enc', 'time_delta_scaled']].head(10))
    
    # 시퀀스 길이와 사용 피처 정의 (학습 시와 동일)
    max_seq_length = 10
    feature_columns = ['event_type_enc', 'hub_type_enc', 'time_delta_scaled']
    
    # (6) 그룹화: epc_code, product_serial 기준으로 시퀀스 생성
    grouped = df.groupby(['epc_code', 'product_serial'])
    sequences = grouped.apply(lambda x: create_sequence(x, max_seq_length, feature_columns))
    
    # 만약 그룹이 하나도 없으면 빈 응답 반환
    if sequences.empty:
        return PredictionResponse(results=[])
    
    X_input = np.stack(sequences.values)
    
    # (7) 모델 예측: 각 그룹에 대해 재구성 오차 계산
    X_pred = loaded_model.predict(X_input)
    reconstruction_errors = np.mean(np.square(X_pred - X_input), axis=(1,2))
    
    # (8) 저장된 임계치와 비교하여 이상 여부 판단
    anomalies = reconstruction_errors > threshold
    
    # (9) 각 그룹의 키는 MultiIndex (epc_code, product_serial)
    results = []
    for (epc_code, product_serial), is_anomaly in zip(sequences.index, anomalies):
        results.append(PredictionResult(
            epc_code=epc_code,
            product_serial=product_serial,
            is_anomaly=bool(is_anomaly)
        ))
    
    return PredictionResponse(results=results)

# 4. FastAPI 서버 실행
@app.get("/")
async def read_root():
    return {"message": "LSTM프로젝트"}
