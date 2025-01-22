import pandas as pd
import random
from datetime import datetime, timedelta

# 랜덤 데이터 생성 함수
def generate_random_datetime(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def generate_dummy_data(num_samples, rule_type):
    data = []
    for _ in range(num_samples):
        # 공통 데이터 생성
        epc_code = f"001.8805843.{random.randint(100000, 999999)}.{random.randint(1, 999999):06d}"
        hub_type = random.choice(["Yeongju", "Busan", "Daejeon"])
        event_type = random.choice(["commissioning", "aggregation", "WMS_inbound", "WMS_outbound",
                                     "stock_inbound(HUB)", "stock_outbound(HUB)", 
                                     "stock_inbound(Wholesaler)", "stock_outbound(Wholesaler)",
                                     "stock_inbound(Reseller)", "stock_outbound(Sell)"])
        event_time = generate_random_datetime(datetime(2025, 1, 1), datetime(2025, 4, 30))
        
        # 규칙 기반 데이터 생성
        if rule_type == "위조":
            # commissioning 없이 SCM 내 다른 read_point에서 시작
            if event_type != "commissioning":
                hub_type = random.choice(["Yeongju", "Busan", "Daejeon", "Daegu", "Seoul", "Gwangju"])
        
        elif rule_type == "불법":
            # 포장 해제 없이 다른 read_point에서 제품 검출
            if event_type not in ["aggregation", "WMS_inbound"]:
                hub_type = random.choice(["Busan", "Daejeon"])
        
        elif rule_type == "밀수":
            # Custom_inbound 이후 read 기록이 존재하지 않음
            if event_type == "stock_outbound(Sell)":
                event_type = "Custom_inbound"
                hub_type = random.choice(["Busan", "Daejeon"])
        
        # 데이터 추가
        data.append([epc_code, hub_type, event_type, event_time])
    return data

# 각 규칙에 맞는 데이터 생성
num_samples = 500
columns = ["epc_code", "hub_type", "event_type", "event_time"]

# 위조 데이터
forged_data = generate_dummy_data(num_samples, "위조")
forged_df = pd.DataFrame(forged_data, columns=columns)

# 불법 데이터
illegal_data = generate_dummy_data(num_samples, "불법")
illegal_df = pd.DataFrame(illegal_data, columns=columns)

# 밀수 데이터
smuggling_data = generate_dummy_data(num_samples, "밀수")
smuggling_df = pd.DataFrame(smuggling_data, columns=columns)

# 데이터 결합
final_df = pd.concat([forged_df, illegal_df, smuggling_df], ignore_index=True)

# 결과 저장
final_df.to_csv("dummy_scm_data.csv", index=False)

print(f"Generated dummy data with {len(final_df)} rows:")
print(final_df.head())
