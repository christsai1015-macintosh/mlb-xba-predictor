import pandas as pd
import numpy as np
from pybaseball import statcast
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_park_adjusted_model():
    print("🚀 正在抓取 MLB 全聯盟數據 (建議抓取約兩週以確保球場樣本足夠)...")
    # 抓取 2024 年 5 月份資料
    raw_data = statcast('2024-05-01', '2024-05-15')

    # 1. 資料清洗
    # 加入 home_team (代表球場)
    cols = ['launch_speed', 'launch_angle', 'home_team', 'events']
    df = raw_data.dropna(subset=cols).copy()

    # 2. 定義標籤 (Target)
    hit_events = ['single', 'double', 'triple', 'home_run']
    df['is_hit'] = df['events'].apply(lambda x: 1 if x in hit_events else 0)

    # 3. 特徵工程：球場 One-Hot Encoding
    # 這會把 'NYY', 'BOS' 等轉成獨立的 0/1 欄位
    stadium_dummies = pd.get_dummies(df['home_team'], prefix='stadium')
    df_final = pd.concat([df, stadium_dummies], axis=1)

    # 紀錄所有的特徵名稱，這對之後 Web App 調用至關重要
    stadium_features = stadium_dummies.columns.tolist()
    all_features = ['launch_speed', 'launch_angle'] + stadium_features

    X = df_final[all_features]
    y = df_final['is_hit']

    # 4. 訓練模型 (增加 max_iter 確保收斂)
    print(f"🧠 正在訓練模型... (特徵總數: {len(all_features)})")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # 5. 封裝並儲存
    # 我們不只存模型，連同特徵清單和球隊名單一起存成一個字典
    save_data = {
        'model': model,
        'features': all_features,
        'teams': sorted(df['home_team'].unique().tolist())
    }
    
    joblib.dump(save_data, 'xBA_park_model.joblib')
    print(f"✅ 模型已成功儲存！共包含 {len(save_data['teams'])} 個球場參數。")

if __name__ == "__main__":
    train_park_adjusted_model()