import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 網頁配置 ---
st.set_page_config(page_title="MLB 球場修正預測器", page_icon="🏟️")

@st.cache_resource
def load_assets():
    # 載入模型包
    return joblib.load('xBA_park_model.joblib')

try:
    assets = load_assets()
    model = assets['model']
    features = assets['features']
    teams = assets['teams']
except:
    st.error("❌ 找不到模型檔案！請先執行 python baseball_data.py 進行訓練。")
    st.stop()

# --- UI 介面 ---
st.title("🏟️ xBA 預測器 (含球場因子修正)")
st.write("本系統利用 Logistic Regression 評估不同球場對擊球結果的影響。")

st.sidebar.header("📥 輸入擊球數據")

ev = st.sidebar.slider("擊球初速 (Exit Velocity, mph)", 60, 120, 100)
la = st.sidebar.slider("擊球仰角 (Launch Angle, deg)", -30, 60, 15)
selected_team = st.sidebar.selectbox("選擇比賽球場 (主場球隊)", teams)

# --- 預測邏輯 ---
if st.sidebar.button("分析擊球機率", type="primary"):
    # 建立一個與訓練時相同長度的全 0 向量 (DataFrame 格式)
    input_df = pd.DataFrame(0, index=[0], columns=features)
    
    # 填入數值
    input_df['launch_speed'] = ev
    input_df['launch_angle'] = la
    
    # 填入對應的球場 One-Hot
    stadium_col = f"stadium_{selected_team}"
    if stadium_col in input_df.columns:
        input_df[stadium_col] = 1
    
    # 計算機率
    prob = model.predict_proba(input_df)[0][1]
    
    # 顯示結果
    st.subheader(f"📍 在 {selected_team} 主場的預測結果")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("期望安打率 (xBA)", f"{prob:.1%}")
    
    with col2:
        # 計算不含球場修正的基礎機率 (模擬平均球場) 作為比較
        # 這裡簡單以不勾選任何球場來模擬
        base_df = pd.DataFrame(0, index=[0], columns=features)
        base_df['launch_speed'] = ev
        base_df['launch_angle'] = la
        base_prob = model.predict_proba(base_df)[0][1]
        diff = prob - base_prob
        st.metric("球場對結果的增益/減損", f"{diff:+.1%}")

    # 專業註解
    st.info(f"💡 註：在統計模型中，這球在 {selected_team} 的成功率比聯盟平均{'高' if diff > 0 else '低'}。")

st.markdown("---")
st.caption("工管系專題實作：運動統計與決策分析工具")