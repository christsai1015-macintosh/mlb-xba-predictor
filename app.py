import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# --- 網頁設定 ---
st.set_page_config(page_title="xBA 打擊模擬器", page_icon="⚾")

# 設定標題與簡介
st.title("⚾ MLB 期望安打率 (xBA) 計算機")
st.markdown("""
這個網頁是由中研院統計所暑期實習申請者（工管系）所開發。
它利用 2024 年 5 月的 Statcast 大數據訓練 Logistic Regression 模型，
根據您輸入的**擊球初速**與**仰角**，預測該球形成安打的機率。
""")

# --- 1. 載入模型 (加入快取機制，提升網頁效率) ---
@st.cache_resource  # 這樣模型只需要載入一次，不用每次整理網頁都重載
def load_my_model():
    model_path = 'xBA_model.joblib'
    if not os.path.exists(model_path):
        st.error(f"找不到模型檔案 '{model_path}'。請先執行訓練模型的程式碼。")
        st.stop()
    return joblib.load(model_path)

model = load_my_model()

# --- 2. 側邊欄輸入介面 (讓使用者輸入數據) ---
st.sidebar.header("📊 輸入擊球參數")

# 使用滑桿 (Slider) 或數字輸入框 (Number Input)
exit_velocity = st.sidebar.slider(
    "擊球初速 (Exit Velocity, mph)", 
    min_value=60, 
    max_value=120, 
    value=105, 
    step=1,
    help="大聯盟平均約 88-90 mph"
)

launch_angle = st.sidebar.slider(
    "擊球仰角 (Launch Angle, degrees)", 
    min_value=-30, 
    max_value=60, 
    value=25, 
    step=1,
    help="平飛球約 10-25 度，高飛球 25-50 度"
)

# --- 3. 計算與展示結果 ---
if st.sidebar.button("開始預測", type="primary"):
    
    # 準備輸入數據：形狀必須是 [[speed, angle]]
    input_data = np.array([[exit_velocity, launch_angle]])
    
    # 使用模型預測機率 (Class 1 的機率)
    hit_prob = model.predict_proba(input_data)[0][1]
    
    # 顯示結果區塊
    st.subheader("🔮 預測結果")
    
    # 用醒目的 Metric 展示數字
    st.metric(label="期望安打率 (xBA)", value=f"{hit_prob:.1%}")
    
    # 根據機率給予動態評語 (工管系的結果詮釋力)
    if hit_prob >= 0.9:
        st.success("🔥 這是顆火箭般的 Barrel！防守者幾乎不可能接到。")
    elif 0.7 <= hit_prob < 0.9:
        st.success("✅ 擊球品質極佳，高機率落地形成安打。")
    elif 0.3 <= hit_prob < 0.7:
        st.warning("⚖️ 勝負未定。這取決於防守站位與球場大小。")
    else:
        st.error("📉 很遺憾，統計上這是一個高機率出局的球。")
        
    # --- 進階：視覺化呈現 ---
    # 這裡可以用簡單的 Pandas DataFrame 展示輸入
    st.markdown("---")
    st.write("您輸入的數據：")
    df_display = pd.DataFrame(input_data, columns=['初速 (mph)', '仰角 (deg)'])
    st.dataframe(df_display.style.format("{:.0f}"))

else:
    st.info("👈 請在左側輸入擊球參數，並點擊「開始預測」。")

# 網頁頁尾
st.markdown("---")
st.caption("Disclaimer: 此模型僅供學術演示使用，數據來源為 MLB Statcast 公開資料。")
