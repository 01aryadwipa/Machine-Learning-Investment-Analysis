import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import re

# Title and description
st.markdown("<h1 style='text-align: center; margin-bottom: 50px;'>Machine Learning Investment Analysis</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar: Panduan Penggunaan
st.sidebar.title("Machine Learning Investment Analysis")
st.sidebar.markdown("### 📌 Panduan Penggunaan")
st.sidebar.info("""
1️⃣ **Pilih Saham**  
   - Gunakan dropdown untuk memilih saham yang ingin dianalisis.  

2️⃣ **Pilih Model Machine Learning**  
   - Tentukan model prediksi yang ingin digunakan.  
   - Model yang tersedia:  
     - **Random Forest** (Akurasi tinggi, kompleks)  
     - **Gradient Boosting** (Bagus untuk data non-linear)  
     - **XGBoost** (Performa tinggi untuk dataset besar)  
     - **Linear Regression** (Sederhana, interpretatif)  
     - **Lasso/Ridge Regression** (Untuk mengurangi multikolinearitas)  

3️⃣ **Analisis Data & Visualisasi**  
   - Pilih variabel untuk divisualisasikan.  
   - Lihat tren data dari waktu ke waktu.  

4️⃣ **Prediksi Harga Saham**  
   - Model akan memprediksi **harga wajar saham di kuartal berikutnya**.  
   - Hasil prediksi akan ditampilkan di bagian bawah.  

⚠️ **Catatan:**  
Hasil prediksi adalah perkiraan berdasarkan **model machine learning**. Keputusan investasi tetap menjadi tanggung jawab pengguna sepenuhnya.
""")

# 📌 STOCK SELECTION DROPDOWN
st.markdown("### 📌 Pilih Saham")
stock_options = {
    "BBCA - PT Bank Central Asia Tbk": "bbca.xlsx",
    "BBRI - PT Bank Rakyat Indonesia Tbk": "bbri.xlsx",
    "TLKM - PT Telkom Indonesia Tbk": "tlkm.xlsx"
}

selected_stock = st.selectbox("Pilih Saham untuk Analisis", list(stock_options.keys()))
data_file = stock_options[selected_stock]  # Get the corresponding file name

# Try to load from local file first, fallback to GitHub if not found
try:
    data = pd.read_excel(data_file)
except FileNotFoundError:
    github_base_url = "https://raw.githubusercontent.com/01aryadwipa/Machine-Learning-Investment-Analysis/main/"
    data_url = github_base_url + data_file
    data = pd.read_excel(data_url)

st.write(f"### 📊 Pratinjau Data ({selected_stock})")
st.write(data.head(10))

# Function to parse the quarter column correctly
def parse_quarter(quarter_str):
    match = re.match(r'(q\d)_(\d+)', quarter_str.lower())  # Match 'qX_YYYY'
    if match:
        quarter, year = match.groups()
        return (int(year), quarter)  # Sort by year first, then quarter
    else:
        return (0, quarter_str)  # If format is invalid, keep it at the bottom

# Ensure the dataset has a 'quarter' column
if 'quarter' not in data.columns:
    st.error("Dataset harus memiliki kolom 'quarter' untuk visualisasi yang tepat.")
else:
    # Reverse the dataset order to ensure Q1 2009 is at the top and Q4 2024 at the bottom
    data = data.iloc[::-1].reset_index(drop=True)

    # Sort quarters in **ascending order** (Q1 2009 → Q4 2024)
    sorted_quarters = sorted(data['quarter'].unique(), key=parse_quarter)
    data['quarter'] = pd.Categorical(data['quarter'], categories=sorted_quarters, ordered=True)

    # 📈 Visualization Section
    st.subheader("📈 Visualisasi Data")
    available_metrics = sorted([col for col in data.columns if col not in ["quarter", "price"]])
    selected_metric = st.selectbox("📌 Pilih Variabel untuk Divisualisasikan", available_metrics)
    
    if selected_metric:
        metric_data = data[['quarter', selected_metric]].copy()
        metric_data.set_index('quarter', inplace=True)
        st.write(f"📊 **Perkembangan {selected_metric.upper()} dari Waktu ke Waktu**")
        st.line_chart(metric_data)

    # 🤖 Machine Learning Section
    st.subheader("🤖 Analisis Machine Learning")

    model_options = {
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "XGBoost": XGBRegressor(),
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(),
        "Ridge Regression": Ridge()
    }

    selected_model_name = st.selectbox("📌 Pilih Model Machine Learning", list(model_options.keys()))
    selected_model = model_options[selected_model_name]

    features = [col for col in data.columns if col not in ['quarter', 'price']]
    target = 'price'

    if all(col in data.columns for col in [target] + features):
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        grid_search = GridSearchCV(
            estimator=selected_model,
            param_grid={"Random Forest": {'n_estimators': [50, 100, 200]},
                        "Gradient Boosting": {'n_estimators': [50, 100, 200]},
                        "XGBoost": {'n_estimators': [50, 100, 200]},
                        "Lasso Regression": {'alpha': [0.01, 0.1, 1.0]},
                        "Ridge Regression": {'alpha': [0.01, 0.1, 1.0]},
                        "Linear Regression": {}}.get(selected_model_name, {}),
            cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        mae = round(mean_absolute_error(y_test, y_pred), 2)
        mape = round(np.mean(np.abs((y_test - y_pred) / y_test)) * 100, 2)
        r2 = round(r2_score(y_test, y_pred), 2)

        st.write(f"### 📊 Metrik Kinerja Model ({selected_model_name})")
        st.write(f"✅ Parameter Terbaik: {grid_search.best_params_}")
        st.write(f"📉 RMSE: Rp {rmse}")
        st.write(f"📉 MAE: Rp {mae}")
        st.write(f"📉 MAPE: {mape}%")
        st.write(f"📈 R²: {r2}")

        # Feature importance (for tree-based models)
        if selected_model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            feature_importances = pd.DataFrame({
            "Feature": X.columns,
            "Importance": best_model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

        st.write("### 🔥 Feature Importance")
        st.write(feature_importances)        

#        st.subheader("📈 Perkiraan Harga Wajar untuk Periode Selanjutnya")
#        latest_row = data.iloc[-1][features].values.reshape(1, -1)
#        predicted_next_price = best_model.predict(latest_row)[0]
#        st.write(f"📌 **Perkiraan Harga Wajar Saham ({selected_stock}):** **{round(predicted_next_price, 2)}**")
        st.subheader("📈 Perkiraan Harga Wajar untuk Periode Selanjutnya")

        latest_row = data.iloc[-1][features].values.reshape(1, -1)
        predicted_next_price = best_model.predict(latest_row)[0]

        last_quarter = data['quarter'].iloc[-1]
        match = re.match(r'(q)(\d)_(\d+)', last_quarter.lower())

        if match:
            q, quarter_num, year = match.groups()
            quarter_num, year = int(quarter_num), int(year)
            next_quarter = f"Q{1 if quarter_num == 4 else quarter_num + 1}_{year + (1 if quarter_num == 4 else 0)}"
            st.write(f"📌 **Perkiraan Harga Wajar Saham untuk Periode {next_quarter} ({selected_model_name}):** **{round(predicted_next_price, 2)}**")
        st.write("")
        st.write("")
        st.markdown("### ⚠️ Disclaimer")
        st.markdown("""
                Hasil analisis yang disajikan pada website ini murni hanya untuk tujuan informasi dan edukasi berdasarkan sudut pandang machine learning. 
                Hasil analisis ini bukan untuk tujuan saran, rekomendasi, ajakan, dorongan, ataupun tekanan untuk melakukan keputusan investasi, baik itu pembelian maupun penjualan suatu instrumen investasi. 
                Analisis ini tidak menjamin kepastian hasil, melainkan hanya merupakan perkiraan berdasarkan pemodelan machine learning.

                Investasi memiliki berbagai risiko, yang mungkin tidak tercerminkan dalam dataset yang digunakan. Risiko ini dapat berupa pengaruh sentimen pasar, dinamika sosial-ekonomi-politik, perubahan struktur manajemen atau kebijakan operasional perusahaan, kejadian luar biasa (force majeure), serta variabel-variabel lainnya, termasuk yang sulit diperoleh ataupun sulit dikonversi/dikuantifikasi untuk pemodelan. 
                
                Dengan demikian, pengembang website menyatakan bahwa pemodelan yang telah dilakukan masih memiliki berbagai keterbatasan sebagaimana yang telah dijabarkan, dan hasil pemodelan ini tidak dianjurkan untuk menjadi satu-satunya dasar pengambilan keputusan investasi, melainkan hanya sebagai sekadar referensi tambahan dalam konteks penggunaan metode machine learning. 
                
                **Segala keputusan investasi merupakan tanggung jawab pengguna sepenuhnya.**
                """)
st.markdown("---")
