import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time  # For tracking execution time

# Main screen title
st.markdown("<h1 style='text-align: center; margin-bottom: 50px;'>Machine Learning Investment Analysis</h1>", unsafe_allow_html=True)

# Sidebar for title and file upload
st.sidebar.title("Welcome to Machine Learning Investment Analysis")

# File upload section
uploaded_file = st.sidebar.file_uploader("Upload your dataset (Excel or CSV)", type=["xlsx", "csv"], help="Limit 200MB per file • XLSX, CSV")

# Navigation bar for language selection
st.sidebar.markdown("### Select Language:")
language = st.sidebar.radio("", ["Indonesian", "English"], horizontal=True)

# Display usage instructions based on selected language
if language == "Indonesian":
    st.sidebar.markdown("### Cara Menggunakan Website:")
    st.sidebar.write("""
    1. **Masukkan file data keuangan** pada box file upload di atas.  
       Format dataset dan contoh-contohnya dapat diakses pada [link berikut](https://drive.google.com/drive/folders/1fJp8NOyLTMmiZQ6gpgcscn2IMf9TohuD?usp=drive_link).
    2. **Dataset Preview**: Anda dapat melihat dan memeriksa data pada bagian Dataset Preview.
    3. **Data Visualization**: Anda dapat melakukan visualisasi data pada bagian Data Visualization.
    4. **Analisis Machine Learning**:
       - Pilih rasio keuangan yang ingin digunakan untuk analisis dari daftar *Select ratios for analysis*.
       - Pilih kolom target (harga saham perusahaan) dari dropdown *Select the target column*.
       - Klik *Run Model and Predict Stock Prices* untuk menjalankan model machine learning dan melihat hasil prediksi.
    5. **Hasil Analisis Machine Learning**:
       - Website akan menampilkan metrik evaluasi model, seperti RMSE, MAE, MAPE, dan R².
       - Anda dapat melihat tabel perbandingan antara *Actual vs Predicted Stock Prices*.
       - Prediksi harga saham untuk periode berikutnya akan ditampilkan di bagian *Predicted Stock Price for the Next Period*.
       - Anda juga dapat melihat *Feature Importance* jika model mendukung.
    """)
elif language == "English":
    st.sidebar.markdown("### How to Use the Website:")
    st.sidebar.write("""
    1. **Upload your financial dataset** in the file upload box above.  
       You can access dataset format and examples at [this link](https://drive.google.com/drive/folders/1fJp8NOyLTMmiZQ6gpgcscn2IMf9TohuD?usp=drive_link).
    2. **Dataset Preview**: You can view and check your data in the Dataset Preview section.
    3. **Data Visualization**: You can visualize the data in the Data Visualization section.
    4. **Machine Learning Analysis**:
       - Select the financial ratios to use for analysis from the *Select ratios for analysis* list.
       - Select the target column (company's stock price) from the *Select the target column* dropdown.
       - Click *Run Model and Predict Stock Prices* to run the machine learning model and see the predictions.
    5. **Machine Learning Analysis Results**:
       - The website will display model evaluation metrics such as RMSE, MAE, MAPE, and R².
       - You can view a comparison table between *Actual vs Predicted Stock Prices*.
       - The stock price prediction for the next period will be displayed in the *Predicted Stock Price for the Next Period* section.
       - You can also view *Feature Importance* if the model supports it.
    """)

if uploaded_file:
    # Load data based on file type
    if uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)

    # Standardize column names to uppercase
    data.columns = [col.upper() for col in data.columns]

    # Ensure the dataset has a 'QUARTER' column
    if "QUARTER" not in data.columns:
        st.error("The dataset must have a 'QUARTER' column for proper visualization.")
    else:
        # Treat 'QUARTER' as a categorical variable with row-based ordering
        data['QUARTER'] = pd.Categorical(data['QUARTER'], categories=data['QUARTER'], ordered=True)

        # Add space before dataset preview
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.write("Dataset Preview (Most Recent Data):")
        st.write(data.tail())  # Show the last 5 rows of the dataset

        # Extract available banks and metrics dynamically
        banks = list(set(col.split('_')[1] for col in data.columns if '_' in col))
        available_metrics = list(set(col.split('_')[0] for col in data.columns if '_' in col and not col.startswith('PRICE_')))

        # Visualization Section
        st.subheader("Data Visualization")
        selected_bank = st.selectbox("Select a company for visualization", banks)
        selected_metric = st.selectbox("Select a metric for visualization", available_metrics + ['PRICE'])

        if selected_bank and selected_metric:
            # Build column name dynamically for metric and price
            selected_column_metric = f"{selected_metric.upper()}_{selected_bank.upper()}" if selected_metric != 'PRICE' else f"PRICE_{selected_bank.upper()}"
            
            if selected_column_metric in data.columns:
                visualization_data = data[['QUARTER', selected_column_metric]].set_index('QUARTER')
                st.line_chart(visualization_data)
            else:
                st.warning(f"No data available for metric {selected_metric} for company {selected_bank}.")

        # Analysis Section
        st.subheader("Stock Price Prediction and Analysis")
        selected_ratios = st.multiselect("Select ratios for analysis", available_metrics)
        selected_target = st.selectbox("Select the target column (e.g., stock price)", 
                                        [f"PRICE_{bank}" for bank in banks])

        if selected_ratios and selected_target:
            # Dynamically build feature columns for the selected bank
            selected_bank = selected_target.split("_")[1]  # Extract bank from target column
            feature_columns = [f"{ratio.upper()}_{selected_bank.upper()}" for ratio in selected_ratios if f"{ratio.upper()}_{selected_bank.upper()}" in data.columns]

            if feature_columns:
                X = data[feature_columns]
                y = data[selected_target]

                # Handle missing values
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Model and hyperparameter selection
                model_choice = st.selectbox("Choose a machine learning model", 
                                             ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", 
                                              "Support Vector Regression (SVR)", "K-Nearest Neighbors (KNN)", 
                                              "Linear Regression"])

                # Initialize model with hyperparameter tuning
                if st.button("Run Model and Predict Stock Prices"):
                    start_time = time.time()  # Start timer

                    if model_choice == "Random Forest":
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [None, 10, 20],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4]
                        }
                        model = RandomForestRegressor(random_state=42)
                    elif model_choice == "Gradient Boosting":
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'max_depth': [3, 5, 10]
                        }
                        model = GradientBoostingRegressor(random_state=42)
                    elif model_choice == "XGBoost":
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'max_depth': [3, 5, 10]
                        }
                        model = XGBRegressor(objective='reg:squarederror', random_state=42)
                    elif model_choice == "LightGBM":
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'learning_rate': [0.01, 0.1, 0.2],
                            'num_leaves': [31, 50, 100]
                        }
                        model = LGBMRegressor(random_state=42)
                    elif model_choice == "Support Vector Regression (SVR)":
                        param_grid = {
                            'C': [0.1, 1.0, 10],
                            'kernel': ['linear', 'rbf']
                        }
                        model = Pipeline([
                            ('scaler', StandardScaler()),
                            ('svr', SVR())
                        ])
                    elif model_choice == "K-Nearest Neighbors (KNN)":
                        param_grid = {
                            'n_neighbors': [3, 5, 10],
                            'weights': ['uniform', 'distance']
                        }
                        model = Pipeline([
                            ('scaler', StandardScaler()),
                            ('knn', KNeighborsRegressor())
                        ])
                    elif model_choice == "Linear Regression":
                        st.write("Linear Regression does not require hyperparameter tuning.")
                        model = Pipeline([
                            ('scaler', StandardScaler()),
                            ('lr', LinearRegression())
                        ])
                        param_grid = {}

                    # Hyperparameter tuning
                    if param_grid:
                        st.write("Performing Hyperparameter Tuning...")
                        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                        st.write(f"Best Hyperparameters: {grid_search.best_params_}")

                    # Train and evaluate the model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Predict next period
                    latest_features = X.tail(1).values  # Use the latest feature values for prediction
                    next_period_prediction = model.predict(latest_features)[0]

                    # Calculate execution time
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    # Display evaluation metrics in a styled way
                    st.markdown("### Model Evaluation Metrics")
                    st.write(f"**Model Used:** {model_choice}")
                    st.write(f"**Time Taken:** {elapsed_time:.2f} seconds")
                    st.write(f"**Company Analyzed:** {selected_bank}")
                    st.write(f"**Ratios Used for Analysis:** {', '.join(selected_ratios)}")
                    st.write(f"**Root Mean Squared Error (RMSE):** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
                    st.write(f"**Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred):.2f}")
                    st.write(f"**Mean Absolute Percentage Error (MAPE):** {mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}%")
                    st.write(f"**R-squared (R²):** {r2_score(y_test, y_pred):.2f}")

                    # Display feature importance below evaluation metrics (if applicable)
                    if hasattr(model, "feature_importances_"):
                        st.markdown("### Feature Importance")
                        feature_importances = pd.DataFrame({
                            "Feature": X.columns,
                            "Importance": model.feature_importances_
                        }).sort_values(by="Importance", ascending=False)
                        st.dataframe(feature_importances)

                    # Display predictions vs actual as a table
                    st.markdown("### Actual vs Predicted Stock Prices")
                    predictions = pd.DataFrame({
                        "Actual": y_test.reset_index(drop=True),
                        "Predicted": y_pred
                    })
                    st.dataframe(predictions)

                    # Display next period prediction
                    st.markdown("### Predicted Stock Price for the Next Period")
                    st.write(f"**{next_period_prediction:.2f}**")
