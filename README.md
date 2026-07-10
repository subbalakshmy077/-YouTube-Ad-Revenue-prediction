# YouTube Ad Revenue Prediction (Streamlit + ML)

This project predicts **YouTube ad revenue (USD)** based on video performance metrics such as views, likes, comments, video length, subscribers, and other metadata.  
It includes a full machine learning pipeline in Python and an interactive Streamlit web app where users can input video details and get an estimated revenue.

---

## 🚀 Project Overview

- Task: Regression – predict `ad_revenue_usd` from video metrics.
- Model: `DecisionTreeRegressor` from scikit‑learn.
- Dataset: `youtube_ad_revenue_dataset.csv` (watch time, views, likes, comments, subscribers, category, country, device, date, video_id, etc.).
- App: Streamlit web app (`app2.py`) for real‑time revenue prediction.

This is a good portfolio project for data science / ML engineer roles because it shows:
- End‑to‑end data preprocessing and feature engineering.
- Model training & evaluation.
- Deployment as a simple web app using Streamlit.

---

## 🧠 Modeling & Preprocessing

Key steps in the training script (`train_model.py`):

1. **Load data**

   ```python
   df = pd.read_csv("youtube_ad_revenue_dataset.csv")
   df.drop_duplicates(inplace=True)
   df.dropna(inplace=True)
   ```

2. **Encode categorical variables**

   ```python
   df = pd.get_dummies(df, columns=["category"], dtype="int")
   df = pd.get_dummies(df, columns=["country"], dtype="int")
   df = pd.get_dummies(df, columns=["device"], dtype="int")
   ```

3. **Feature engineering (date, video_id)**

   ```python
   df["date"] = pd.to_datetime(df["date"])
   df["year"] = df["date"].dt.year
   df["month"] = df["date"].dt.month
   df["day"] = df["date"].dt.day

   df.drop(columns=["date", "minute", "second", "hour"], errors="ignore", inplace=True)

   df["video_id"] = df["video_id"].str.replace("vid_", "", regex=False).astype(int)
   ```

4. **Scaling**

   - StandardScaler on `watch_time_minutes`.
   - MinMaxScaler on `["views", "likes", "video_length_minutes", "subscribers", "comments"]`.

5. **Model training**

   ```python
   X = df.drop(["ad_revenue_usd"], axis=1)
   y = df["ad_revenue_usd"].astype(int)

   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeRegressor

   x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=40)

   model = DecisionTreeRegressor()
   model.fit(x_train, y_train)
   ```

6. **Saving artifacts**

   ```python
   import joblib

   joblib.dump(model, "yt_ad_revenue_model.pkl")
   joblib.dump(X.columns.tolist(), "feature_cols.pkl")
   ```

---

## 🌐 Streamlit App (Prediction UI)

The Streamlit app (`app2.py`) does:

- Loads `yt_ad_revenue_model.pkl` and `feature_cols.pkl`.
- Collects video inputs: views, likes, comments, subscribers, length, date, video_id, category, country, device.
- Applies the same preprocessing (scaling + `get_dummies` + column alignment).
- Calls `model.predict(input_df)` and returns estimated `ad_revenue_usd` in USD.

Basic usage inside the app:

```python
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("yt_ad_revenue_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

# user inputs via Streamlit widgets
# build input_df, preprocess it, align to feature_cols, then:
pred = model.predict(input_df)
st.success(f"Predicted ad revenue: ${pred:.2f} USD")
```


## ▶️ Run the Streamlit App Locally

From the project folder:

```bash
streamlit run app2.py
```


## 🔍 Possible Improvements

- Try other models (Random Forest, Gradient Boosting, XGBoost) and compare metrics.  
- Add evaluation plots (feature importance, actual vs predicted).  
- Improve UI (tooltips, default values, validation) in Streamlit.  
- Add authentication or simple user logging for a more realistic “creator dashboard” feel.

---


