import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("yt_ad_revenue_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

# 1. get user input
views = st.number_input("Views", min_value=0.0, step=100.0)
likes = st.number_input("Likes", min_value=0.0, step=10.0)
video_length_minutes = st.number_input("Video length (minutes)", min_value=0.0, step=1.0)
subscribers = st.number_input("Subscribers", min_value=0.0, step=100.0)
comments = st.number_input("Comments", min_value=0.0, step=10.0)

year = st.number_input("Year", min_value=2005, max_value=2030, step=1)
month = st.number_input("Month", min_value=1, max_value=12, step=1)
day = st.number_input("Day", min_value=1, max_value=31, step=1)

video_id = st.number_input("Video ID (numeric)", min_value=0, step=1)

category = st.selectbox("Category", ["Education", "Gaming", "Music"])
country = st.selectbox("Country", ["US", "IN", "UK"])
device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet"])

if st.button("Predict revenue"):
    # 2. create a row matching your original df BEFORE dummies/scaling
    row = {
        "views": views,
        "likes": likes,
        "video_length_minutes": video_length_minutes,
        "subscribers": subscribers,
        "comments": comments,
        "year": year,
        "month": month,
        "day": day,
        "video_id": video_id,
        "category": category,
        "country": country,
        "device": device,
    }
    input_df = pd.DataFrame([row])

    # 3. apply same preprocessing as training:
    #    - StandardScaler on watch_time_minutes
    #    - MinMaxScaler on views, likes, video_length_minutes, subscribers, comments
    #    - get_dummies on category, country, device
    #    Ideally you saved your fitted scalers and load them here, then use .transform()

    input_df = pd.get_dummies(input_df,
                              columns=['category', 'country', 'device'],
                              dtype='int')

    # 4. align columns with training feature_cols
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0  # add missing dummy/feature as 0

    # extra columns created by get_dummies in app but not in training should be removed:
    input_df = input_df[feature_cols]

    # 5. predict
    pred = model.predict(input_df)[0]
    st.success(f"Predicted ad revenue: ${pred:.2f} USD")