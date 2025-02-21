import os
import sys
sys.path.append(os.path.abspath("../Transformers/"))
import streamlit as st
import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # For progress bar
from models.transformer import TransformerEncoderOnly as Tfrmer

# Streamlit UI
st.title("Transformer-based Time Series Forecasting")
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Function to read data
def read_data(file):
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(axis=1, inplace=True)
    return df

# Function to scale data
def scale_data(df):
    scaler = StandardScaler()
    scaled_df = df.copy()
    for col in df.select_dtypes(include=["float", "int"]).columns:
        scaled_df[col] = scaler.fit_transform(df[[col]]).flatten()
    return scaled_df

# Function to create sequences
def create_data(data, target, historical_lookup, forecast_horizon):
    X, y = [], []
    del data["Date"]
    features = [col for col in data.columns if col != target]
    for i in range(len(data) - historical_lookup - forecast_horizon):
        X.append(data.iloc[i:i+historical_lookup][features].values)
        y.append(data.iloc[i+historical_lookup:i+historical_lookup+forecast_horizon][target].values)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

if uploaded_file is not None:
    df = read_data(uploaded_file)
    st.write("### Sample Data:")
    st.dataframe(df.head())

    train_df, test_df = df.iloc[:int(0.8 * len(df))], df.iloc[int(0.8 * len(df)):]  # Train-test split
    scaled_train_df, scaled_test_df = scale_data(train_df), scale_data(test_df)

    target_column = st.sidebar.selectbox("Select Target Column", df.columns,index=None)
    historical_lookup = st.sidebar.slider("Historical Lookback", min_value=5, max_value=30, value=14)
    forecast_horizon = st.sidebar.slider("Forecast Horizon", min_value=1, max_value=10, value=1)

    input_data, output_data = create_data(scaled_train_df, target=target_column, 
                                          historical_lookup=historical_lookup, forecast_horizon=forecast_horizon)
    st.write(f"**Input Shape:** {input_data.shape}, **Output Shape:** {output_data.shape}")

    train_dataset = TensorDataset(input_data, output_data)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    input_size = input_data.shape[2]
    output_size = output_data.shape[1]
    
    model = Tfrmer(input_size=input_size, output_size=output_size, d_model=128, nhead=8, num_enc_layers=2,
                    norm_flag="instance", dropout=0.1, lstm_hidden_size=50, lstm_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    if st.sidebar.button("Train Model"):
        progress_bar = st.progress(0)
        loss_text = st.empty()
        for epoch in range(150):
            epoch_loss = 0
            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/150", leave=False):
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            progress_bar.progress((epoch + 1) / 150)
            loss_text.write(f"**Epoch {epoch+1} Loss:** {avg_loss:.4f}")

        st.success("Training Complete! 🎉")
