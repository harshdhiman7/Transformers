import os
import sys
sys.path.append(os.path.abspath("../Transformers/"))
import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import imageio
import streamlit as st
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.transformer import TransformerEncoderOnly as Tfrmer



# Streamlit UI
st.title("📈 Transformer-based Time Series Forecasting")
st.sidebar.header("Upload CSV File")

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df.dropna(axis=1, inplace=True)
    
    # Train-Test Split
    train_ratio = 0.8
    train_df = df.iloc[:int(train_ratio * df.shape[0]), :]
    test_df = df.iloc[int(train_ratio * df.shape[0]):, :]

    # Scale Data
    def scale_data(df):
        df = df.copy()
        scaler = StandardScaler()
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):  
                df[col] = scaler.fit_transform(df[[col]]).flatten()
            elif np.issubdtype(df[col].dtype, np.datetime64):  
                df[col] = df[col].map(lambda x: x.toordinal())  
        return df

    scaled_train_df = scale_data(train_df)
    scaled_test_df = scale_data(test_df)

    # Data Preparation
    def create_data(data, target, historical_lookup, forecast_horizon):
        X, y = [], []
        del data["Date"]
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)   
        features = [name for name in data.columns if name != target]   
        for i in range(len(data) - historical_lookup - forecast_horizon):
            X.append(data.iloc[i:i + historical_lookup][features].values)
            y.append(data.iloc[i + historical_lookup:i + historical_lookup + forecast_horizon][target].values)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32) 

    # Select Target Column
    target_column = st.sidebar.selectbox("Select Target Column", df.columns,index=None)
    historical_lookup = st.sidebar.slider("Historical Lookup (Timesteps)", min_value=5, max_value=30, value=14)
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Timesteps)", min_value=1, max_value=5, value=1)

    # Create training data
    input_data, output_data = create_data(scaled_train_df, target=target_column, 
                                          historical_lookup=historical_lookup, forecast_horizon=forecast_horizon)
    st.write(f"📊 Input shape: {input_data.size()}, Output shape: {output_data.size()}")

    # Dataloader
    train_dataset = TensorDataset(input_data, output_data)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # Model Setup
    input_size = input_data.shape[2]
    output_size = output_data.shape[1]
    
    model = Tfrmer(input_size=input_size, output_size=output_size, d_model=128, nhead=8, 
                   num_enc_layers=2, norm_flag="instance", dropout=0.1, 
                   lstm_hidden_size=50, lstm_layers=2)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Training & Loss Animation
    losses = []
    frames = []

    st.write("🚀 Training in Progress...")
    progress_bar = st.progress(0)

    for epoch in tqdm(range(150), desc="Training Progress"):
        epoch_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        
        # Update progress bar in Streamlit
        progress_bar.progress((epoch + 1) / 150)

        # Generate frames for animation
        plt.figure(figsize=(6, 4))
        plt.plot(losses, label="Loss Curve", color="blue")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Animation")
        plt.legend()
        plt.savefig(f"frame_{epoch}.png")
        plt.close()
        frames.append(imageio.imread(f"frame_{epoch}.png"))

    # Create GIF
    gif_path = "training_loss.gif"
    imageio.mimsave(gif_path, frames, duration=0.2)

    # Clean up frames
    for epoch in range(150):
        os.remove(f"frame_{epoch}.png")

    st.image(gif_path, caption="📉 Loss Curve Animation", use_column_width=True)
    st.success("✅ Training Complete! Model is ready for predictions.")
