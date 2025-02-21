import os
import sys
from datetime import datetime
sys.path.append(os.path.abspath("../Transformers/"))
import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset 
from sklearn.preprocessing import StandardScaler
from models.transformer import TransformerEncoderOnly as Tfrmer


def read_data(file_path):
    df=pd.read_csv(file_path)
    df["Date"]=pd.to_datetime(df["Date"])
    df.dropna(axis=1,inplace=True)
    return df

def split_data(df,train_ratio):
    train_df=df.iloc[:int(train_ratio*df.shape[0]),:]
    test_df=df.iloc[int(train_ratio*df.shape[0]):,:]
    return train_df,test_df

def scale_data(df):
    scaled_data=pd.DataFrame()
    scaler=StandardScaler()
    for col in df.columns:
        if df[col].dtype=='float' or df[col].dtype=="int":
           scaled_data[col]=scaler.fit_transform(df[[col]]).flatten()
    return scaled_data

def create_data(data,target,historical_lookup,forecast_horizon):
    X=[]
    y=[]
    if not isinstance(data,pd.DataFrame):
       data=pd.DataFrame(data)   
    features=[name for name in data.columns if name!=target]   
    for i in range(len(data)-historical_lookup-forecast_horizon):
        X.append(data.iloc[i:i+historical_lookup][features].values)
        y.append(data.iloc[i+historical_lookup:i+historical_lookup+forecast_horizon][target].values)
    return torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.float32) 

file_path="../Transformers/Data/federalbank.csv"
df=read_data(file_path=file_path)

train_df,test_df=split_data(df,0.8)
scaled_train_df=scale_data(train_df)
scaled_test_df=scale_data(test_df)


input,output=create_data(scaled_train_df,target="Open Price",historical_lookup=14,forecast_horizon=1)
print(f"input shape is {input.size()} and output shape is {output.size()}")

train_dataset=TensorDataset(input,output)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=False)

input_size=input.shape[2]
output_size=output.shape[1]

model=Tfrmer(input_size=input_size,output_size=output_size,d_model=128,nhead=8,num_enc_layers=2,
                              norm_flag="instance",dropout=0.1,lstm_hidden_size=50,lstm_layers=2)
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)

for epoch in range(10):
    for X,y in train_loader:
        optimizer.zero_grad()
        pred=model(X)
        loss=criterion(pred,y)
        loss.backward()
        optimizer.step()
    if epoch%1==0:
        print(f"Epoch {epoch+1} Loss: {np.round(loss.item(),4)}")     

def save_model(path):
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(path, exist_ok=True)  # Create directory if it doesn't exist
    model_path = os.path.join(path, f"encoder_transformer_{formatted_timestamp}.pth")
    torch.save(model.state_dict(), model_path)

save_model("../Transformers/Trained_Models")
