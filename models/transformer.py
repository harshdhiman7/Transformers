import math
import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PostionalEncoding(nn.Module):
      def __init__(self, d_model, max_len=36):
          super(PostionalEncoding,self).__init__()
          self.d_model=d_model
          self.max_len=max_len

          #Precompute positional encoding for max sequence length
          pe=torch.zeros(max_len,d_model)
          position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1) #Shape: (max_len,1)
          div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

          pe[:,0::2]=torch.sin(position*div_term)
          pe[:,1::2]=torch.cos(position*div_term)

          self.pe=pe.unsqueeze(0) # Add batch dimension, (1,max_len,d_model)

      def forward(self,x):   
          batch_size,seq_len,_=x.size()

          #if input sequence length exceeds max_len, extend positional encodings
          if seq_len> self.max_len:
             pe=self._get_extended_positional_encodings(seq_len)
          else:
             pe=self.pe[:,seq_len,:]      

          pe=pe.expand(batch_size,-1,-1)

          return x+pe 

      def _get_extended_positional_encodings(self,seq_len):
          position=torch.arange(self.max_len,seq_len,dtype=torch.float).unsqueeze(1) #Shape: (max_len,1)
          div_term=torch.exp(torch.arange(0,self.d_model,2).float()*(-math.log(10000.0)/self.d_model))
          
          pe_extended=torch.zeros(seq_len-self.max_len,self.d_model)
          pe_extended[:,0::2]=torch.sin(position*div_term)
          pe_extended[:,1::2]=torch.cos(position*div_term)

          pe_combined=torch.cat((self.pe.squeeze(0),pe_extended),dim=0).unsqueeze(0) # Shape (1,seq_len,d_model)

          return pe_combined

class TransformerEncoderOnly(nn.Module):
      def __init__(self,input_size,output_size,d_model,nhead,num_enc_layers,norm_flag,
                    dropout=0.1,lstm_hidden_size=None,lstm_layers=None):
          super(TransformerEncoderOnly,self).__init__()        
          self.pos_encoder=PostionalEncoding(d_model)
          encoder_layers=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward=d_model*4,
                                                    dropout=dropout,batch_first=True)
          self.transformer_encoder=nn.TransformerEncoder(encoder_layers,num_layers=num_enc_layers)
          self.input_embedding=nn.Linear(input_size,d_model) #Project input dimension to d_model dimension
          self.d_model=d_model
          self.lstm=nn.LSTM(input_size=d_model,hidden_size=lstm_hidden_size,num_layers=lstm_layers)
          self.relu=nn.ReLU()
          self.LeakyReLU=nn.LeakyReLU(0.1)
          self.norm_flag=norm_flag
          self.batch_norm=nn.BatchNorm1d(num_features=lstm_hidden_size)
          self.instance_norm=nn.InstanceNorm1d(num_features=lstm_hidden_size)
          self.decoder=nn.Linear(lstm_hidden_size,output_size)

      def forward(self,x):
          x_embedded=self.input_embedding(x)*math.sqrt(self.d_model)
          x=self.pos_encoder(x_embedded)
          encoder_output=self.transformer_encoder(x)
          if encoder_output.dim()==2:
             encoder_output=encoder_output.unsqueeze(1)
          lstm_out,(hn,cn)=self.lstm(encoder_output)
          batch_size,seq_len,hidden_size=lstm_out.size()
          if self.norm_flag=="batch":
             lstm_out=lstm_out.view(-1,hidden_size)                 
             lstm_out=self.batch_norm(lstm_out)
             lstm_out=lstm_out.view(batch_size,seq_len,hidden_size)
          elif self.norm_flag=="instance":
               lstm_out=lstm_out.view(-1,hidden_size)                 
               lstm_out=self.instance_norm(lstm_out)
               lstm_out=lstm_out.view(batch_size,seq_len,hidden_size)
          lstm_out=lstm_out[:,-1,:]
          output=self.decoder(lstm_out)

          return output

    