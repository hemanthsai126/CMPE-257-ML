import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
# Define StockLSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        output = self.fc(last_hidden_state)
        return output

# Model Hyperparameters
input_size = 3  # Number of features (High, Low, Close)
hidden_size = 64
num_layers = 2
output_size = 3  # Predict High, Low, and Avg Close
sequence_length = 10  # Sequence length for training
