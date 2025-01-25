import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from LSTM import LSTM, BiLSTM

# Import data
data = pd.read_csv("combined3_102_sorted.csv")
scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data[['delay', 'scheduled_time', 'day', 'day_of_year']])

# Z score normalization
data = data[['delay', 'scheduled_time', 'day', 'day_of_year']].values
mean = np.mean(data)
std_dev = np.std(data)
scaled_data = (data-mean)/std_dev
# # create train and test sets
# sizes = int(len(scaled_data) * 0.3)
# train_data = scaled_data[:sizes]
# test_data = scaled_data[sizes:]

batchs = DataLoader(dataset=scaled_data, batch_size=600, shuffle=False)   
# Sequence creator
def createSequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x_data = data[i:(seq_length+i)]
        y_data = data[seq_length+i][0]
        if len(x_data) == 30:
            x.append(x_data)
            y.append(y_data)
    return torch.stack(x, dim=0), torch.stack(y, dim=0)

# Model
# model = LSTM(inputdim=4, outputdim=1, layerdim=1, dropout=0.2)  # NON Bidirectional
model = BiLSTM(inputdim=4, outputdim=1, layerdim=1, dropout=0.2)  # Bi Directional
loss_fcn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

h1, c1, h2, c2 = None, None, None, None

for batch in batchs:
    # Create sequences
    X_train, y_train = createSequences(batch, 30)
    y_train = y_train.reshape(-1,1)
    X_train = X_train.float()
    # print(X_train.shape, y_train.shape)

    # Train
    epochs = 1000
    h1, c1, h2, c2 = None, None, None, None
    for epoch in range(epochs):

        pred, h1, c1, h2, c2= model(X_train, h1, c1, h2, c2)
        # print(pred.shape)

        loss = loss_fcn(pred, y_train)
        if epoch%100 == 0:
            print(epoch, loss.item())

        optimizer.zero_grad()
        loss.backward(retain_graph = True)  
        optimizer.step()

        h1 = h1.detach()
        c1 = c1.detach()
        h2 = h2.detach()
        c2 = c2.detach()
