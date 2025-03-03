import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from LSTM import LSTM, BiLSTM, AttentionBiLSTM
from sklearn.preprocessing import LabelEncoder

# Import data
data = pd.read_csv("combined3_102_sorted.csv")
data2 = pd.read_csv("102_with_weather.csv")
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data[['delay', 'scheduled_time', 'day', 'day_of_year']])

# Z score normalization
weather = data2["Weather"].values
encoder = LabelEncoder()
unique = encoder.fit_transform(weather).reshape(-1,1)
unique -=1

# unique = [len(data[col].unique()) for col in catergories]
data = data[['delay', 'scheduled_time', 'day', 'day_of_year']].values
mean = np.mean(data)
std_dev = np.std(data)
scaled_data = (data-mean)/std_dev
scaled_data = np.concatenate((scaled_data, unique), axis=1)
torch.manual_seed(100)

# # create train and test sets
sizes = int(len(scaled_data) * 0.2)
test_data = scaled_data[:sizes]
train_data = scaled_data[sizes:]

# print(scaled_data.shape)
# print(train_data.shape)
# print(test_data.shape)

train_batchs = DataLoader(dataset=train_data, batch_size=150, shuffle=False)  
test_batchs = DataLoader(dataset=test_data, batch_size=150, shuffle=False)
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

# print(scaled_data)
# print(scaled_data[:,4])
# Model
# model = LSTM(inputdim=5, outputdim=1, layerdim=1, dropout=0.2)  # NON Bidirectional
# model = BiLSTM(inputdim=5, outputdim=1, layerdim=1, dropout=0.2)  # Bi Directional
model = AttentionBiLSTM(inputdim=5, outputdim=1,numheads=4, layerdim=1, dropout=0.2) # Bi Directional with Attention
loss_fcn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

h1, c1, h2, c2 = None, None, None, None

for batch in train_batchs:
    # Create sequences
    X_train, y_train = createSequences(batch, 30)
    y_train = y_train.reshape(-1,1)
    X_train = X_train.float()
    # print(X_train.shape, y_train.shape)

    # Train
    epochs = 50
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

y_list = []
for batch in test_batchs:
    X_test, y_test= createSequences(batch, 30)
    y_test = y_test.reshape(-1,1)
    X_test = X_test.float()

    y_pred=model(X_test, h1, c1, h2, c2)
    y_list.append(y_pred)

data_verify = pd.DataFrame(y_test.tolist(), columns=["Test"])
data_predicted = pd.DataFrame(y_pred.tolist(),columns=['Predictions'])

final_output = pd.concat([data_verify, data_predicted], axis=1)
final_output['difference'] = final_output['Test'] - final_output['Predictions']
print(final_output.head())