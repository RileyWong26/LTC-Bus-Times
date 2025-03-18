import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# Import data

df = pd.read_csv("Data/CondensedDataFiles/102_traffic.csv")
df = df.sample(n=100000, random_state=42)

data = df[['delay','stop_id','scheduled_time','vehicle_id','temperature','Windspeed','Visibility','Traffic']]
embeds = df[['day_of_year','day','weather','conditions']]
data = data.fillna(0) # Fill with 0s if nan

# Z score normalization
data_scaler = StandardScaler() # Own seperate scaler for the delay, so we can inverse transform the output
scaler = StandardScaler()

data['delay'] = data['delay'].values/60
data['delay'] = data_scaler.fit_transform(data[['delay']])
data[['stop_id','scheduled_time','vehicle_id', 'temperature','Windspeed','Visibility','Traffic']] = scaler.fit_transform(data[['stop_id','scheduled_time','vehicle_id','temperature','Windspeed','Visibility','Traffic']])

data = pd.concat([data, embeds], axis=1)

# Split into train and test
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# Sort so there is some sort of order in the sequences
train_set = train_set.sort_values(by=['day_of_year','scheduled_time'])
test_set = test_set.sort_values(by=['day_of_year','scheduled_time'])

print(train_set)
print(test_set)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
batch_size = 120

# Batches
train_data = torch.tensor(train_set.values, dtype=torch.float32).to(device)
test_data = torch.tensor(test_set.values, dtype=torch.float32).to(device)

train_batchs = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False,pin_memory=False)
test_batchs = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, pin_memory=False)

# Sequence creator
def createSequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x_data = data[i:(seq_length+i)]
        y_data = data[seq_length+i][0]
        if len(x_data) < seq_length:
            for i in range (seq_length - len(x_data)):
                x_data.append(torch.zeros([1, 12]))
        x.append(x_data)
        y.append(y_data)
    return torch.stack(x, dim=0), torch.stack(y, dim=0)

class AttentionBiLSTM(nn.Module):
    def __init__(self, inputdim, hiddendim1, hiddendim2, outputdim, numheads, layerdim, dropout):
        super(AttentionBiLSTM, self).__init__()
        self.layerdim = layerdim
        self.device = device  # Store the device as an instance variable
        
        self.embedding1 = nn.Embedding(num_embeddings=365, embedding_dim=1).to(device)
        self.embedding2 = nn.Embedding(num_embeddings=7, embedding_dim=1).to(device)
        self.embedding3 = nn.Embedding(num_embeddings=25, embedding_dim=1).to(device)
        self.embedding4 = nn.Embedding(num_embeddings=3, embedding_dim=1).to(device)

        self.lstm1 = nn.LSTM(inputdim, hiddendim1, layerdim, batch_first=True, bidirectional=True).to(device)
        self.batchnorm = nn.BatchNorm1d(hiddendim1*2).to(device)
        self.batchnorm2 = nn.BatchNorm1d(hiddendim2*2).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.lstm2 = nn.LSTM(hiddendim1*2, hiddendim2, layerdim, batch_first=True, bidirectional=True).to(device)
        self.attention = nn.MultiheadAttention(embed_dim=120, num_heads=numheads, batch_first=True).to(device)
        self.layers = nn.Sequential(
            nn.Linear(120,60),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.Linear(60,30),
            nn.Dropout(dropout),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(30, outputdim)
        ).to(device)
    
    def forward(self, x, h1=None, c1=None, h2=None, c2=None):
        if h1 is None or c1 is None or h2 is None or c2 is None:
            h1 = torch.zeros(self.layerdim*2, x.size(0), 120).to(self.device)  # Move to the correct device
            c1 = torch.zeros(self.layerdim*2, x.size(0), 120).to(self.device)  # Move to the correct device
            h2 = torch.zeros(self.layerdim*2, x.size(0), 60).to(self.device)   # Move to the correct device
            c2 = torch.zeros(self.layerdim*2, x.size(0), 60).to(self.device)   # Move to the correct device
        
        # Embeddings
        emb1 = x[:, :, 8].to(torch.long)
        emb2 = x[:, :, 9].to(torch.long)
        emb3 = x[:, :, 10].to(torch.long)
        emb4 = x[:, :, 11].to(torch.long)
        
        embed1 = self.embedding1(emb1).to(torch.float32)
        embed2 = self.embedding2(emb2).to(torch.float32)
        embed3 = self.embedding3(emb3).to(torch.float32)
        embed4 = self.embedding4(emb4).to(torch.float32)
        # Continous
        x = x[:, :, :8]
        
        x = torch.cat([x, embed1, embed2, embed3, embed4], dim=2)
        # First LSTM
        out,(h1, c1) = self.lstm1(x, (h1,c1))

        # Drop out between layers
        out = self.dropout(out)

        # Batch Normalization
        # batch_size, seq_len, hidd_size = out.shape
        # out = out.reshape(batch_size * seq_len, hidd_size)
        # out = self.batchnorm(out)
        # out = out.reshape(batch_size, seq_len, hidd_size)

        # Second LSTM layer
        out, (h2, c2) = self.lstm2(out, (h2, c2))

        # Add attention layer
        # out, attn_weights = self.attention(query=out, key=out,value=out)
        
        # last time step output
        out = out[:, -1, :]
        
        out = self.dropout(out)
        out = self.batchnorm2(out)
        # Final dense layers
        # out = self.dropout(out)
        out = self.layers(out)
        return out, h1, c1, h2, c2


# Model
model = AttentionBiLSTM(inputdim=12, hiddendim1=120, hiddendim2=60, outputdim=1, numheads=30, layerdim=1, dropout=0.4).to(device) # Bi Directional with Attention
# loss_fcn = nn.SmoothL1Loss()
loss_fcn = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)


# Training
print(next(model.parameters()).device)
h1, c1, h2, c2 = None, None, None, None
epochs = 15

model.train()
from tqdm import tqdm

bar = tqdm(range(epochs))
for epoch in bar:
    epoch_loss = 0
    batch_count = 0

    for batch in train_batchs:
        batch_count += 1
        optimizer.zero_grad() # Reset your gradient
        # Create sequences
        X_train, y_train = createSequences(batch, 30)
        y_train = y_train.reshape(-1,1)
        X_train = X_train.float()
        
        if len(X_train) < batch_size-30:
            break

        # Move hidden states to the correct device
        if h1 is not None:
            h1, c1, h2, c2 = h1.to(device), c1.to(device), h2.to(device), c2.to(device)

        # Train
        pred, h1, c1, h2, c2 = model(X_train, h1, c1, h2, c2)
        loss = loss_fcn(pred, y_train)
        
        loss.backward(retain_graph=True)  
        epoch_loss += loss.item()

        h1 = h1.detach()
        c1 = c1.detach()
        h2 = h2.detach()
        c2 = c2.detach()
        
        optimizer.step()
    
    bar.set_description(f"Epoch: {epoch}, Loss: {epoch_loss/batch_count:.4f}")

# Evaluation
y_test_list, y_pred_list = [], []
h1, c1, h2, c2 = None, None, None, None
test_loss = 0
test_batch_count = 0

model.eval()

with torch.no_grad():
    for batch in test_batchs:
        test_batch_count += 1
        X_test, y_test= createSequences(batch, 30)

        # Pad with zeros
        if len(X_test) < batch_size-30:
            break
        
        y_test = y_test.reshape(-1,1)
        X_test = X_test.float()

        # print(X_test.shape)

        y_pred, h1, c1, h2, c2 = model(X_test, h1, c1, h2, c2)
       
        test_loss += loss_fcn(y_pred, y_test).item()

        y_pred_list.append(y_pred)
        y_test_list.append(y_test)

y_pred_list = torch.cat(y_pred_list, dim=0).cpu().numpy()
y_test_list = torch.cat(y_test_list, dim=0).cpu().numpy()


y_pred_list = data_scaler.inverse_transform(y_pred_list)
y_test_list = data_scaler.inverse_transform(y_test_list)
# test_loss = inverse_Z_Score(test_loss)

data_verify = pd.DataFrame(y_test_list.tolist(), columns=["Test"])
data_predicted = pd.DataFrame(y_pred_list.tolist(),columns=['Predictions'])

final_output = pd.concat([data_verify, data_predicted], axis=1)
final_output['difference'] = final_output['Test'] - final_output['Predictions']
final_output = final_output.round(3)
final_output.to_csv('Protoype outputs.csv', index=False)
print(final_output)

total_difference = np.sum(np.abs(final_output['difference']))
pred_dev = np.std(final_output['Predictions'])
pred_mean = np.mean(final_output['Predictions'])

print(f"Average Loss: {(test_loss/test_batch_count):.5f}")
print(f"Average Difference: {total_difference/len(final_output['difference']):.5f}")
print(f"Standard deviation for the predictions: {pred_dev:.5f}")
print(f"Prediction mean: {pred_mean:.5f}")
# print(final_output['Predictions'])