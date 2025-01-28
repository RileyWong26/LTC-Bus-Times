import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, inputdim, outputdim, layerdim, dropout):
        super(LSTM, self).__init__()
        self.layerdim = layerdim
        self.lstm1 = nn.LSTM(inputdim, 108, layerdim, batch_first=True)
        self.batchnorm = nn.BatchNorm1d(108)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(108, 56, layerdim, batch_first=True)
        self.layers = nn.Sequential(
            nn.Linear(56,32),
            nn.ReLU(),
            nn.Linear(32, outputdim)
        )
    
    def forward(self, x, h1=None, c1=None, h2=None, c2=None):
        if h1 is None or c1 is None or h2 is None or c2 is None:
            h1 = torch.zeros(self.layerdim, x.size(0), 108)
            c1 = torch.zeros(self.layerdim, x.size(0), 108)
            h2 = torch.zeros(self.layerdim, x.size(0), 56)
            c2 = torch.zeros(self.layerdim, x.size(0), 56)
        
        out,(h1, c1) = self.lstm1(x, (h1,c1))

        # Batch Normalization
        batch_size, seq_len, hidd_size = out.shape
        out = out.reshape(batch_size * seq_len, hidd_size)
        out = self.batchnorm(out)
        out = out.reshape(batch_size, seq_len, hidd_size)

        # Dropout between layers
        out = self.dropout(out)
        # Second LSTM
        out, (h2, c2) = self.lstm2(out, (h2, c2))
        # Dense layers 
        out = self.layers(out)
        out = out[:, -1, :]
        return out, h1, c1, h2, c2


class BiLSTM(nn.Module):
    def __init__(self, inputdim, outputdim, layerdim, dropout):
        super(BiLSTM, self).__init__()
        self.layerdim = layerdim
        self.embedding = nn.Embedding(num_embeddings=25, embedding_dim=1)

        self.lstm1 = nn.LSTM(inputdim, 108, layerdim, batch_first=True, bidirectional=True)
        self.batchnorm = nn.BatchNorm1d(216)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(108 * 2, 56, layerdim, batch_first=True, bidirectional=True)
        self.layers = nn.Sequential(
            nn.Linear(56 * 2,56),
            nn.ReLU(),
            nn.Linear(56, 32),
            nn.ReLU(),
            nn.Linear(32, outputdim)
        )
    
    def forward(self, x, h1=None, c1=None, h2=None, c2=None):
        if h1 is None or c1 is None or h2 is None or c2 is None:
            h1 = torch.zeros(self.layerdim*2, x.size(0), 108)
            c1 = torch.zeros(self.layerdim*2, x.size(0), 108)
            h2 = torch.zeros(self.layerdim*2, x.size(0), 56)
            c2 = torch.zeros(self.layerdim*2, x.size(0), 56)
        
        emb = x[:, :, 4].to(torch.long)

        embed = self.embedding(emb).to(torch.float32)
        x = x[:, :, :4]
        
        x = torch.cat([embed, x], dim=2)
        out,(h1, c1) = self.lstm1(x, (h1,c1))

        # Batch Normilization
        batch_size, seq_len, hidd_size = out.shape
        out = out.reshape(batch_size * seq_len, hidd_size)
        out = self.batchnorm(out)
        out = out.reshape(batch_size, seq_len, hidd_size)

        # Drop out between layers
        out = self.dropout(out)
        # Second LSTM layer
        out, (h2, c2) = self.lstm2(out, (h2, c2))
        out = self.layers(out)
        out = out[:, -1, :56]
        return out, h1, c1, h2, c2

#  Attention Process
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # self.attention = nn.Linear(hidden_dim, 1, bias=False)
        self.attention = nn.MultiheadAttention(hidden_dim, 1, batch_first=True, )

    def forward(self, encoder_outputs):
        # attn_scores = self.attention(encoder_outputs).squeeze(-1)
        # attn_weights = torch.softmax(attn_scores, dim=1)
        # context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        attn_output, attn_weights = self.attention()
        return

# ANother BiLSTM model with attention attatched this time
class AttentionBiLSTM(nn.Module):
    def __init__(self, inputdim, outputdim, layerdim, dropout):
        super(AttentionBiLSTM, self).__init__()
        self.layerdim = layerdim
        self.lstm1 = nn.LSTM(inputdim, 108, layerdim, batch_first=True, bidirectional=True)
        self.batchnorm = nn.BatchNorm1d(216)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(108 * 2, 56, layerdim, batch_first=True, bidirectional=True)
        self.attention = Attention(216)
        self.layers = nn.Sequential(
            nn.Linear(56 * 2,56),
            nn.ReLU(),
            nn.Linear(56, 32),
            nn.ReLU(),
            nn.Linear(32, outputdim)
        )
    
    def forward(self, x, h1=None, c1=None, h2=None, c2=None):
        if h1 is None or c1 is None or h2 is None or c2 is None:
            h1 = torch.zeros(self.layerdim*2, x.size(0), 108)
            c1 = torch.zeros(self.layerdim*2, x.size(0), 108)
            h2 = torch.zeros(self.layerdim*2, x.size(0), 56)
            c2 = torch.zeros(self.layerdim*2, x.size(0), 56)
        
        # First LSTM
        out,(h1, c1) = self.lstm1(x, (h1,c1))
        # ADD attention for second lstm

        context, attn_weights = self.attention(out)
        context = context.unsqueeze(1)

        # Batch Normilization
        batch_size, seq_len, hidd_size = context.shape
        context = context.reshape(batch_size * seq_len, hidd_size)
        context = self.batchnorm(context)
        context = context.reshape(batch_size, seq_len, hidd_size)

        # Drop out between layers
        context = self.dropout(context)
        # Second LSTM layer
        context, (h2, c2) = self.lstm2(context, (h2, c2))
        context = self.layers(context)
        context = context[:, -1, :56]
        return context, h1, c1, h2, c2
         