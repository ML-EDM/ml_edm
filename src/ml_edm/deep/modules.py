import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 n_layers=1, 
                 dropout=0.2, 
                 return_all_states=False,
                 *args, **kwargs):
        
        super(LSTM, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_all_states = return_all_states

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout, *args, **kwargs)
    
    def forward(self, x):
        if self.return_all_states:
            return self.lstm(x)[0] # return all hidden states 
        else:
            return self.lstm(x)[0][:,-1,:]

class LSTM_FCN(nn.Module):

    def __init__(self, 
                 input_dim,
                 seq_length, 
                 output_dim, 
                 hidden_dim=64, 
                 dropout=0.8,
                 filter_sizes=[128, 256, 128],
                 kernel_sizes=[7, 5, 3],
                 temperature=1.
                 ):
        
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.temperature = temperature

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(self.input_dim, self.filter_sizes[0], self.kernel_sizes[0], padding="same")
        self.conv2 = nn.Conv1d(self.filter_sizes[0], self.filter_sizes[1], self.kernel_sizes[1], padding="same")
        self.conv3 = nn.Conv1d(self.filter_sizes[1], self.filter_sizes[2], self.kernel_sizes[2], padding="same")

        self.bn1 = nn.BatchNorm1d(self.filter_sizes[0])
        self.bn2 = nn.BatchNorm1d(self.filter_sizes[1])
        self.bn3 = nn.BatchNorm1d(self.filter_sizes[2])

        self.out = ClassificationHead(self.hidden_dim + self.filter_sizes[-1], 
                                      self.output_dim, self.temperature)

    def forward(self, x):

        x_time = self.dropout(self.lstm(x)[0][:,-1,:])

        x = torch.permute(x, (0, 2, 1))
        x_conv = F.relu(self.conv1(x))
        x_conv = F.relu(self.conv2(x_conv))
        x_conv = F.relu(self.conv3(x_conv))
        x_conv = torch.mean(x_conv, dim=-1)

        x = torch.cat((x_time, x_conv), dim=1)
        out = self.out(x)

        return out


class ClassificationHead(nn.Module):

    def __init__(self, hidden_dim, n_classes, temperature=1.):
        super(ClassificationHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.linear = nn.Linear(self.hidden_dim, self.n_classes, bias=True)
        self.temperature = temperature

    def forward(self, x):
        return F.softmax(self.linear(x) / self.temperature, dim=-1)