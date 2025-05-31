import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchaudio
import numpy as np

class Attention(nn.Module):
    def __init__(self, lstm_hidden_dim):
        super(Attention, self).__init__()
        self.attention_layer = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, lstm_out):
        attention_scores = self.attention_layer(lstm_out)
        attention_weights = F.softmax(attention_scores, dim=1)  # Normalize over the sequence length
        weighted_sum = torch.sum(attention_weights * lstm_out, dim=1)
        return weighted_sum, attention_weights

class SpatialAttention(nn.Module):
    def __init__(self, input_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 1, kernel_size=1)

    def forward(self, x):
        attention_scores = self.conv1(x)
        attention_weights = torch.sigmoid(attention_scores)  # Normalize scores
        return x * attention_weights

class Hypernetwork(nn.Module):
    def __init__(self, input_size, output_size=256*128 + 128, hidden_size=128, dropout_rate=0.2):
        super(Hypernetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.in_features = 256
        self.out_features = 128
        self.bias_size = 128

    def forward(self, metadata):
        x = F.relu(self.fc1(metadata))
        x = self.dropout(x)
        output = self.fc2(x)
        
        weights = output[:, :self.in_features * self.out_features]
        biases = output[:, self.in_features * self.out_features:]
        
        weights = weights.view(-1, self.in_features, self.out_features)
        return weights, biases


class PCGClassifier(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(PCGClassifier, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=(2,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(dropout_rate)
        )
        
        self.spatial_attention_1 = SpatialAttention(32)
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1,5), padding=(0,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(dropout_rate)
        )
        
        self.spatial_attention_2 = SpatialAttention(64)
        
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5,1), padding=(2,0)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(dropout_rate)
        )
        
        self.spatial_attention_3 = SpatialAttention(128)
        
        self.lstm = nn.LSTM(
            input_size=3072,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.attention = Attention(lstm_hidden_dim=256)
        
        self.hypernetwork = Hypernetwork(
            input_size=6,
            output_size=256*128 + 128,
            hidden_size=128,
            dropout_rate=dropout_rate
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, metadata):
        x = self.conv_block_1(x)
        x = self.spatial_attention_1(x)
        x = self.conv_block_2(x)
        x = self.spatial_attention_2(x)
        x = self.conv_block_3(x)
        x = self.spatial_attention_3(x)
        
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, time, channels * freq)
        
        lstm_out, _ = self.lstm(x)
        lstm_out, attention_weights = self.attention(lstm_out)
        
        dynamic_weights, dynamic_bias = self.hypernetwork(metadata)
        
        lstm_out = lstm_out.unsqueeze(1)
        intermediate_output = torch.bmm(lstm_out, dynamic_weights)
        intermediate_output = intermediate_output.squeeze(1)
        
        intermediate_output = intermediate_output + dynamic_bias
        
        intermediate_output = F.relu(intermediate_output)
        
        out = self.classifier(intermediate_output)
        return out.view(-1)
