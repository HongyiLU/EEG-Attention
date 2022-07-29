import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout, batch_first=True)
    

    def forward(self, input):
        h_0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        return output


class AttnDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size , num_layers, sequence_length, dropout):
        super(AttnDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.dropout = dropout
        # Soft attention ------------------------------------
        self.softattn = SoftAttention(sequence_length, input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, encoded_input):
        h_0 = torch.zeros(self.num_layers, encoded_input.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, encoded_input.size(0), self.hidden_size)
        # encoded_input.size => [32,128,128] h_0.size => [4, 128, 128]
        # print('h_0.size()' , h_0.size())
        # print('encoded_input.size()' , encoded_input.size())

        encoded_out = encoded_input.reshape(encoded_input.size(0), -1)  # => [batch_size, seq_len*hidden_size]
        # print('encoded_out.size()' , encoded_out.size())
        attn_input = torch.cat((encoded_out, h_0[0]), 1) # => [batch_size, seq_len*hidden_size+input_size]
        # print('attn_input.size()' , attn_input.size())
        attn_weights = self.softattn(attn_input) # => [batch_size, seq_len]
        # print('attn_weights.size()' , attn_weights.size())
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoded_input) # => [batch_size, 1, seq_len]
        # print('attn_applied.size()' , attn_applied.size())
        input_lstm = torch.cat((attn_applied, encoded_out.unsqueeze(1)), dim=2) # => [batch_size, 1, seq_len*hidden_size+input_size]
        # print('input_lstm.size()' , input_lstm.size())
        input_lstm = input_lstm.reshape(encoded_input.size(0), -1, self.input_size) # => [batch_size, seq_len + 1, hidden_size]
        # print('input_lstm.size()' , input_lstm.size())

        output_lstm, (h_n, c_n) = self.lstm(input_lstm, (h_0, c_0))
        # print('output_lstm.size()', output_lstm.size())
        output = self.fc(output_lstm[:, -1, :]) # => [batch_size, output_size]
        # print('output.size()', output.size())
        return output

class AutoEncoderRNN(nn.Module):
    def __init__(self, input_size, input_size_dec, hidden_size, num_layers,sequence_length,output_size, dropout):
        super(AutoEncoderRNN, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = AttnDecoder(input_size_dec, hidden_size, output_size , num_layers, sequence_length, dropout)

    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)

        return decoded_x


class SoftAttention(nn.Module):
    def __init__(self, sequence_length, input_size_dec, output_size):
        super(SoftAttention, self).__init__()
        self.attn = nn.Linear(sequence_length * input_size_dec + output_size, output_size, bias=False)
        # self.attn = nn.Linear(hidden_size, output_size, bias = False)
        self.v = nn.Linear(output_size, sequence_length, bias=False)

    def forward(self, attn_input):
        energy = self.attn(attn_input)
        # print('energy.size', energy.size())

        attn_weights = self.v(energy)
        # print('attn_weights.size', attn_weights.size())
        # attn_weights, (h_n, c_n) = self.attn(attn_input, (h_0, c_0))

        weights_soft = F.softmax(attn_weights, dim=1)  # [256, 20]
        # print('weights_soft', weights_soft)
        # print('weights_soft.size()', weights_soft.size())
        return weights_soft
