"""
Implementation of the “Attention Layer” proposed by Luong et al. in 2015
https://arxiv.org/abs/1508.04025
"""

import torch
from torch import nn

class Attn(torch.nn.Module):
    """
    Inputs:
        decoder_outputs: sequences of batch outputs(hiddens); shape=(output_len, batch_size, hidden_size)
        encoder_outputs: sequences of betch outputs(hiddens); shape=(input_len, batch_size, hidden_size)

    Outputs:
        context: context vector computed as the weighted average of all the encoder outputs; shape=(output_len, batch_size, hidden_size)
    """

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.method = method

        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, 'is not an appropriate attention method.')
        if self.method == 'general':
            self.attn = torch.nn.Linear(hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def score(self, decoder_outputs, encoder_outputs):
        # transpose the dims seq_length and batch
        decoder_outputs = decoder_outputs.transpose(0 ,1)
        encoder_outputs = encoder_outputs.transpose(0 ,1)

        if self.method == 'dot':
            # (batch, output_len, hidden_size) * (batch, hidden_size, input_len) = (batch, output_len, input_len)
            return decoder_outputs.bmm(encoder_outputs.transpose(1, 2))

        elif self.method == 'general':
            energy = self.attn(encoder_outputs)
	        # (batch, output_len, hidden_size) * (batch, hidden_size, input_len) = (batch, output_len, input_len)
            return decoder_outputs.bmm(encoder_outputs.transpose(1, 2))

        elif self.method == 'concat':
            output_len = decoder_outputs.size(1)
            input_len = encoder_outputs.size(1)

            # expand to (batch, output_len, input_len, hidden_size)
            decoder_outputs_exp = decoder_outputs.unsqueeze(2).expand(-1, -1, input_len, -1)
            encoder_outputs_exp = encoder_outputs.unsqueeze(1).expand(-1, output_len, -1, -1)

            combined = torch.cat((decoder_outputs_exp, encoder_outputs_exp), 3)
            energy = self.attn(combined).tanh()

            # (batch, output_len, input_len)
            return torch.sum(self.v * energy, dim=3)

    def forward(self, decoder_outputs, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        attn_energies = self.score(decoder_outputs, encoder_outputs)

        # Return the aligned vector, the softmax normalized probability scores; shape=(batch, output_len, input_len)
        attn_weights = nn.functional.softmax(attn_energies, dim=2)

        # Multiply attention weights and encoder outputs to get new "weighted sum" context vector
        # (batch, output_len, input_len) * (batch, input_len, hidden_size) = (batch, output_len, hidden_size)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # change back the batch_size & output_len
        return context.transpose(0, 1)
