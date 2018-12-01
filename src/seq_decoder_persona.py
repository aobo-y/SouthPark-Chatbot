"""
Decoder
For the decoder, we manually feed our batch one time step at a time.
"""

import torch
from torch import nn
from attn import Attn

class DecoderRNN(nn.Module):
    """
    This means that our embedded word tensor and GRU output will both have shape (1, batch_size, hidden_size).

    Inputs:
        input_step: one time step (one word) of input sequence batch; shape=(1, batch_size)
        speakers: speaker id, shape = (1, batch_size)
        last_hidden: final hidden layer of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)
        encoder_outputs: encoder model’s output; shape=(max_length, batch_size, hidden_size)

    Outputs:
        output: softmax normalized tensor giving probabilities of each word being the correct next word in the decoded sequence;
                        shape=(batch_size, voc.num_words)
        hidden: final hidden state of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)
    """

    def __init__(self, attn_model, embedding, personas, output_size,
                 n_layers=1, dropout=0.5, use_persona=True, rnn_type='GRU'):
        super(DecoderRNN, self).__init__()
        self.attn_model = attn_model

        hidden_size = embedding.embedding_dim
        self.hidden_size = hidden_size
        self.input_size = hidden_size + personas.embedding_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.personas = personas
        self.personas.weight.requires_grad = use_persona
        if rnn_type == 'LSTM':
            self.decoder = nn.LSTM(self.input_size, hidden_size, n_layers,
                                   dropout=(0 if n_layers == 1 else dropout))
        else:
            self.decoder = nn.GRU(self.input_size, hidden_size, n_layers,
                                  dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, self.output_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, speaker, last_hidden, encoder_outputs):
        # Note: we run this one step(word) at a time

        # Get embedding of current input word
        # shape = (1, batch_size, hidden_size)
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # shape = (1, batch_size, persona_size)
        persona = self.personas(speaker)
        # shape = (1, batch_size, hidden_size+persona_size)
        features = torch.cat((embedded, persona), 2)

        # Forward through GRU
        # rnn_output shape = (1, batch_size, hidden_size)
        # hidden shape = (n_layers*num_directions, batch_size, hidden_size)
        rnn_output, hidden = self.decoder(features, last_hidden)
        # Calculate attention weights from the current GRU output
        # attn_weights shape = (batch_size, 1, max_length)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        # shape = (batch_size, hidden_size, 1)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        # shape = (batch_size, hidden_size)
        rnn_output = rnn_output.squeeze(0)
        # shape = (batch_size, hidden_size)
        context = context.squeeze(1)
        # shape = (batch_size, hidden_size*2)
        concat_input = torch.cat((rnn_output, context), 1)
        # shape = (batch_size, hidden_size)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        # shape = (batch_size, voc_dict.length)
        output = self.out(concat_output)
        output = nn.functional.softmax(output, dim=1)
        return output, hidden
