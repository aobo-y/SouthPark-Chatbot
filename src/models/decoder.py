''' Decoder '''

import torch
from torch import nn
from .attn import Attn

class DecoderRNN(nn.Module):
    """
    This means that our embedded word tensor and GRU output will both have shape (1, batch_size, hidden_size).

    Inputs:
        input_seq: one time step (one word) of input sequence batch; shape=(seq_length, batch_size)
        speakers: speaker id, shape=(batch_size)
        init_hidden: initial hidden state; shape=(n_layers x num_directions, batch_size, hidden_size)
        encoder_outputs: encoder model’s output; shape=(max_length, batch_size, hidden_size)

    Outputs:
        output: softmax normalized tensor giving probabilities of each word being the correct next word in the decoded sequence;
                        shape=(seq_length, batch_size, voc.num_words)
        hidden: final hidden state of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)
    """

    def __init__(self, embedding, personas, hidden_size, output_size,
                 n_layers, dropout, rnn_type, attn_type):
        super(DecoderRNN, self).__init__()
        self.attn_type = attn_type

        self.hidden_size = hidden_size
        self.input_size = embedding.embedding_dim + personas.embedding_dim
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.personas = personas

        if rnn_type == 'LSTM':
            self.decoder = nn.LSTM(self.input_size, hidden_size, n_layers,
                                   dropout=(0 if n_layers == 1 else dropout))
        else:
            self.decoder = nn.GRU(self.input_size, hidden_size, n_layers,
                                  dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, self.output_size)
        self.attn = Attn(attn_type, hidden_size)

    def forward(self, input_seq, speakers, init_hidden, encoder_outputs):
        seq_len = input_seq.size(0)

        # Get embedding of current input word
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)

        # Expand persona to shape=(seq_len, batch_size, PERSONA_EMBEDDING_SIZE)
        persona = self.personas(speakers)
        persona = persona.unsqueeze(0).expand(seq_len, -1, -1)

        # Concat embeddings; shape=(seq_len, batch_size, hidden_size+PERSONA_EMBEDDING_SIZE)
        features = torch.cat((embedded, persona), 2)

        # Forward through RNN; rnn_output shape = (seq_len, batch_size, hidden_size)
        rnn_output, hidden = self.decoder(features, init_hidden)

        # Calculate context vector from the current RNN output; shape = (seq_len, batch_size, max_length)
        context = self.attn(rnn_output, encoder_outputs)

        # Concatenate weighted context vector and RNN output using Luong eq. 5
        concat_input = torch.cat((rnn_output, context), 2)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        output = nn.functional.softmax(output, dim=2)
        return output, hidden
