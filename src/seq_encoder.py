"""
Encoder
Our encoder is a multi-layered Gated Recurrent Unit, invented by Cho et al. in 2014.
"""

import torch
from torch import nn

class EncoderRNN(nn.Module):
    """
    The outputs of each network are summed at each time step.

    Inputs:
        input_seq: batch of input sentences; shape=(max_length, batch_size)
        input_lengths: list of sentence lengths corresponding to each sentence in the batch;
                                     shape=(batch_size)
        hidden: hidden state; shape=(n_layers x num_directions, batch_size, hidden_size)

    Outputs:
        outputs: output features from the last hidden layer of the GRU (sum of bidirectional outputs);
                         shape=(max_length, batch_size, hidden_size)
        hidden: updated hidden state from GRU;
                        shape=(n_layers x num_directions, batch_size, hidden_size)
    """

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0.5):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU:
        # a) the input_size and hidden_size params are both set to 'hidden_size'
        #        because our input size is a word embedding with number of features == hidden_size
        # b) use bidirectional GRU to capture context words
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)

    def forward(self, input_seq, input_length, hidden=None):
        # Convert word indexes to embeddings
        # shape = (max_length, batch_size, hidden_size)
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        # shape = (max_length, batch_size, hidden_size)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)
        # Forward pass through GRU
        # outputs shape = (max_length, batch_size, hidden_size*num_directions)
        # hidden shape = (n_layers*num_directions, batch_size, hidden_size)
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        # shape = (max_length, batch_size, hidden_size*num_directions)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        # shape = (max_length, batch_size, hidden_size)
        outputs = outputs[:, :, :self.hidden_size]+outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden