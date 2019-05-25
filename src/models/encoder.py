''' Encoder '''

import torch
from torch import nn

class EncoderRNN(nn.Module):
    """
    The outputs of each network are summed at each time step.

    Inputs:
        input_seq: batch of input sentences; shape=(max_length, batch_size)
        input_lengths: list of sentence lengths corresponding to each sentence in the batch; shape=(batch_size)
        init_hidden: initial hidden state; shape=(n_layers x num_directions, batch_size, hidden_size)

    Outputs:
        outputs: output features from the last hidden layer of the GRU (sum of bidirectional outputs);
                         shape=(max_length, batch_size, hidden_size)
        hidden: updated hidden state from GRU;
                        shape=(n_layers x num_directions, batch_size, hidden_size)
    """

    def __init__(self, embedding, hidden_size, n_layers, dropout, rnn_type):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers

        input_size = embedding.embedding_dim
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU/LSTM:
        # a) the input_size and hidden_size params are both set to 'hidden_size'
        #        because our input size is a word embedding with number of features == hidden_size
        # b) use bidirectional GRU/LSTM to capture context words
        if rnn_type == 'LSTM':
            self.encoder = nn.LSTM(input_size, hidden_size, n_layers,
                                   dropout=(0 if n_layers == 1 else dropout),
                                   bidirectional=True)
        else:
            self.encoder = nn.GRU(input_size, hidden_size, n_layers,
                              dropout=(0 if n_layers == 1 else dropout),
                              bidirectional=True)

    def forward(self, input_seq, input_length, init_hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)

        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)

        # Forward pass through GRU
        # outputs shape = (max_length, batch_size, hidden_size*num_directions)
        # hidden shape = (n_layers*num_directions, batch_size, hidden_size)
        outputs, hidden = self.encoder(packed, init_hidden)

        # Unpack padding; shape = (max_length, batch_size, hidden_size*num_directions)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        # shape = (max_length, batch_size, hidden_size)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden
