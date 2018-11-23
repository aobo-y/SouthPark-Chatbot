"""
Greedy decoding is the decoding method
that we use during training when we are NOT using teacher forcing.
"""

import torch
from torch import nn

import config
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class GreedySearchDecoder(nn.Module):
    """
    Inputs:
        input_seq: an input sequence of shape (input_seq length, 1)
        input_length: a scalar input length tensor
        max_length: a max_length to bound the response sentence length.

    Outputs:
        all_tokens: collections of words tokens
        all_scores: collections of words scores
    """

    def __init__(self, encoder, decoder, speaker_id):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.speaker = speaker_id

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long)*config.SOS_TOKEN
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            # Transform speaker_id from int into tensor with shape=(1, 1)
            speaker_id = torch.LongTensor([self.speaker])
            speaker_id = torch.unsqueeze(speaker_id, 1)
            decoder_output, decoder_hidden = self.decoder(decoder_input, speaker_id, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Returen collections of word tokens and scores
        return all_tokens, all_scores

