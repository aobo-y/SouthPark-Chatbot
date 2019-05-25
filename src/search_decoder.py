"""
Greedy decoding is the decoding method
that we use during training when we are NOT using teacher forcing.
"""

import math
import random
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

    def __init__(self, model):
        super(GreedySearchDecoder, self).__init__()
        self.model = model

    def forward(self, input_seq, input_length, speaker_id, sos, eos):
        target_var = torch.full((1, 1), sos, dtype=torch.long, device=device)

        speaker_var = torch.tensor([speaker_id], dtype=torch.long, device=device)

        output_var = self.model(input_seq, input_length, target_var, speaker_var, config.MAX_LENGTH)

        # squeeze batch dim
        output_var = output_var.squeeze(1)

        # Obtain most likely word token and its softmax score
        scores, tokens = torch.max(output_var, dim=1)

        return tokens, scores

class BeamSearchDecoder(nn.Module):
    """
    Inputs:
        input_seq: an input sequence of shape (input_seq length, 1)
        input_length: a scalar input length tensor
        max_length: a max_length to bound the response sentence length.

    Outputs:
        all_tokens: collections of words tokens
        all_scores: collections of words scores
    """

    def __init__(self, model):
        super(BeamSearchDecoder, self).__init__()
        self.model = model
        self.beam_width = config.BEAM_WIDTH

    class BeamSearchNode:
        def __init__(self, hidden, idx, value=None, previousNode=None, logp=0, depth=0):
            self.prevnode = previousNode
            self.hidden = hidden
            self.value = value
            self.idx = idx
            self.logp = logp
            self.depth = depth

        def eval(self):
            # for now, simply choose the one with maximum average
            return self.logp / float(self.depth)


    def forward(self, input_var, lengths, speaker_id, sos, eos):
        # decoding goes sentence by sentence
        encoder_outputs, encoder_hidden = self.model.encoder(input_var, lengths)

        decoder_hidden = self.model.cvt_hidden(encoder_hidden)

        # Transform speaker_id from int into tensor with shape=(1, 1)
        speaker_var = torch.tensor([speaker_id], dtype=torch.long, device=device)

        # Number of sentence to generate
        endnodes = []

        # Start with the start of the sentence token
        root_idx = torch.tensor(sos, dtype=torch.long, device=device)
        root = self.BeamSearchNode(decoder_hidden, root_idx)
        leaf = [root]

        for dep in range(config.MAX_LENGTH):
            candidates = []

            for prevnode in leaf:
                decoder_input = prevnode.idx.view(1, 1)

                # Forward pass through decoder
                # decode for one step using decoder
                decoder_output, decoder_hidden = self.model.decoder(decoder_input, speaker_var, prevnode.hidden, encoder_outputs)

                values, indexes = decoder_output.topk(self.beam_width)

                for i in range(self.beam_width):
                    idx = indexes[0][0][i]
                    value = values[0][0][i]
                    logp = math.log(value)

                    node = self.BeamSearchNode(decoder_hidden, idx, value, prevnode, logp + prevnode.logp, dep + 1)

                    candidates.append(node)

            candidates.sort(key=lambda n: n.logp, reverse=True)

            leaf = []
            for candiate in candidates[:self.beam_width]:
                if candiate.idx == eos:
                    endnodes.append(candiate)
                else:
                    leaf.append(candiate)

            # sentecnes don't need to be beam_width exactly, here just for simplicity
            if len(endnodes) >= self.beam_width:
                endnodes = endnodes[:self.beam_width]
                break

        # arrive max length before having enough results
        if len(endnodes) < self.beam_width:
            endnodes = endnodes + leaf[:self.beam_width - len(endnodes)]

        # choose the max/random from the results
        if config.BEAM_MODE == 'random':
            endnode = random.choice(endnodes)
        else:
            endnode = max(endnodes, key=lambda n: n.eval())

        tokens = []
        scores = []
        while endnode.idx != sos:
            tokens.append(endnode.idx)
            scores.append(endnode.value)
            endnode = endnode.prevnode

        tokens.reverse()
        scores.reverse()

        tokens = torch.stack(tokens)
        scores = torch.stack(scores)

        return tokens, scores
