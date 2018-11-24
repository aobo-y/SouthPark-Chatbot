"""
Greedy decoding is the decoding method
that we use during training when we are NOT using teacher forcing.
"""

import operator
import torch
from torch import nn
import torch.nn.functional as F
from queue import PriorityQueue
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
            speaker_id = speaker_id.to(device)
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
    
    def __init__(self, encoder, decoder, speaker_id):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.speaker = speaker_id

    class BeamSearchNode(object):
        def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
            '''
            :param hiddenstate:
            :param previousNode:
            :param wordId:
            :param logProb:
            :param length:
            '''
            self.h = hiddenstate
            self.prevNode = previousNode
            self.wordid = wordId
            self.logp = logProb
            self.leng = length

        def eval(self, alpha=1.0):
            reward = 0
            # Add here a function for shaping a reward
            return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def forward(self, input_seq, input_length, max_length):
        beam_width = config.BEAM_WIDTH
        topk = 1  # how many sentence do you want to generate
        
        # decoding goes sentence by sentence
        # for idx in range(target_tensor.size(0)):
        batch_size = 1 # TODO: don't know how to calculate
        for _ in range(batch_size):
            encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
            decoder_hidden = encoder_hidden[:self.decoder.n_layers]

            # Start with the start of the sentence token
            decoder_input = torch.ones(1, 1, device=device, dtype=torch.long)*config.SOS_TOKEN

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = self.BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == config.EOS_TOKEN and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # Forward pass through decoder
                # Transform speaker_id from int into tensor with shape=(1, 1)
                speaker_id = torch.LongTensor([self.speaker])
                speaker_id = torch.unsqueeze(speaker_id, 1)
                speaker_id = speaker_id.to(device)
                # decode for one step using decoder
                decoder_output, decoder_hidden = self.decoder(decoder_input, speaker_id, decoder_hidden, encoder_outputs)
                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = self.BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            # Record token and score
            all_tokens = []
            all_scores = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    all_tokens.append(n.wordid)
                    all_scores.append(n.logp)
                all_tokens = all_tokens[::-1][1:]
                all_scores = all_scores[::-1][1:]

        return all_tokens, all_scores
