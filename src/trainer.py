"""
Train seq2seq
"""

import os
import math
import random
from datetime import datetime
import torch
from torch import optim
import config

from data_util import batch2TrainData, data_2_indexes

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Inverse sigmoid decay
def teacher_forcing_rate(idx):
    k_factor = config.TF_RATE_DECAY_FACTOR
    rate = k_factor / (k_factor + math.exp(idx / k_factor))
    return rate

# TODO consider use nn.CrossEntropyLoss
def mask_nll_loss(inp, target, mask):
    # Calculate our loss based on our decoder’s output tensor, the target tensor,
    # and a binary mask tensor describing the padding of the target tensor.
    n_total = mask.sum().float()
    cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(DEVICE)
    return loss, n_total.mean()

class Trainer:
    '''Trainer to train the seq2seq model'''

    def __init__(self, encoder, decoder, word_map, person_map, embedding, personas):
        self.encoder = encoder
        self.decoder = decoder
        self.word_map = word_map
        self.person_map = person_map
        self.embedding = embedding
        self.personas = personas

        self.encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.LR)
        self.decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.LR * config.DECODER_LR)

        # trained iteration
        self.trained_iteration = 0

    def log(self, string):
        '''formatted log output for training'''

        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{time}\t{string}')

    def load(self, checkpoint):
        '''load checkpoint'''

        self.trained_iteration = checkpoint['iteration']

        self.encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        self.decoder_optimizer.load_state_dict(checkpoint['de_opt'])


    def train_batch(self, training_batch, tf_rate=1):
        '''
        train a batch of any batch size

        Inputs:
            training_batch: train data batch created by batch2TrainData
            tf_rate: teacher forcing rate, the smaller the rate the higher the scheduled sampling
        '''

        # extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len, speaker_variable = training_batch

        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Set DEVICE options
        input_variable = input_variable.to(DEVICE)
        lengths = lengths.to(DEVICE)
        target_variable = target_variable.to(DEVICE)
        mask = mask.to(DEVICE)
        speaker_variable = speaker_variable.to(DEVICE)

        # Initialize variables
        loss = 0.
        print_loss = []
        n_totals = 0.

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        # Create initial decoder input
        sos = self.word_map.get_index(config.SPECIAL_WORD_EMBEDDING_TOKENS['SOS'])

        batch_size = input_variable.size(1)
        decoder_input = torch.LongTensor([[sos for _ in range(batch_size)]])
        decoder_input = decoder_input.to(DEVICE)

        decoder_layers = self.decoder.n_layers
        # Set initial decoder hidden state to the encoder's final hidden state
        if config.RNN_TYPE == 'LSTM':
            decoder_hidden = (encoder_hidden[0][:decoder_layers],   # hidden state
                            encoder_hidden[1][:decoder_layers])   # cell state
        else:
            decoder_hidden = encoder_hidden[:decoder_layers]

        # Forward batch of sequences through decoder one time step at a time
        for t in range(max_target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, speaker_variable, decoder_hidden, encoder_outputs)

            if random.random() < tf_rate:
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
            else:
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(DEVICE)

            # Calculate and accumulate loss
            mask_loss, n_total = mask_nll_loss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_loss.append(mask_loss.item() * n_total)
            n_totals += n_total


        # Perform backpropagation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), config.CLIP)
        _ = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), config.CLIP)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(print_loss) / n_totals


    def train(self, pairs, n_iteration, batch_size=1):
        """
        When we save our model, we save a tarball containing the encoder and decoder state_dicts (parameters),
        the optimizers’ state_dicts, the loss, the iteration, etc.
        After loading a checkpoint, we will be able to use the model parameters to run inference,
        or we can continue training right where we left off.
        """
        # convert sentence & speaker name to indexes
        index_pair = [data_2_indexes(pair, self.word_map, self.person_map) for pair in pairs]

        batch_size = config.BATCH_SIZE

        # Load batches for each iteration
        training_batches = [batch2TrainData([random.choice(index_pair) for _ in range(batch_size)], self.word_map)
                            for _ in range(n_iteration)]

        # Initializations
        print_loss = 0

        # Training loop
        start_iteration = self.trained_iteration + 1

        self.log(f'Start training from iteration {start_iteration} to {n_iteration}...')
        for iteration in range(start_iteration, n_iteration + 1):
            training_batch = training_batches[iteration - 1]

            tf_rate = teacher_forcing_rate(iteration)
            # run a training iteration with batch
            loss = self.train_batch(training_batch, tf_rate)
            print_loss += loss

            self.trained_iteration = iteration

            # Print progress
            if iteration % config.PRINT_EVERY == 0:
                print_loss_avg = print_loss / config.PRINT_EVERY
                self.log('Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}'.format(iteration, iteration / n_iteration * 100, print_loss_avg))
                print_loss = 0

            # Save checkpoint
            if iteration % config.SAVE_EVERY == 0:
                directory = os.path.join(config.SAVE_DIR, config.MODEL_NAME, '{}-{}_{}'.format(config.ENCODER_N_LAYERS, config.DECODER_N_LAYERS, config.HIDDEN_SIZE))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                filename = f'{config.TRAIN_MODE}_{iteration}.tar'
                filepath = os.path.join(directory, filename)

                torch.save({
                    'iteration': iteration,
                    'loss': loss,
                    'en': self.encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': self.encoder_optimizer.state_dict(),
                    'de_opt': self.decoder_optimizer.state_dict(),
                    'word_map_dict': self.word_map.__dict__,
                    'person_map_dict': self.person_map.__dict__,
                    'embedding': self.embedding.state_dict(),
                    'persona': self.personas.state_dict(),
                }, filepath)

                self.log(f'Save checkpoin {filename}')

