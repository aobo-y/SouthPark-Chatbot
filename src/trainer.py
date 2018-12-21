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

DIR_PATH = os.path.dirname(__file__)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Inverse sigmoid decay
def teacher_forcing_rate(idx):
    k_factor = config.TF_RATE_DECAY_FACTOR
    rate = k_factor / (k_factor + math.exp(idx / k_factor))
    return rate

def mask_nll_loss(inp, target, mask):
    # Calculate our loss based on our decoder’s output tensor, the target tensor,
    # and a binary mask tensor describing the padding of the target tensor.
    cross_entropy = -torch.log(torch.gather(inp, 2, target.unsqueeze(2)).squeeze(2))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(DEVICE)
    return loss

class Trainer:
    '''Trainer to train the seq2seq model'''

    def __init__(self, model, word_map, person_map, checkpoint_mng):
        self.model = model

        self.checkpoint_mng = checkpoint_mng

        self.word_map = word_map
        self.person_map = person_map

        self.encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=config.LR)
        self.decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=config.LR * config.DECODER_LR)

        # trained iteration
        self.trained_iteration = 0

    def log(self, *args):
        '''formatted log output for training'''

        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{time}   ', *args)

    def load(self, checkpoint):
        '''load checkpoint'''

        self.trained_iteration = checkpoint['iteration']

        self.encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        self.decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    def reset_iter(self):
        self.trained_iteration = 0

    def train_batch(self, training_batch, tf_rate=1):
        '''
        train a batch of any batch size

        Inputs:
            training_batch: train data batch created by batch2TrainData
            tf_rate: teacher forcing rate, the smaller the rate the higher the scheduled sampling
        '''

        # extract fields from batch
        input_var, lengths, target_var, mask, max_target_len, speaker_var = training_batch

        # target var start with sos
        sos = self.word_map.get_index(config.SPECIAL_WORD_EMBEDDING_TOKENS['SOS'])
        batch_size = target_var.size(1)
        start_tensor = torch.full((1, batch_size), sos, dtype=torch.long)
        target_var = torch.cat((start_tensor, target_var))

        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Set DEVICE options
        input_var = input_var.to(DEVICE)
        lengths = lengths.to(DEVICE)
        target_var = target_var.to(DEVICE)
        mask = mask.to(DEVICE)
        speaker_var = speaker_var.to(DEVICE)

        # Pass through model
        output_var = self.model(input_var, lengths, target_var[:-1, :], speaker_var, max_target_len, tf_rate)

        # Calculate and accumulate loss
        loss = mask_nll_loss(output_var, target_var[1:, :], mask)

        # Perform backpropagation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), config.CLIP)
        _ = torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), config.CLIP)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss


    def train(self, pairs, n_iteration, batch_size=1, stage=None):
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
                self.log('Iter: {}; Percent: {:.1f}%; Avg loss: {:.4f}; TF rate: {:.4f}'.format(iteration, iteration / n_iteration * 100, print_loss_avg, tf_rate))
                print_loss = 0

            # Save checkpoint
            if iteration % config.SAVE_EVERY == 0:
                cp_name = f'{stage}_{iteration}'
                self.checkpoint_mng.save(cp_name, {
                    'iteration': iteration,
                    'loss': loss,
                    'stage': stage,
                    'model': self.model.state_dict(),
                    'en_opt': self.encoder_optimizer.state_dict(),
                    'de_opt': self.decoder_optimizer.state_dict(),
                    'word_map_dict': self.word_map.__dict__,
                    'person_map_dict': self.person_map.__dict__,
                })

                self.log('Save checkpoint:', cp_name)

