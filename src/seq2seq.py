"""
Train seq2seq
"""

import os
import math
import random
import torch
import config

from data_util import batch2TrainData, data_2_indexes

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Inverse sigmoid decay
def teacher_forcing_rate(idx):
    k_factor = config.TF_RATE_DECAY_FACTOR
    rate = k_factor / (k_factor + math.exp(idx / k_factor))
    return rate

def maskNLLLoss(inp, target, mask):
    # Calculate our loss based on our decoder’s output tensor, the target tensor,
    # and a binary mask tensor describing the padding of the target tensor.
    n_total = mask.sum().float()
    cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, n_total.mean()


def train(word_map, input_variable, lengths, target_variable, mask, max_target_len, speaker_variable,
          encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, iteration):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    speaker_variable = speaker_variable.to(device)

    # Initialize variables
    loss = 0.
    print_loss = []
    n_totals = 0.

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input
    sos = word_map.get_index(config.SPECIAL_WORD_EMBEDDING_TOKENS['SOS'])
    decoder_input = torch.LongTensor([[sos for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    if config.RNN_TYPE == 'LSTM':
        decoder_hidden = (encoder_hidden[0][:decoder.n_layers],   # hidden state
                          encoder_hidden[1][:decoder.n_layers])   # cell state
    else:
        decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Forward batch of sequences through decoder one time step at a time
    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(decoder_input, speaker_variable, decoder_hidden, encoder_outputs)

        if random.random() < teacher_forcing_rate(iteration):
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
        else:
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)

        # Calculate and accumulate loss
        mask_loss, n_total = maskNLLLoss(decoder_output, target_variable[t], mask[t])
        loss += mask_loss
        print_loss.append(mask_loss.item() * n_total)
        n_totals += n_total


    # Perform backpropagation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.CLIP)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.CLIP)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_loss)/n_totals


def trainIters(word_map, person_map, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, personas, n_iteration, corpus_name, start_iteration):
    """
    When we save our model, we save a tarball containing the encoder and decoder state_dicts (parameters),
    the optimizers’ state_dicts, the loss, the iteration, etc.
    After loading a checkpoint, we will be able to use the model parameters to run inference,
    or we can continue training right where we left off.
    """
    # convert sentence & speaker name to indexes
    index_pair = [data_2_indexes(pair, word_map, person_map) for pair in pairs]

    batch_size = config.BATCH_SIZE

    # Load batches for each iteration
    training_batches = [batch2TrainData([random.choice(index_pair) for _ in range(batch_size)], word_map)
                        for _ in range(n_iteration)]

    # Initializations
    print_loss = 0

    # Training loop
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len, speaker_variable = training_batch

        # run a training iteration with batch
        loss = train(word_map, input_variable, lengths, target_variable, mask, max_target_len, speaker_variable,
                     encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size, iteration)
        print_loss += loss

        # Print progress
        if iteration % config.PRINT_EVERY == 0:
            print_loss_avg = print_loss / config.PRINT_EVERY
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if iteration % config.SAVE_EVERY == 0:
            directory = os.path.join(config.SAVE_DIR, config.MODEL_NAME, corpus_name, '{}-{}_{}'.format(config.ENCODER_N_LAYERS, config.DECODER_N_LAYERS, config.HIDDEN_SIZE))
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'word_map_dict': word_map.__dict__,
                'person_map_dict': person_map.__dict__,
                'embedding': embedding.state_dict(),
                'persona': personas.state_dict(),
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
