"""
Train seq2seq
"""

import os
import random
import torch
import config

from data_util import batch2TrainData

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

def maskNLLLoss(inp, target, mask):
    # Calculate our loss based on our decoder’s output tensor, the target tensor,
    # and a binary mask tensor describing the padding of the target tensor.
    nTotal = mask.sum().float()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.mean()


def train(input_variable, lengths, target_variable, mask, max_target_len, speaker_id,
          encoder, decoder, encoder_optimizer, decoder_optimizer,
          batch_size, clip, teacher_forcing_ratio, max_length = config.MAX_LENGTH):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    speaker_id = speaker_id.to(device)

    # Initialize variables
    loss = 0.
    print_loss = []
    n_totals = 0.

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input
    decoder_input = torch.LongTensor([[config.SOS_TOKEN for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, speaker_id[t], decoder_hidden, encoder_outputs)
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_loss.append(mask_loss.item()*nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, speaker_id[t], decoder_hidden, encoder_outputs)
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_loss.append(mask_loss.item()*nTotal)
            n_totals += nTotal

    # Perform backpropagation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_loss)/n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, personas, encoder_n_layers, decoder_n_layers, hidden_size, save_dir, n_iteration,
               batch_size, print_every, save_every, clip, teacher_forcing_ratio, corpus_name, load_filename):
    """
    When we save our model, we save a tarball containing the encoder and decoder state_dicts (parameters),
    the optimizers’ state_dicts, the loss, the iteration, etc.
    After loading a checkpoint, we will be able to use the model parameters to run inference,
    or we can continue training right where we left off.
    """

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    start_iteration = 1
    print_loss = 0
    if load_filename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(load_filename)
        start_iteration = checkpoint['iteration']+1

    # Training loop
    for iteration in range(start_iteration, n_iteration+1):
        training_batch = training_batches[iteration-1]
        # extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len, speaker = training_batch

        # run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, speaker,
                     encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size,
                     clip, teacher_forcing_ratio)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if iteration % save_every == 0:
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)

            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict(),
                'persona': personas.state_dict(),
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
