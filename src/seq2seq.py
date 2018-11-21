import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import config
from data_util import batch2TrainData, normalizeString, indexesFromSentence


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class EncoderRNN(nn.Module):
  """
  Our encoder is a multi-layered Gated Recurrent Unit, invented by Cho et al. in 2014.
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
    #    because our input size is a word embedding with number of features == hidden_size
    # b) use bidirectional GRU to capture context words
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                      dropout=(0 if n_layers==1 else dropout),
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


class Attn(torch.nn.Module):
  """
  We implement the “Attention Layer” proposed by Luong et al. in 2015 as a separate nn.Module called Attn.
  The output of this module is a softmax normalized weights tensor of shape (batch_size, 1, max_length).

  """

  def __init__(self, method, hidden_size):
    super(Attn, self).__init__()
    self.hidden_size = hidden_size
    self.method = method
    if self.method not in ['dot', 'general', 'concat']:
      raise ValueError(self.method, 'is not an appropriate attention method.')
    if self.method == 'general':
      self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
    elif self.method == 'concat':
      self.attn = torch.nn.Linear(self.hidden_size*2, hidden_size)
      self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

  def dot_score(self, hidden, encoder_output):
	# shape = (max_length, batch_size)
    return torch.sum(hidden*encoder_output, dim=2)

  def general_score(self, hidden, encoder_output):
	# shape = (max_length, batch_size, hidden_size)
    energy = self.attn(encoder_output)
	# shape = (max_length, batch_size)
    return torch.sum(hidden*energy, dim=2)

  def concat_score(self, hidden, encoder_output):
	# shape = (max_length, batch_size, hidden_size)
    energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
	# shape = (max_length, batch_size)
    return torch.sum(self.v*energy, dim=2)

  def forward(self, hidden, encoder_outputs):
    # Calculate the attention weights (energies) based on the given method
    if self.method == 'general':
      attn_energies = self.general_score(hidden, encoder_outputs)
    elif self.method == 'concat':
      attn_energies = self.concat_score(hidden, encoder_outputs)
    elif self.method == 'dot':
      attn_energies = self.dot_score(hidden, encoder_outputs)

    # Transpose max_length and batch_size dimensions
	# shape = (batch_size, max_length)
    attn_energies = attn_energies.t()

    # Return the softmax normalized probability scores (with added dimension)
	# shape = (batch_size, 1, max_length)
    return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
  """
  For the decoder, we manually feed our batch one time step at a time.
  This means that our embedded word tensor and GRU output will both have shape (1, batch_size, hidden_size).

  Inputs:
    input_step: one time step (one word) of input sequence batch; shape=(1, batch_size)
    last_hidden: final hidden layer of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)
    encoder_outputs: encoder model’s output; shape=(max_length, batch_size, hidden_size)

  Outputs:
    output: softmax normalized tensor giving probabilities of each word being the correct next word in the decoded sequence;
            shape=(batch_size, voc.num_words)
    hidden: final hidden state of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)
  """

  def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.5):
    super(LuongAttnDecoderRNN, self).__init__()
    self.attn_model = attn_model
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout = dropout

    # Define layers
    self.embedding = embedding
    self.embedding_dropout = nn.Dropout(dropout)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                      dropout=(0 if n_layers==1 else dropout))
    self.concat = nn.Linear(hidden_size*2, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)
    self.attn = Attn(attn_model, hidden_size)

  def forward(self, input_step, last_hidden, encoder_outputs):
    # Note: we run this one step(word) at a time

    # Get embedding of current input word
	# shape = (1, batch_size, hidden_size)
    embedded = self.embedding(input_step)
    embedded = self.embedding_dropout(embedded)
    # Forward through GRU
	# rnn_output shape = (1, batch_size, hidden_size)
	# hidden shape = (n_layers*num_directions, batch_size, hidden_size)
    rnn_output, hidden = self.gru(embedded, last_hidden)
    # Calculate attention weights from the current GRU output
	# attn_weights shape = (batch_size, 1, max_length)
    attn_weights = self.attn(rnn_output, encoder_outputs)
    # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
	# shape = (batch_size, hidden_size, 1)
    context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
    # Concatenate weighted context vector and GRU output using Luong eq. 5
	# shape = (batch_size, hidden_size)
    rnn_output = rnn_output.squeeze(0)
	# shape = (batch_size, hidden_size)
    context = context.squeeze(1)
	# shape = (batch_size, hidden_size*2)
    concat_input = torch.cat((rnn_output, context), 1)
	# shape = (batch_size, hidden_size)
    concat_output = torch.tanh(self.concat(concat_input))
    # Predict next word using Luong eq. 6
	# shape = (batch_size, voc_dict.length)
    output = self.out(concat_output)
    output = F.softmax(output, dim=1)
    return output, hidden


class GreedySearchDecoder(nn.Module):
  """
  Greedy decoding is the decoding method that we use during training when we are NOT using teacher forcing.

  Inputs:
    input_seq: an input sequence of shape (input_seq length, 1)
    input_length: a scalar input length tensor
    max_length: a max_length to bound the response sentence length.

  Outputs:
    all_tokens: collections of words tokens
    all_scores: collections of words scores
  """

  def __init__(self, encoder, decoder):
    super(GreedySearchDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

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
      decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
      # Obtain most likely word token and its softmax score
      decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
      # Record token and score
      all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
      all_scores = torch.cat((all_scores, decoder_scores), dim=0)
      # Prepare current token to be next decoder input (add a dimension)
      decoder_input = torch.unsqueeze(decoder_input, 0)
    # Returen collections of word tokens and scores
    return all_tokens, all_scores


def maskNLLLoss(inp, target, mask):
  # Calculate our loss based on our decoder’s output tensor, the target tensor,
  # and a binary mask tensor describing the padding of the target tensor.
  nTotal = mask.sum().float()
  crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
  loss = crossEntropy.masked_select(mask).mean()
  loss = loss.to(device)
  return loss, nTotal.mean()


def train(input_variable, lengths, target_variable, mask, max_target_len,
          encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
          batch_size, clip, teacher_forcing_ratio, max_length=config.MAX_LENGTH):
  # Zero gradients
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  # Set device options
  input_variable = input_variable.to(device)
  lengths = lengths.to(device)
  target_variable = target_variable.to(device)
  mask = mask.to(device)

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
      decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
      # Teacher forcing: next input is current target
      decoder_input = target_variable[t].view(1, -1)
      # Calculate and accumulate loss
      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
      loss += mask_loss
      print_loss.append(mask_loss.item()*nTotal)
      n_totals += nTotal
  else:
    for t in range(max_target_len):
      decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
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
               embedding, encoder_n_layers, decoder_n_layers, hidden_size, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, teacher_forcing_ratio, corpus_name, loadFilename):
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
  print('Initializing...')
  start_iteration = 1
  print_loss = 0
  if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    start_iteration = checkpoint['iteration']+1

  # Training loop
  print('Training')
  for iteration in range(start_iteration, n_iteration+1):
    training_batch = training_batches[iteration-1]
    # extract fields from batch
    input_variable, lengths, target_variable, mask, max_target_len = training_batch

    # run a training iteration with batch
    loss = train(input_variable, lengths, target_variable, mask, max_target_len,
                 encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size,
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
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=config.MAX_LENGTH):
  ### Format input sentence as a batch
  # words -> indexes
  indexes_batch = [indexesFromSentence(voc, sentence)]
  # create lengths tensor
  lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
  # transpose dimensions of batch to match model's expectations
  input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
  input_batch = input_batch.to(device)
  lengths = lengths.to(device)
  # decode sentence with searcher
  tokens, scores = searcher(input_batch, lengths, max_length)
  # indexes -> words
  decoded_words = [voc.index2word[token.item()] for token in tokens]
  return decoded_words


# Evaluate inputs from user input (stdin)
def evaluateInput(encoder, decoder, searcher, voc):
  input_sentence = ''
  while(1):
    try:
      # get input sentence
      input_sentence = input('> ')
      # check if it is quit case
      if input_sentence == 'q' or input_sentence =='quit':
        break
      input_sentence = normalizeString(input_sentence)
      # evaluate sentence
      output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
      # format and print reponse sentence
      output_words[:] = [x for x in output_words if not (x=='EOS' or x=='PAD')]
      print('Bot:', ' '.join(output_words))
    except KeyError:
      print('Error: Encountered unknown word.')


# Normalize input sentence and call evaluate()
def evaluateExample(sentence, encoder, decoder, searcher, voc):
  print('> '+sentence)
  # normalize sentence
  input_sentence = normalizeString(sentence)
  # evaluate sentence
  output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
  output_words[:] = [x for x in output_words if not (x=='EOS' or x=='PAD')]
  print('Bot:', ' '.join(output_words))
