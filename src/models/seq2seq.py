''' Core Seq2seq model '''

import random
import torch
from torch import nn
from .encoder import EncoderRNN
from .decoder import DecoderRNN

class Seq2Seq(nn.Module):
    '''
    A dedicated Persona Chatbot Seq2Seq Model

    Inputs:
        input_var: sequence of batch of indices; shape=(seq_len, batch_size)
        lengths: batch of lengths of the input; shape=(batch_size)
        target_var: sequence of batch of indices; shape=(seq_len, batch_size)
        speaker_var: batch of speakers; shape=(batch_size)
        max_length: maximum length to generate; int
        tf_rate: teacher forcing rate

    Outputs:
        output_var: sequence of batch of output vector; shape=(seq_len, batch_size, vocabulary_size)

    '''


    def __init__(self, word_ebd_shape, persona_ebd_shape, hidden_size, layer_size=1, dropout_rate=0, rnn_type='LSTM', attn_type='dot'):
        '''
        word_ebd_shape: word embedding shape, tuple (vocabulary_size, dim)
        persona_ebd_shape: persona embedding shape, tuple (person_count, dim)
        hidden_size: hidden vector size for both encoder & decoder
        layer_size: number of layers for both encoder & decoder
        rnn_type: rnn model type, allow ['GRU', 'LSTM']
        attn_type: attention type, allow ['dot', 'general', 'concat']
        '''

        super(Seq2Seq, self).__init__()

        self.rnn_type = rnn_type

        self.word_ebd = nn.Embedding(*word_ebd_shape)

        # Initialize persona embedding with 0
        weight = torch.zeros(persona_ebd_shape)
        self.persona_ebd = nn.Embedding.from_pretrained(weight, False)

        # Initialize encoder & decoder models
        self.encoder = EncoderRNN(self.word_ebd, hidden_size, layer_size, dropout_rate, rnn_type)

        self.decoder = DecoderRNN(self.word_ebd, self.persona_ebd, hidden_size, word_ebd_shape[0], layer_size, dropout_rate, rnn_type, attn_type)

    def load_pretrained_word_ebd(self, weight):
        self.word_ebd.weight = nn.Parameter(weight)

    def freeze_persona(self):
        self.persona_ebd.weight.requires_grad = False


    def cvt_hidden(self, encoder_hidden):
        ''' convert encoder hidden to decoder hidden '''

        decoder_layers = self.decoder.n_layers
        # Set initial decoder hidden state to the encoder's final hidden state
        if self.rnn_type == 'LSTM':
            decoder_hidden = (encoder_hidden[0][:decoder_layers],   # hidden state
                            encoder_hidden[1][:decoder_layers])   # cell state
        else:
            decoder_hidden = encoder_hidden[:decoder_layers]

        return decoder_hidden


    def forward(self, input_var, lengths, target_var, speaker_var, max_length, tf_rate=0):
        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_var, lengths)

        decoder_hidden = self.cvt_hidden(encoder_hidden)

        # teacher forcing applies to entire sequence
        teacher_forcing = random.random() < tf_rate

        if teacher_forcing:
            # Teacher forcing: feed the entire sequence of batch ground truth
            output_var, decoder_hidden = self.decoder(target_var, speaker_var, decoder_hidden, encoder_outputs)
        else:
            # No teacher forcing: Forward batch of sequences through decoder one time step at a time
            outputs = []
            decoder_var = target_var[0].view(1, -1)

            for _ in range(max_length):
                output_step, decoder_hidden = self.decoder(decoder_var, speaker_var, decoder_hidden, encoder_outputs)

                outputs.append(output_step)

                # next input is decoder's own current output
                _, decoder_var = output_step.max(2)

            output_var = torch.cat(outputs, 0)

        return output_var
