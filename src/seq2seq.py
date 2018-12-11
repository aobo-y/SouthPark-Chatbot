''' Core Seq2seq model '''

import torch
from torch import nn
from seq_encoder import EncoderRNN
from seq_decoder_persona import DecoderRNN

class Seq2Seq(nn.Module):
    '''
    A dedicated Persona Chatbot Seq2Seq Model
    '''


    def __init__(self, word_ebd_shape, persona_ebd_shape, hidden_size, layer_size=1, dropout_rate=0, rnn_type='LSTM', attn_type='dot'):
        '''
        word_ebd_shape: word embedding shape, tuple (word count, dim)
        persona_ebd_shape: persona embedding shape, tuple (person count, dim)
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


    def forward(self, input_var, searcher):
        pass

