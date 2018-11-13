import time

import numpy as np
import tensorflow as tf

import config


class ChatBotModel:
    def __init__(self, forward_only, batch_size, use_lstm=True, beam_search=True):
        """forward_only: if set, we do not construct the backward pass in the model.
        """
        print('Initialize new model')
        self.fw_only = forward_only
        self.batch_size = batch_size
        self.beam_search = beam_search
        single_cell = tf.contrib.rnn.GRUCell(config.EMBEDDING_SIZE)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.LSTMCell(config.EMBEDDING_SIZE)
        if config.USE_DROPOUT:
            single_cell = tf.nn.rnn_cell.DropoutWrapper(
                single_cell,
                config.DROPOUT_INPUT_KEPP_PROB,
                config.DROPOUT_OUTPUT_KEEP_PROB,
                config.DROPOUT_STATE_KEPP_PROB)
        self.cell = tf.contrib.rnn.MultiRNNCell(
            [single_cell for _ in range(config.NUM_LAYERS)])

    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create placeholders')
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='encoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name='decoder{}'.format(i))
                               for i in range(config.BUCKETS[-1][1] + 1)]
        self.decoder_masks = [tf.placeholder(tf.float32, shape=[None], name='mask{}'.format(i))
                              for i in range(config.BUCKETS[-1][1] + 1)]

        # Our targets are decoder inputs shifted by one (to ignore <GO> symbol)
        self.targets = self.decoder_inputs[1:]

    def _inference(self):
        print('Create inference')
        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            w = tf.get_variable(
                'proj_w', [config.EMBEDDING_SIZE, config.DEC_VOCAB])
            b = tf.get_variable('proj_b', [config.DEC_VOCAB])
            self.output_projection = (w, b)

        def sampled_loss(logits, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(weights=tf.transpose(w),
                                              biases=b,
                                              inputs=logits,
                                              labels=labels,
                                              num_sampled=config.NUM_SAMPLES,
                                              num_classes=config.DEC_VOCAB)
        self.softmax_loss_function = sampled_loss

    def _create_loss(self):
        print('Creating loss... \nIt might take a couple of minutes depending on how many buckets you have.')
        start = time.time()

        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            setattr(tf.contrib.rnn.GRUCell,
                    '__deepcopy__', lambda self, _: self)
            setattr(tf.contrib.rnn.MultiRNNCell,
                    '__deepcopy__', lambda self, _: self)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, self.cell,
                num_encoder_symbols=config.ENC_VOCAB,
                num_decoder_symbols=config.DEC_VOCAB,
                embedding_size=config.EMBEDDING_SIZE,
                num_heads=config.ATTENTION_HEADS,
                output_projection=self.output_projection,
                feed_previous=do_decode)

            # if self.fw_only:
            #     self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            #                                 self.encoder_inputs,
            #                                 self.decoder_inputs,
            #                                 self.targets,
            #                                 self.decoder_masks,
            #                                 config.BUCKETS,
            #                                 lambda x, y: _seq2seq_f(x, y, True),
            #                                 softmax_loss_function=self.softmax_loss_function)
            #     # If we use output projection, we need to project outputs for decoding.
            #     if self.output_projection:
            #         for bucket in range(len(config.BUCKETS)):
            #             self.outputs[bucket] = [tf.matmul(output,
            #                                     self.output_projection[0]) + self.output_projection[1]
            #                                     for output in self.outputs[bucket]]
            if self.fw_only:
                if beam_search:  # 如果是beam_search的话，则调用自己写的embedding_attention_seq2seq函数，而不是legacy_seq2seq下面的
                    self.beam_outputs, _, self.beam_path, self.beam_symbol = embedding_attention_seq2seq(
                        self.encoder_inputs,
                        self.decoder_inputs,
                        self.cell,
                        num_encoder_symbols=config.ENC_VOCAB,
                        num_decoder_symbols=config.DEC_VOCAB,
                        embedding_size=config.EMBEDDING_SIZE,
                        output_projection=self.output_projection,
                        feed_previous=True)
                else:
                    pass  # Write me in the future!
            # else:
            #     self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            #                                 self.encoder_inputs,
            #                                 self.decoder_inputs,
            #                                 self.targets,
            #                                 self.decoder_masks,
            #                                 config.BUCKETS,
            #                                 lambda x, y: _seq2seq_f(x, y, True),
            #                                 softmax_loss_function=self.softmax_loss_function)
            # print('Time:', time.time() - start)
            else:
                # 因为不需要将output作为下一时刻的输入，所以不用output_projection
                self.outputs, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                    self.encoder_inputs, self.decoder_inputs, self.cell, num_encoder_symbols=config.ENC_VOCAB,
                    num_decoder_symbols=config.DEC_VOCAB, embedding_size=config.EMBEDDING_SIZE, output_projection=self.output_projection,
                    feed_previous=False)
                self.loss = tf.contrib.legacy_seq2seq.sequence_loss(
                    self.outputs, self.decoder_masks, self.targets, softmax_loss_function=self.softmax_loss_function)

    def _creat_optimizer(self):
        print('Create optimizer... \nIt might take a couple of minutes depending on how many buckets you have.')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(
                0, dtype=tf.int32, trainable=False, name='global_step')

            if not self.fw_only:
                self.optimizer = tf.train.GradientDescentOptimizer(config.LR)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                start = time.time()
                for bucket in range(len(config.BUCKETS)):

                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket],
                                                                              trainables),
                                                                 config.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables),
                                                                         global_step=self.global_step))
                    print('Creating opt for bucket {} took {} seconds'.format(
                        bucket, time.time() - start))
                    start = time.time()

    def _create_summary(self):
        pass

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._creat_optimizer()
        self._create_summary()
