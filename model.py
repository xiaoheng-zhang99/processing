import tensorflow as tf
#from tensorflow.contrib import rnn
#from tensorflow.contrib.rnn import DropoutWrapper



from tensorflow.core.framework import summary_pb2
from random import shuffle
import numpy as np
from config import *


class Encoder_Text:

    def __init__(self, dic_size,
                 batch_size,
                 encoder_size,
                 num_layer, lr,
                 hidden_dim,
                 dr):

        self.dic_size = dic_size
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.num_layers = num_layer
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.dr = dr
        self.encoder_inputs = []
        self.encoder_seq_length = []
        self.y_labels = []
        self.M = None
        self.b = None
        self.y = None
        self.optimizer = None
        self.batch_loss = None
        self.loss = 0
        self.batch_prob = None
        self.embed_dim = 300
        # for global counter
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        print('[launch-text] placeholders')
        with tf.name_scope('text_placeholder'):
            self.encoder_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.encoder_size],
                                                 name="encoder")  # [batch,time_step]
            self.encoder_seq = tf.placeholder(tf.int32, shape=[self.batch_size],
                                              name="encoder_seq")  # [batch] - valid word step
            self.y_labels = tf.placeholder(tf.float32, shape=[self.batch_size, N_CATEGORY], name="label")
            self.dr_prob = tf.placeholder(tf.float32, name="dropout")
            # for using pre-trained embedding
            self.embedding_placeholder = tf.placeholder(tf.float32, shape=[self.dic_size, self.embed_dim],
                                                        name="embedding_placeholder")

    def _create_embedding(self):
        print('[launch-text] create embedding')
        with tf.name_scope('embed_layer'):
            self.embed_matrix = tf.Variable(tf.random.normal([self.dic_size, self.embed_dim],
                                                             mean=0.0,
                                                             stddev=0.01,
                                                             dtype=tf.float32,
                                                             seed=None),
                                            trainable=EMBEDDING_TRAIN,
                                            name='embed_matrix')

            self.embed_en = tf.nn.embedding_lookup(self.embed_matrix, self.encoder_inputs, name='embed_encoder')

    def _use_external_embedding(self):
        self.embedding_init = self.embed_matrix.assign(self.embedding_placeholder)


    # cell instance with drop-out wrapper applied
    def gru_drop_out_cell(self):
        gru_cell=tf.contrib.rnn.GRUCell(num_units=self.hidden_dim)
        return tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob=self.dr_prob,
                                             output_keep_prob=self.dr_prob)

    def test_cross_entropy_with_logit(self, logits, labels):
        x = logits
        z = labels
        return tf.maximum(x, 0) - x * z + tf.log(1 + tf.exp(-tf.abs(x)))

    def _create_gru_model(self):
        print('[launch-text] create gru cell')
        with tf.name_scope('text_RNN') as scope:
            with tf.variable_scope("text_GRU", reuse=False, initializer=tf.orthogonal_initializer()):
                cells_en = tf.contrib.rnn.MultiRNNCell([self.gru_drop_out_cell() for _ in range(self.num_layers)])

                (self.outputs_en, last_states_en) = tf.nn.dynamic_rnn(
                    cell=cells_en,
                    inputs=self.embed_en,
                    dtype=tf.float32,
                    sequence_length=self.encoder_seq,
                    time_major=False)

                self.final_encoder = last_states_en[-1]
        self.final_encoder_dimension = self.hidden_dim

    def _create_optimizer(self):
        print('[launch-text] create optimizer')
        with tf.name_scope('text_optimizer') as scope:
            opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
            gvs = opt_func.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(t=grad, clip_value_min=-10, clip_value_max=10), var) for grad, var in gvs]
            self.optimizer = opt_func.apply_gradients(grads_and_vars=capped_gvs, global_step=self.global_step)

    def _create_output_layers(self):
        print
        '[launch-text] create output projection layer'

        with tf.name_scope('text_output_layer') as scope:
            self.M = tf.Variable(tf.random_uniform([self.final_encoder_dimension, N_CATEGORY],
                                                   minval=-0.25,
                                                   maxval=0.25,
                                                   dtype=tf.float32,
                                                   seed=None),
                                 trainable=True,
                                 name="similarity_matrix")

            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                 trainable=True,
                                 name="output_bias")

            # e * M + b
            self.batch_pred = tf.matmul(self.final_encoder, self.M) + self.b

        with tf.name_scope('loss') as scope:
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.batch_pred, labels=self.y_labels)
            self.loss = tf.reduce_mean(self.batch_loss)
    def _create_output_layers_for_multi(self):
        print('[launch-text] create output projection layer for multi')

        with tf.name_scope('text_output_layer') as scope:
            self.M = tf.Variable(tf.random_uniform([self.final_encoder_dimension, int(self.final_encoder_dimension / 2)],
                                                   minval=-0.25,
                                                   maxval=0.25,
                                                   dtype=tf.float32,
                                                   seed=None),
                                 trainable=True,
                                 name="similarity_matrix")

            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                 trainable=True,
                                 name="output_bias")

            # e * M + b
            self.batch_pred = tf.matmul(self.final_encoder, self.M) + self.b
    def _create_summary(self):
        print('[launch-text] create summary')

        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholders()
        self._create_embedding()
        self._use_external_embedding()
        self._create_gru_model()
        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()

#encoder_speech

class Encoder_Audio:

    def __init__(self, batch_size,
                 encoder_size,
                 num_layer, lr,
                 hidden_dim,
                 dr):
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.num_layers = num_layer
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.dr = dr

        self.encoder_inputs = []
        self.encoder_seq_length = []
        self.y_labels = []

        self.M = None
        self.b = None

        self.y = None
        self.optimizer = None

        self.batch_loss = None
        self.loss = 0
        self.batch_prob = None

        # for global counter
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        print
        '[launch-audio] placeholders'
        with tf.name_scope('audio_placeholder'):
            self.encoder_inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.encoder_size, N_AUDIO_MFCC],
                                                 name="encoder")  # [batch, time_step, audio]
            self.encoder_seq = tf.placeholder(tf.int32, shape=[self.batch_size],
                                              name="encoder_seq")  # [batch] - valid audio step
            self.encoder_prosody = tf.placeholder(tf.float32, shape=[self.batch_size, N_AUDIO_PROSODY],
                                                  name="encoder_prosody")
            self.y_labels = tf.placeholder(tf.float32, shape=[self.batch_size, N_CATEGORY], name="label")
            self.dr_prob = tf.placeholder(tf.float32, name="dropout")

    # cell instance
    def gru_cell(self):
        return tf.contrib.rnn.GRUCell(num_units=self.hidden_dim)

    # cell instance with drop-out wrapper applied
    def gru_drop_out_cell(self):
        return tf.contrib.rnn.DropoutWrapper(self.gru_cell(), input_keep_prob=self.dr_prob,
                                             output_keep_prob=self.dr_prob)

    def test_cross_entropy_with_logit(self, logits, labels):
        x = logits
        z = labels
        return tf.maximum(x, 0) - x * z + tf.log(1 + tf.exp(-tf.abs(x)))

    def _create_gru_model(self):
        print
        '[launch-audio] create gru cell'

        with tf.name_scope('audio_RNN') as scope:
            with tf.variable_scope("audio_GRU", reuse=False, initializer=tf.orthogonal_initializer()):
                cells_en = tf.contrib.rnn.MultiRNNCell([self.gru_drop_out_cell() for _ in range(self.num_layers)])

                (self.outputs_en, last_states_en) = tf.nn.dynamic_rnn(
                    cell=cells_en,
                    inputs=self.encoder_inputs,
                    dtype=tf.float32,
                    sequence_length=self.encoder_seq,
                    time_major=False)

                self.final_encoder = last_states_en[-1]

        self.final_encoder_dimension = self.hidden_dim

    def _add_prosody(self):
        print
        '[launch-audio] add prosody feature, dim: ' + str(N_AUDIO_PROSODY)
        self.final_encoder = tf.concat([self.final_encoder, self.encoder_prosody], axis=1)
        self.final_encoder_dimension = self.hidden_dim + N_AUDIO_PROSODY

    def _create_output_layers(self):
        print
        '[launch-audio] create output projection layer'

        with tf.name_scope('audio_output_layer') as scope:
            self.M = tf.Variable(tf.random_uniform([self.final_encoder_dimension, N_CATEGORY],
                                                   minval=-0.25,
                                                   maxval=0.25,
                                                   dtype=tf.float32,
                                                   seed=None),
                                 trainable=True,
                                 name="similarity_matrix")

            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                 trainable=True,
                                 name="output_bias")

            # e * M + b
            self.batch_pred = tf.matmul(self.final_encoder, self.M) + self.b

        with tf.name_scope('loss') as scope:
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.batch_pred, labels=self.y_labels)
            self.loss = tf.reduce_mean(self.batch_loss)

    def _create_output_layers_for_multi(self):
        print
        '[launch-audio] create output projection layer for multi'

        with tf.name_scope('audio_output_layer') as scope:

            self.M = tf.Variable(tf.random.uniform([self.final_encoder_dimension, (int(self.final_encoder_dimension / 2))],
                                                   minval=-0.25,
                                                   maxval=0.25,
                                                   dtype=tf.float32,
                                                   seed=None),trainable=True,name="similarity_matrix")

            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                 trainable=True,
                                 name="output_bias")

            # e * M + b
            self.batch_pred = tf.matmul(self.final_encoder, self.M) + self.b

    def _create_optimizer(self):
        print
        '[launch-audio] create optimizer'

        with tf.name_scope('audio_optimizer') as scope:
            opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
            gvs = opt_func.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(t=grad, clip_value_min=-10, clip_value_max=10), var) for grad, var in gvs]
            self.optimizer = opt_func.apply_gradients(grads_and_vars=capped_gvs, global_step=self.global_step)

    def _create_summary(self):
        print('[launch-audio] create summary')

        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholders()
        self._create_gru_model()
        self._add_prosody()
        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()


class Encoder_Multi:

    def __init__(self,
                 batch_size,
                 lr,
                 encoder_size_audio,  # for audio
                 num_layer_audio,
                 hidden_dim_audio,
                 dr_audio,
                 dic_size,  # for text
                 encoder_size_text,
                 num_layer_text,
                 hidden_dim_text,
                 dr_text
                 ):

        # for audio
        self.encoder_size_audio = encoder_size_audio
        self.num_layers_audio = num_layer_audio
        self.hidden_dim_audio = hidden_dim_audio
        self.dr_audio = dr_audio
        self.encoder_inputs_audio = []
        self.encoder_seq_length_audio = []

        # for text
        self.dic_size = dic_size
        self.encoder_size_text = encoder_size_text
        self.num_layers_text = num_layer_text
        self.hidden_dim_text = hidden_dim_text
        self.dr_text = dr_text
        self.encoder_inputs_text = []
        self.encoder_seq_length_text = []

        # common
        self.batch_size = batch_size
        self.lr = lr
        self.y_labels = []
        self.M = None
        self.b = None
        self.y = None
        self.optimizer = None
        self.batch_loss = None
        self.loss = 0
        self.batch_prob = None
        self.embed_dim = 300
        # for global counter
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        print( '[launch-multi] placeholders')
        with tf.name_scope('multi_placeholder'):
            # for audio
            self.encoder_inputs_audio = self.model_audio.encoder_inputs  # [batch, time_step, audio]
            self.encoder_seq_audio = self.model_audio.encoder_seq
            self.encoder_prosody = self.model_audio.encoder_prosody
            self.dr_prob_audio = self.model_audio.dr_prob

            # for text
            self.encoder_inputs_text = self.model_text.encoder_inputs
            self.encoder_seq_text = self.model_text.encoder_seq
            self.dr_prob_text = self.model_text.dr_prob

            # common
            self.y_labels = tf.placeholder(tf.float32, shape=[self.batch_size, N_CATEGORY], name="label")

            # for using pre-trained embedding
            self.embedding_placeholder = self.model_text.embedding_placeholder

    def _create_model_audio(self):
        print('[launch-multi] create audio model')
        self.model_audio = Encoder_Audio(
            batch_size=self.batch_size,
            encoder_size=self.encoder_size_audio,
            num_layer=self.num_layers_audio,
            hidden_dim=self.hidden_dim_audio,
            lr=self.lr,
            dr=self.dr_audio
        )
        self.model_audio._create_placeholders()
        self.model_audio._create_gru_model()
        self.model_audio._add_prosody()
        self.model_audio._create_output_layers_for_multi()

    def _create_model_text(self):
        print('[launch-multi] create text model')
        self.model_text = Encoder_Text(
            batch_size=self.batch_size,
            dic_size=self.dic_size,
            encoder_size=self.encoder_size_text,
            num_layer=self.num_layers_text,
            hidden_dim=self.hidden_dim_text,
            lr=self.lr,
            dr=self.dr_text
        )

        self.model_text._create_placeholders()
        self.model_text._create_embedding()
        self.model_text._use_external_embedding()
        self.model_text._create_gru_model()
        self.model_text._create_output_layers_for_multi()

    def _create_output_layers(self):
        print('[launch-multi] create output projection layer from (audio_final_dim/2) + (text_final_dim/2)')

        with tf.name_scope('multi_output_layer') as scope:
            print("dim1",self.model_audio.final_encoder_dimension)
            print("dim2",self.model_text.final_encoder_dimension)
            self.M = tf.Variable(tf.random_uniform(
                [int(self.model_audio.final_encoder_dimension / 2) + int(self.model_text.final_encoder_dimension / 2),
                 N_CATEGORY],
                minval=-0.25,
                maxval=0.25,
                dtype=tf.float32,
                seed=None),
                                 trainable=True,
                                 name="similarity_matrix")

            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                 trainable=True,
                                 name="output_bias")

            self.final_encoder = tf.concat([self.model_audio.batch_pred, self.model_text.batch_pred], axis=1)
            print("audio batch_pred",self.model_audio.batch_pred)
            print("text batch_pred", self.model_text.batch_pred)

            # e * M + b
            self.batch_pred = tf.matmul(self.final_encoder, self.M) + self.b

        with tf.name_scope('loss') as scope:
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.batch_pred, labels=self.y_labels)
            self.loss = tf.reduce_mean(self.batch_loss)

    def _create_optimizer(self):
        print('[launch-multi] create optimizer')

        with tf.name_scope('multi_optimizer') as scope:
            opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
            gvs = opt_func.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(t=grad, clip_value_min=-10, clip_value_max=10), var) for grad, var in gvs]
            self.optimizer = opt_func.apply_gradients(grads_and_vars=capped_gvs, global_step=self.global_step)

    def _create_summary(self):
        print('[launch-multi] create summary')

        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_model_audio()
        self._create_model_text()
        self._create_placeholders()
        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()


def luong_attention(batch_size, target, condition, batch_seq, max_len, hidden_dim):
    # same dim [batch, max_seq, embed]
    batch_seq_embed_target = tf.reshape(target, [batch_size, max_len, hidden_dim])

    batch_embed_given = condition
    batch_seq_embed_given = tf.reshape(batch_embed_given, [batch_size, hidden_dim, 1])

    # calculate similarity
    dot = tf.matmul(batch_seq_embed_target, batch_seq_embed_given)
    dot = tf.squeeze(dot)

    mask = tf.sequence_mask(lengths=batch_seq, maxlen=max_len, dtype=tf.float32)
    mask_value = -tf.ones_like(mask) * tf.float32.max
    mask_value = tf.multiply(mask_value, (1 - mask))
    base = mask_value

    norm_dot = tf.nn.softmax(dot + base, axis=-1)

    # weighted sum by using similarity (normalized)
    target_mul_norm = tf.multiply(batch_seq_embed_target, tf.expand_dims(norm_dot, -1))
    weighted_sum = tf.reduce_sum(target_mul_norm, axis=1)

    return weighted_sum, norm_dot

class Encoder_MultiAttn:

    def __init__(self,
                 batch_size,
                 lr,
                 encoder_size_audio,  # for audio
                 num_layer_audio,
                 hidden_dim_audio,
                 dr_audio,
                 dic_size,  # for text
                 encoder_size_text,
                 num_layer_text,
                 hidden_dim_text,
                 dr_text
                 ):
        # for audio
        self.encoder_size_audio = encoder_size_audio
        self.num_layers_audio = num_layer_audio
        self.hidden_dim_audio = hidden_dim_audio
        self.dr_audio = dr_audio
        self.encoder_inputs_audio = []
        self.encoder_seq_length_audio = []

        # for text
        self.dic_size = dic_size
        self.encoder_size_text = encoder_size_text
        self.num_layers_text = num_layer_text
        self.hidden_dim_text = hidden_dim_text
        self.dr_text = dr_text

        self.encoder_inputs_text = []
        self.encoder_seq_length_text = []

        # common
        self.batch_size = batch_size
        self.lr = lr
        self.y_labels = []

        self.M = None
        self.b = None

        self.y = None
        self.optimizer = None

        self.batch_loss = None
        self.loss = 0
        self.batch_prob = None
        self.embed_dim = 300
        # for global counter
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        print('[launch-multi] placeholders')
        with tf.name_scope('multi_placeholder'):
            # for audio
            self.encoder_inputs_audio = self.model_audio.encoder_inputs  # [batch, time_step, audio]
            self.encoder_seq_audio = self.model_audio.encoder_seq
            self.encoder_prosody = self.model_audio.encoder_prosody
            self.dr_prob_audio = self.model_audio.dr_prob
            # for text
            self.encoder_inputs_text = self.model_text.encoder_inputs
            self.encoder_seq_text = self.model_text.encoder_seq
            self.dr_prob_text = self.model_text.dr_prob
            # common
            self.y_labels = tf.placeholder(tf.float32, shape=[self.batch_size, N_CATEGORY], name="label")

            # for using pre-trained embedding
            self.embedding_placeholder = self.model_text.embedding_placeholder

    def _create_model_audio(self):
        print('[launch-multi] create audio model')
        self.model_audio = Encoder_Audio(
            batch_size=self.batch_size,
            encoder_size=self.encoder_size_audio,
            num_layer=self.num_layers_audio,
            hidden_dim=self.hidden_dim_audio,
            lr=self.lr,
            dr=self.dr_audio
        )
        self.model_audio._create_placeholders()
        self.model_audio._create_gru_model()
        self.model_audio._add_prosody()
        # self.model_audio._create_output_layers_for_multi()

    def _create_model_text(self):
        print('[launch-multi] create text model')
        self.model_text = Encoder_Text(
            batch_size=self.batch_size,
            dic_size=self.dic_size,
            encoder_size=self.encoder_size_text,
            num_layer=self.num_layers_text,
            hidden_dim=self.hidden_dim_text,
            lr=self.lr,
            dr=self.dr_text
        )

        self.model_text._create_placeholders()
        self.model_text._create_embedding()
        self.model_text._use_external_embedding()
        self.model_text._create_gru_model()
        # self.model_text._create_output_layers_for_multi()

    def _create_attention_module(self):
        print('[launch-multi] create attention module')
        # project audio dimension_size to text dimension_size
        self.attnM = tf.Variable(
            tf.random_uniform([self.model_audio.final_encoder_dimension, self.model_text.final_encoder_dimension],
                              minval=-0.25,
                              maxval=0.25,
                              dtype=tf.float32,
                              seed=None),
            trainable=True,
            name="attn_projection_helper")

        self.attnb = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                 trainable=True,
                                 name="attn_bias")

        self.attn_audio_final_encoder = tf.matmul(self.model_audio.final_encoder, self.attnM) + self.attnb

        self.final_encoder, self.tmp_norm = luong_attention(
            batch_size=self.batch_size,
            target=self.model_text.outputs_en,
            condition=self.attn_audio_final_encoder,
            batch_seq=self.encoder_seq_text,
            max_len=self.model_text.encoder_size,
            hidden_dim=self.model_text.final_encoder_dimension
        )

    def _create_output_layers(self):
        print
        '[launch-multi] create output projection layer from (text_final_dim(==audio) + text_final_dim)'

        with tf.name_scope('multi_output_layer') as scope:
            self.final_encoder = tf.concat([self.final_encoder, self.attn_audio_final_encoder], axis=1)

            self.M = tf.Variable(tf.random_uniform(
                [(self.model_text.final_encoder_dimension) + (self.model_text.final_encoder_dimension), N_CATEGORY],
                minval=-0.25,
                maxval=0.25,
                dtype=tf.float32,
                seed=None),
                                 trainable=True,
                                 name="similarity_matrix")

            self.b = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                 trainable=True,
                                 name="output_bias")

            # e * M + b
            self.batch_pred = tf.matmul(self.final_encoder, self.M) + self.b

        with tf.name_scope('loss') as scope:
            self.batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.batch_pred, labels=self.y_labels)
            self.loss = tf.reduce_mean(self.batch_loss)

    def _create_optimizer(self):
        print
        '[launch-multi] create optimizer'

        with tf.name_scope('multi_optimizer') as scope:
            opt_func = tf.train.AdamOptimizer(learning_rate=self.lr)
            gvs = opt_func.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(t=grad, clip_value_min=-10, clip_value_max=10), var) for grad, var in gvs]
            self.optimizer = opt_func.apply_gradients(grads_and_vars=capped_gvs, global_step=self.global_step)

    def _create_summary(self):
        print('[launch-multi] create summary')
        with tf.name_scope('summary'):
            tf.summary.scalar('mean_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_model_audio()
        self._create_model_text()
        self._create_placeholders()
        self._create_attention_module()
        self._create_output_layers()
        self._create_optimizer()
        self._create_summary()