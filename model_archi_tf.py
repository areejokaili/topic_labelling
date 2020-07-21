# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:29:54 2019

@author: areej
"""

import tensorflow as tf      
import warnings
warnings.filterwarnings("ignore")

class Encoder_bigru_attention(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, x_tokenizer):
    super(Encoder_bigru_attention, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    
    
    
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.forward_gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.back_gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   go_backwards=True, 
                                   recurrent_initializer='glorot_uniform')
    

  def call(self, x, hidden):
    x = self.embedding(x)
    # forward pass
    output_f, state_f = self.forward_gru(x, initial_state = hidden)
    # backward pass
    output_b, state_b = self.back_gru(x, initial_state = hidden)
    state = tf.concat([state_f, state_b], axis=-1)
    
    #print("outputs shape:", output_f.shape)
    output = tf.concat([output_f, output_b], axis= 2)
    #print("after concat", output.shape)
    
    
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))



class Decoder_bigru_attention(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, y_tokenizer):
    super(Decoder_bigru_attention, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units

    
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units*2,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    
    self.fc = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)
    

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights