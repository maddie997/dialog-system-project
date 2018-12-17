#Madhulima Pandey -- updates added for tensorflow 1.11 and above.
#Madhulima Pandey -- added support for multi-layer LSTM and bidirectional LSTM
#MAdhulima Pandey -- BERT transformation

import tensorflow as tf
import numpy as np
from models import helpers

FLAGS = tf.flags.FLAGS

def get_embeddings(hparams):
  if hparams.glove_path and hparams.vocab_path:
    tf.logging.info("Loading Glove embeddings...")
    vocab_array, vocab_dict = helpers.load_vocab(hparams.vocab_path)
    glove_vectors, glove_dict = helpers.load_glove_vectors(hparams.glove_path, vocab=set(vocab_array))
    initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors, hparams.embedding_dim)
  else:
    tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
    initializer = tf.random_uniform_initializer(-0.25, 0.25)

  return tf.get_variable(
    "word_embeddings",
    shape=[hparams.vocab_size, hparams.embedding_dim])
    #initializer=initializer)

def manhattan_distance(left, right):
  return tf.keras.backend.exp(-tf.keras.backend.sum(tf.keras.backend.abs(left - right), axis=1, keepdims=True))

def dual_encoder_model(
    hparams,
    mode,
    context,
    context_len,
    utterance,
    utterance_len,
    targets):
        
  #Verify shapes of context and utterance are (BS, 768)
  print("Context shape", context.get_shape())
  print("Utterance shape", utterance.get_shape())

  with tf.variable_scope("prediction") as vs:
    M = tf.get_variable("M", shape=[768, 768],
                        initializer=tf.truncated_normal_initializer())
    bias = tf.get_variable("b", shape=[768])
  
    # "Predict" a  response: c * M
    generated_response = tf.add(tf.matmul(context, M), bias)
    generated_response = tf.expand_dims(generated_response, 2)
    expanded_utterance = tf.expand_dims(utterance, 2)
    
    # Dot product between generated response and actual response
    # (c * M) * r
    logits = tf.matmul(generated_response, expanded_utterance, True)
    logits = tf.squeeze(logits, [2])
    
    # Apply sigmoid to convert logits to probabilities
    probs = tf.sigmoid(logits)
      
    if mode == tf.contrib.learn.ModeKeys.INFER:
      return probs, None
        
    # Calculate the binary cross-entropy loss
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(targets))

    # Mean loss across the batch of examples
    mean_loss = tf.reduce_mean(losses, name="mean_loss")
    return probs, mean_loss
