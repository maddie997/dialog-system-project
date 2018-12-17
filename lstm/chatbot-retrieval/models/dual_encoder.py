#Madhulima Pandey -- updates added for tensorflow 1.11 and above.
#Madhulima Pandey -- added support for multi-layer LSTM and bidirectional LSTM


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

  # Initialize embedidngs randomly or with pre-trained vectors if available
  embeddings_W = get_embeddings(hparams)

  # Embed the context and the utterance
  context_embedded = tf.nn.embedding_lookup(
      embeddings_W, context, name="embed_context")
  utterance_embedded = tf.nn.embedding_lookup(
      embeddings_W, utterance, name="embed_utterance")

  if(hparams.rnn_bidirectional == True and hparams.rnn_multilayer == True):
    sys.error("Network can't be both bidirectional and multilayer at same time")

  # Build the RNN
  with tf.variable_scope("rnn") as vs:

    if(hparams.rnn_bidirectional == False and hparams.rnn_multilayer == False):
      #Start single layer
      cell = tf.nn.rnn_cell.LSTMCell(
        hparams.rnn_dim,
        forget_bias=2.0,
        use_peepholes=True,
        state_is_tuple=True)

      rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
        cell,
        tf.concat([context_embedded, utterance_embedded], 0),
        sequence_length=tf.concat([context_len, utterance_len], 0),
        dtype=tf.float32)

      encoding_context, encoding_utterance = tf.split(rnn_states.h, 2, 0)
      #End single layer

    if(hparams.rnn_multilayer):
      #Start multi-layer
      def lstm_cell(layer_num):
        with tf.variable_scope("rnn%d"%(layer_num)) as vs:
          # We use an LSTM Cell
          cell = tf.nn.rnn_cell.LSTMCell(
            hparams.rnn_dim,
            forget_bias=2.0,
            use_peepholes=True,
            state_is_tuple=True)
          return cell
      
      cell_multi = tf.contrib.rnn.MultiRNNCell([lstm_cell(i) for i in range(hparams.rnn_layers)])
      
      rnn_outputs, rnn_states_multi = tf.nn.dynamic_rnn(
        cell_multi,
        tf.concat([context_embedded, utterance_embedded], 0),
        sequence_length=tf.concat([context_len, utterance_len], 0),
        dtype=tf.float32)

      rnn_states = rnn_states_multi[hparams.rnn_layers-1]
      encoding_context, encoding_utterance = tf.split(rnn_states.h, 2, 0)
      #End multi-layer

    if(hparams.rnn_bidirectional):      
      #Start Bidi
      cell_fw = tf.nn.rnn_cell.LSTMCell(
        hparams.rnn_dim,
        forget_bias=2.0,
        use_peepholes=True,
        state_is_tuple=True)
      cell_bw = tf.nn.rnn_cell.LSTMCell(
        hparams.rnn_dim,
        forget_bias=2.0,
        use_peepholes=True,
        state_is_tuple=True)

      # Run the utterance and context through the RNN
      rnn_outputs, rnn_states_bidi = tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        tf.concat([context_embedded, utterance_embedded], 0),
        sequence_length=tf.concat([context_len, utterance_len], 0),
        dtype=tf.float32)

      (state_fw, state_bw) = rnn_states_bidi
      state_fw_enc_context, state_fw_enc_utterance = tf.split(state_fw.h, 2, 0)
      state_bw_enc_context, state_bw_enc_utterance = tf.split(state_bw.h, 2, 0)
      encoding_context   = tf.concat([state_fw_enc_context, state_bw_enc_context], 1)
      encoding_utterance = tf.concat([state_fw_enc_utterance, state_bw_enc_utterance], 1)
      #End Bidi

  use_dual_encoder = hparams.use_dual_encoder

  if use_dual_encoder:
    #Begin dual_encoder
    with tf.variable_scope("prediction") as vs:
      
      if(hparams.rnn_bidirectional == False):      
        #Single layer and multilayer LSTM
        M = tf.get_variable("M",
                            shape=[hparams.rnn_dim, hparams.rnn_dim],
                            initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable("b",
                            shape=[hparams.rnn_dim])
        #End single layer and multi-layer
      else:
        #Start Bidi
        M = tf.get_variable("M",
                            shape=[2*hparams.rnn_dim, 2*hparams.rnn_dim],
                            initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable("b",
                            shape=[2*hparams.rnn_dim])
        #End Bidi
        
      # "Predict" a  response: c * M
      generated_response = tf.add(tf.matmul(encoding_context, M), bias)
      generated_response = tf.expand_dims(generated_response, 2)
      encoding_utterance = tf.expand_dims(encoding_utterance, 2)
        
      # Dot product between generated response and actual response
      # (c * M) * r
      logits = tf.matmul(generated_response, encoding_utterance, True)
      logits = tf.squeeze(logits, [2])
    
      # Apply sigmoid to convert logits to probabilities
      probs = tf.sigmoid(logits)
      
      if mode == tf.contrib.learn.ModeKeys.INFER:
        return probs, None
        
      # Calculate the binary cross-entropy loss
      losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(targets))
      #End dual encoder
  else: #use_dual_encoder == False:
    distance = manhattan_distance(encoding_context, encoding_utterance)
    probs = distance
    losses = tf.losses.mean_squared_error(labels=tf.to_float(targets), predictions=distance)

  # Mean loss across the batch of examples
  mean_loss = tf.reduce_mean(losses, name="mean_loss")
  return probs, mean_loss
