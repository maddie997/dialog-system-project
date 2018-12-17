import tensorflow as tf
from collections import namedtuple

# Model Parameters
tf.flags.DEFINE_integer(
  "vocab_size",
  91620,
  "The size of the vocabulary. Only change this if you changed the preprocessing")

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer("rnn_dim", 128, "Dimensionality of the RNN cell")
tf.flags.DEFINE_boolean("rnn_multilayer", False, "If True multilayer model")
tf.flags.DEFINE_integer("rnn_layers", 1, "Number of RNN/LSTM layers")
tf.flags.DEFINE_boolean("rnn_bidirectional", False, "If True bidirectional model")
#Note - Model cant be multilayer and bidirectional at the same time
tf.flags.DEFINE_boolean("use_dual_encoder", True, "If True use dual encoder else manhattan")
tf.flags.DEFINE_integer("max_context_len", 160, "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 80, "Truncate utterance to this length")

# Pre-trained embeddings
tf.flags.DEFINE_string("glove_path", "/home/mpandey/ubuntu_dialogue/cs230_project/dialog-system-project/lstm/chatbot-retrieval/data/glove.6B.100d.txt", "Path to pre-trained Glove vectors")
tf.flags.DEFINE_string("vocab_path", "/home/mpandey/ubuntu_dialogue/cs230_project/dialog-system-project/lstm/chatbot-retrieval/data/vocabulary.txt", "Path to vocabulary.txt file")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 8, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")
tf.flags.DEFINE_string("logdir", "logs", "log directory name")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  "HParams",
  [
    "batch_size",
    "embedding_dim",
    "eval_batch_size",
    "learning_rate",
    "max_context_len",
    "max_utterance_len",
    "optimizer",
    "rnn_dim",
    "rnn_layers",
    "rnn_multilayer",
    "rnn_bidirectional",
    "vocab_size",
    "glove_path",
    "vocab_path",
    "use_dual_encoder",
    "logdir"
  ])

def create_hparams():
  return HParams(
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    vocab_size=FLAGS.vocab_size,
    optimizer=FLAGS.optimizer,
    learning_rate=FLAGS.learning_rate,
    embedding_dim=FLAGS.embedding_dim,
    max_context_len=FLAGS.max_context_len,
    max_utterance_len=FLAGS.max_utterance_len,
    glove_path=FLAGS.glove_path,
    vocab_path=FLAGS.vocab_path,
    rnn_dim=FLAGS.rnn_dim,
    rnn_multilayer=FLAGS.rnn_multilayer,
    rnn_layers=FLAGS.rnn_layers,
    rnn_bidirectional=FLAGS.rnn_bidirectional,
    use_dual_encoder=FLAGS.use_dual_encoder,
    logdir=FLAGS.logdir
  )
