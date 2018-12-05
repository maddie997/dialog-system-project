import random
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Input, Dense
from keras import backend as K
from keras import layers
from keras.initializers import RandomUniform
from keras.optimizers import Adam
from keras.engine.topology import Layer
from numpy import asarray
import numpy as np
import time

MAX_SEQUENCE_LEN = 160
WORD_EMBEDDINGS_LEN = 50
UNITS = 100
DEBUG = True
BATCH_SIZE = 32

# Load the training data, generate a mix where 50% are correct and 50% are incorrect.
# the samples are then shuffled and then returned with their respective labels
def load_data():
    start = time.time()
    train_data = []
    labels = []
    df = pd.read_csv('../data/valid.csv')

    half = df.shape[0] / 2

    for index, row in df.iterrows():
        if index <= half:
            train_data.append( (row['Context'], row['Ground Truth Utterance'], 1))
        else:
            train_data.append( (row['Context'], row['Distractor_0'], 0))

    random.shuffle(train_data)

    for _, _, l in train_data:
        labels.append(l)

    end = time.time()

    if DEBUG:
        print '-> load_data() time: ', (end-start)

    return train_data, labels

# Tokenizes and pads the context and responses so that they can be the input of a RNN or LSTM
def tokenize_pad(data):
    start = time.time()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    vocab_size = len(tokenizer.word_index) + 1
    encoded_docs = tokenizer.texts_to_sequences(data)
    padded_docs = pad_sequences(encoded_docs, maxlen=MAX_SEQUENCE_LEN, padding='post')

    if DEBUG:
        print 'Loaded %d words' % vocab_size
        print 'Input shape: ', padded_docs.shape

    end = time.time()

    if DEBUG:
        print '-> tokenize_pad() time: ', (end - start)

    return vocab_size, padded_docs, tokenizer

# Loads Glove word embeddings
def load_pretrained_embeddings(tokenizer):
    start = time.time()
    embeddings_index = dict()
    f = open('../data/glove.6B.50d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        vector = asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector

    f.close()
    end = time.time()

    if DEBUG:
        print '-> load_pretrained_embeddings() time: ', (end - start)

    return embeddings_index

# Computes the embeddings matrix used for the embedding layer
# in the model
def compute_embeddings_matrix(embeddings_index, vocab_size, tokenizer):
    start = time.time()
    embeddings_matrix = np.zeros((vocab_size, WORD_EMBEDDINGS_LEN))
    i = 0

    for word in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
            i = i + 1

    end = time.time()

    if DEBUG:
        print '-> compute_embeddings_matrix() time: ', (end - start)

    return embeddings_matrix

# Loads data and preprocess it so it can be used by the model
def prepare_data():
    start = time.time()
    train_data, labels = load_data()
    all_data = []
    context = []
    response = []
    for c, r, _ in train_data:
        all_data.append(c)
        all_data.append(r)
        context.append(c)
        response.append(r)

    # All the data need to be part of the embeddings matrix used for the embeddings layer
    vocab_size, padded_data, tokenizer = tokenize_pad(all_data)
    embeddings_index = load_pretrained_embeddings(tokenizer)
    embeddings_matrix = compute_embeddings_matrix(embeddings_index, vocab_size, tokenizer)

    _, tokenized_context, _ = tokenize_pad(context)
    _, tokenized_response, _ = tokenize_pad(response)

    end = time.time()

    if DEBUG:
        print '-> prepared_data() time: ', (end - start)

    return embeddings_matrix, vocab_size, tokenized_context, tokenized_response, labels

def distance(x, y):
    return K.exp(-K.sum(K.abs(x-y), axis=1, keepdims=True))

def distance_output(input_shape):
    return (input_shape[0][0], 1)

def create_model(embeddings_matrix, vocab_size, context, response, labels):

    context_input = Input(shape=(MAX_SEQUENCE_LEN, ), dtype='float32')
    response_input = Input(shape=(MAX_SEQUENCE_LEN, ), dtype='float32')

    init = RandomUniform(minval=-0.01, maxval=0.01)
    embeddings_layer = Embedding(vocab_size, WORD_EMBEDDINGS_LEN, weights=[embeddings_matrix], input_length=MAX_SEQUENCE_LEN, trainable=True)
    rnn_layer = layers.LSTM(units=UNITS, kernel_initializer=init, dropout=0.2)

    c_x = embeddings_layer(context_input)
    r_x = embeddings_layer(response_input)

    c_x = rnn_layer(c_x)
    r_x = rnn_layer(r_x)

    # This layer needs to be fixed, multiplication by
    # the context is missing
    preds = CustomLayer(output_dim=UNITS)([c_x, r_x])
    preds = layers.Dot(axes=-1)([preds, c_x])

    preds = Dense(1, activation='sigmoid')(preds)

    siamese_model = Model(inputs=[context_input, response_input], outputs=preds)
    op = Adam(lr=0.0001, clipvalue=10.0)
    siamese_model.compile(loss='binary_crossentropy', optimizer=op, metrics=['acc', 'binary_accuracy'])
    siamese_model.summary()
    siamese_model.fit([context, response], labels, batch_size=BATCH_SIZE, epochs=100, validation_split=0.1)


class CustomLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CustomLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        _, response = input_shape
        self.M = self.add_weight(name='M',
                                      shape=(response[1], self.output_dim),
                                      initializer='uniform',
                                      dtype='float32',
                                      trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(self.output_dim,),
                                 initializer='uniform',
                                 dtype='float32',
                                 trainable=True)

        super(CustomLayer, self).build(input_shape)

    def call(self, x, mask=None):
        context, response = x
        c_prime = K.dot(response, self.M) + self.b
        return c_prime


    def get_output_shape_for(self, input_shape):
        context, response = input_shape
        return (response[0], self.output_dim)


def main():
    embeddings_matrix, vocab_size, context, response, labels = prepare_data();
    create_model(embeddings_matrix, vocab_size, context, response, labels)


if __name__ == '__main__':
    main()