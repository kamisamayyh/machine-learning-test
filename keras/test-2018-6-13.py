import numpy as np
import os
import sys
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation, Dropout
from keras import optimizers

BASE_DIR = 'data'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
batch_size = 128

# first, build index mapping words in the embeddings set
# to their embedding vector
class Article:

    texts = None
    sequences = None
    labels = None
    word_index = None
    embedding_index = None
    labels_index = None
    embeddings_index = None

    def prepare(self):
        self.embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(self.embeddings_index))
        # Found 400000 word vectors.

        # second, prepare text samples and their labels
        print('Processing text dataset')

        self.texts = []  # list of text samples
        labels_index = {}  # dictionary mapping label name to numeric id
        self.labels = []  # list of label ids
        for name in sorted(os.listdir(TEXT_DATA_DIR)):
            path = os.path.join(TEXT_DATA_DIR, name)
            if os.path.isdir(path):
                label_id = len(labels_index)
                labels_index[name] = label_id
                for fname in sorted(os.listdir(path)):
                    if fname.isdigit():
                        fpath = os.path.join(path, fname)
                        if sys.version_info < (3,):
                            f = open(fpath)
                        else:
                            f = open(fpath, encoding='latin-1')
                        self.texts.append(f.read())
                        f.close()
                        self.labels.append(label_id)
        self.labels_index = labels_index
        print('Found %s texts.' % len(self.texts))
        # print(self.embeddings_index['hi'])

    def tokenize(self):
        # finally, vectorize the text samples into a 2D integer tensor
        tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(self.texts)
        self.sequences = tokenizer.texts_to_sequences(self.texts)

        self.word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))
        # Found 214909 unique tokens.

    def generate(self):
        data = pad_sequences(self.sequences, maxlen=MAX_SEQUENCE_LENGTH)
        labels = to_categorical(np.asarray(self.labels))

        print('Shape of data tensor:', data.shape)
        print('Shape of label tensor:', labels.shape)
        # ('Shape of data tensor:', (19997, 1000))
        # ('Shape of label tensor:', (19997, 20))

        # split the data into a training set and a validation set
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        x_train = data[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]

        x_train.shape
        # (15998, 1000)

        y_train.shape
        # (15998, 20)

        x_val = data[-nb_validation_samples:]
        y_val = labels[-nb_validation_samples:]
        return x_val, y_val, x_train, y_train
        print('Preparing embedding matrix.')

    def embedding_matrix(self):
        nb_words = min(MAX_NB_WORDS, len(self.word_index))
        # 20000
        embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))

        for word, i in self.word_index.items():
            if i > MAX_NB_WORDS:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        print(embedding_matrix.shape)
        # (20001, 100)
        return nb_words, embedding_matrix

    def model(self, nb_words, embedding_matrix):
        embedding_layer = Embedding(nb_words + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    trainable=False,
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    )

        print('Build model...')
        # sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        # embedded_sequences = embedding_layer()
        model = Sequential()
        model.add(embedding_layer)
        model.add(Dropout(0.2))
        model.add(LSTM(100, dropout=0.2))  # try using a GRU instead, for fun
        model.add(Dense(200))
        model.add(Activation('sigmoid'))
        model.add(Dense(len(self.labels_index), activation='softmax'))
        return model

    def train(self):
        self.prepare()
        self.tokenize()
        x_val, y_val, x_train, y_train = self.generate()

        nb_words, embedding_matrix = self.embedding_matrix()
        model = self.model(nb_words, embedding_matrix)
        sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('Train...')
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=5,
                  validation_data=(x_val, y_val))
        score, acc = model.evaluate(x_val, y_val,
                                    batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

article = Article()
article.train()