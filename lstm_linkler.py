from __future__ import print_function
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import nltk.data
from nltk.tokenize import WordPunctTokenizer
import numpy as np
from pathlib import Path
import pickle
import itertools


from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Embedding, Flatten, Dropout
from keras.optimizers import RMSprop, adam
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import L1L2
from keras.callbacks import ModelCheckpoint

def load_tokenized_text(path):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open(path)
    data = fp.read()

    return tokenizer.tokenize(data)


class linkler:
    def __init__(self):
        self._word_index = None
        self._index_word = None
        self._words = None
        self._corpus = ''
        self._corpus_vector = None
        self._model = None
        self._maxlen = 5
        self._stride = 1
        self._chunks = None
        self._nextword = None
        self._embedding_size = 20
        self._X = None
        self._Y = None
        self._epochs = 400
        self._batchsize = 256
        self._modelsavepath = r'C:\Users\eander2\Desktop\Lincler/linkler_word.h5'
        self._word_dict_save = r'C:\Users\eander2\Desktop\Lincler/word_index_dict.pkl'
        self._index_dict_save = r'C:\Users\eander2\Desktop\Lincler/index_word_dict.pkl'
        self._use_chars = False
        self._generator_separator = ' '

        self.load_model()


    def load_model(self):
        '''Load model parameters from file as well as word indices found in corpus'''
        if Path(self._modelsavepath).is_file():
            self._model = load_model(self._modelsavepath)
        if Path(self._word_dict_save).is_file():
            self._word_index = pickle.load(open(self._word_dict_save, 'rb'))
        if Path(self._index_dict_save).is_file():
            self._index_word = pickle.load(open(self._index_dict_save, 'rb'))

    def get_sentence_list(self):
        '''Tokenize the input text using nltk'''
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return tokenizer.tokenize(self._corpus)

    def get_word_tokens(self):
        '''Tokenize the corpus'''
        tokenizer = WordPunctTokenizer()
        return tokenizer.tokenize(self._corpus)

    def load_files_to_corpus(self, source_path):
        '''Load text corpus from disk.  source_path can be file or dir'''
        path = Path(source_path)

        if path.is_file():
            txt = open(path).read().replace('\n', ' ').lower()
            txt = ' '.join(txt.split())
            self._corpus += txt
            return

        files = path.glob("**/*")

        for f in files:
            if f.is_file():
                txt = open(f, encoding="utf8").read().replace('\n', ' ').lower()
                txt = ' '.join(txt.split())
                self._corpus += txt

    def get_encoded_vector(self):
        #Create traning vectors
        x = np.zeros((len(self._chunks), self._maxlen, len(self._words)), dtype=np.bool)
        y = np.zeros((len(self._chunks), len(self._words)), dtype=np.bool)
        for i, sentence in enumerate(self._chunks):
            for t, idx in enumerate(sentence):
                x[i, t, idx] = 1
            y[i, self._nextword[i]] = 1

        self._X = x
        self._Y = y

    def process_corpus_by_sentence(self):
        words = self.get_word_tokens()
        sentences = self.get_sentence_list()

        maxlen = 0
        for s in sentences:
            if len(s) > maxlen:
                maxlen = len(s)

        self._word_index = dict((word, index) for index, word in enumerate(self._words))
        self._index_word = dict((index, word) for index, word in enumerate(self._words))

        pickle.dump(self._word_index, open(self._word_dict_save, 'wb'))
        pickle.dump(self._index_word, open(self._index_dict_save, 'wb'))
        sentences = pad_sequences(sentences, dtype='str', padding='post', truncating='post', value='PADDED')


    def process_corpus(self):
        if not self._use_chars:
            words = self.get_word_tokens()
            # words = text_to_word_sequence(self._corpus)
            word_freq = nltk.FreqDist(words)
            self._words = [i[0] for i in word_freq.most_common(int(len(words)*0.8) - 1)]
            self._words.append("UNKNOWN_TEXT")
            self._corpus = ' '.join(words)
        else:
            self._words = list(set(self._corpus))
            lenwords = len(self._words)
            print(f"The vocabulary size is {lenwords}.")
            words = self._corpus

        print(self._words)

        # self.process_corpus_by_sentence()
        #self._corpus = ' '.join(words)
        #words = self._corpus

        #self._words = sorted(list(set(words)))

        self._word_index = dict((word, index) for index, word in enumerate(self._words))
        self._index_word = dict((index, word) for index, word in enumerate(self._words))

        pickle.dump(self._word_index, open(self._word_dict_save, 'wb'))
        pickle.dump(self._index_word, open(self._index_dict_save, 'wb'))

        print(self._word_index)

        # Encode corpus

        corpus_vector = []
        for word in words:
            if word in self._word_index.keys():
                corpus_vector.append(self._word_index[word])
            else:
                corpus_vector.append(self._word_index["UNKNOWN_TEXT"])
        # corpus_vector = [self._word_index[word] for word in words]
        print(corpus_vector)

        self._chunks = []
        self._nextword = []
        for i in range(0, len(corpus_vector) - self._maxlen, self._stride):
            self._chunks.append(corpus_vector[i: i + self._maxlen])
            self._nextword.append(corpus_vector[i + self._maxlen])

        max_samples = int(np.floor(len(self._chunks) / self._batchsize)) * self._batchsize
        print(f"new sample size is {max_samples}")
        self._chunks = np.asarray(self._chunks[0:max_samples])
        self._nextword = np.asarray(self._nextword[0:max_samples])
        permutation_matrix = np.random.permutation(len(self._chunks))

        self._chunks = np.asarray(self._chunks)[permutation_matrix]
        self._nextword = np.asarray(self._nextword)[permutation_matrix]

        # Not needed if using keras embedding layer
        self.get_encoded_vector()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(len(self._words), self._embedding_size, input_length=self._maxlen))
        model.add(LSTM(512, return_sequences=True, stateful=False))
        model.add(Dropout(0.2))
        model.add(LSTM(512, stateful=False))
        model.add(Dropout(0.2))
        model.add(Dense(len(self._words), activation='softmax'))

        optimizer = RMSprop(lr=0.001)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self._model = model

    def get_model_sequence(self, sequence_length = 800):
        '''Generate a modeled sequence from a random sequence in the corpus.'''
        starting_seed = np.random.random_integers(0, len(self._chunks) - self._batchsize, size=1)
        print(f"starting seed is {starting_seed}")
        seed_sequence = self._chunks[starting_seed]
        print(len(seed_sequence))
        print(seed_sequence.shape)
        seed = ' '.join(self._index_word[i] for i in seed_sequence[-1])
        print(seed)

        generated_sequence = []

        for i in range(sequence_length):
            #print(f"seed sequence is: {seed_sequence}")
            prediction = self._model.predict(seed_sequence)[0]
            #print(f"prediction is: {prediction}")

            #idx = np.argmax(prediction)
            idx = self.sample_softmax(prediction, temperature=1.0)
            generated_sequence.append(idx)

            seed_sequence[0][:-1] = seed_sequence[0][1:]
            seed_sequence[0][len(seed_sequence[0])-1] = idx
            seed = seed[1:]
            seed += self._index_word[idx]
            #seed_sequence[:-1] = seed_sequence[1:]
            #seed_sequence[-1][:-1] = seed_sequence[-1][1:]
            #seed_sequence[-1][-1] = idx

            #seed_sequence = self.get_sequence_to_model(seed)
            #print(seed)



        print(generated_sequence)

        sentence = ' '.join(self._index_word[i] for i in generated_sequence)

        print(f"The generated sequence is: '{sentence}'")

    def sample_softmax(self, a, temperature=1.0):
        a = np.array(a) ** (1.0 / temperature)
        p_sum = a.sum()
        sample_temp = a / (p_sum + 1e-6)
        return np.argmax(np.random.multinomial(1, sample_temp, 1))

    def get_sequence_to_model(self, characters):
        x = np.zeros((1, self._maxlen, len(self._words)), dtype=np.bool)


        for t, idx in enumerate(characters):
            x[0, t, self._word_index[idx]] = 1

        return x


    def train_model(self):

        self._model.fit(self._chunks, self._Y,
                  batch_size=self._batchsize,
                  epochs=self._epochs,
                        validation_split=0.0)


        self._model.save(self._modelsavepath)





if __name__ == "__main__":
    txtpath = r'C:\Users\eander2\Desktop\Lincler\training'

    g = linkler()

    g.load_files_to_corpus(txtpath)
    g.process_corpus()
    g.build_model()
    g.train_model()
    g.get_model_sequence()




