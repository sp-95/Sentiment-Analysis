import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from glob import glob
import os.path
import pickle
import random
import numpy as np
import re


class Sentiment:
    def __init__(self, pos=None, neg=None):
        if not pos:
            self.__pos = [open(f).read() for f in glob('review_polarity/txt_sentoken/pos/*.txt')]
        else:
            self.__pos = pos
        if not neg:
            self.__neg = [open(f).read() for f in glob('review_polarity/txt_sentoken/neg/*.txt')]
        else:
            self.__neg = neg

        if os.path.isfile('classifier.pickle'):
            # Load the features
            with open('classifier.pickle', 'rb') as f:
                self.__classifier = pickle.load(f)
        else:
            # Train a data set
            self.__classifier = nltk.NaiveBayesClassifier.train(self.__train_data())

            # Cache the features for faster predictions
            with open('classifier.pickle', 'wb') as f:
                pickle.dump(self.__classifier, f)

    def __get_tag(self, tag):
        if tag.startswith('R'):
            return 'r'
        elif tag.startswith('V'):
            return 'v'
        elif tag.startswith('A'):
            return 'a'
        else:
            return 'n'

    def __process_chunks(self, chunk):
        tagged = nltk.pos_tag(chunk)
        chunks = [(w[0], self.__get_tag(w[1])) for w in tagged]

        # Convert to base words
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(*w) for w in chunks]

        # Remove stop words
        stop_words = stopwords.words("english")
        tagged = nltk.pos_tag(stop_words)
        stop_words = stop_words[:30] + [x[0] for x in tagged[30:130] if not re.match('[JIR]', x[1])] + stop_words[130:] + ['s']
        words = [w for w in lemmas if w not in stop_words]

        # Replace "n't" with 'not'
        result = [w.replace("n't", 'not') for w in words]
        return ' '.join(result)

    def __preprocess(self, sentence):
        # Tokenize into words
        words = nltk.word_tokenize(sentence)

        # Remove non-alphnumeric characters
        words = [re.sub("^(?!')[\W_]+|(?!')[\W_]+$", '', i) for i in words]
        words = list(filter(None, words))

        # Empty sentence
        if not words:
            return []

        # Parts of Speech tagging
        tagged = nltk.pos_tag(words)
        # print(tagged)

        # Chunk required group of words
        grammer = r'''Chunk: {<RB.?>+<VB.?>?(<DT>?<JJ.?>)+(<IN><PRP.?>?|<IN>?)(<DT>?<JJ.?>)*<NN.?>}
                             {<RB.?>+<VB.?>?(<DT>?<JJ.?>)*(<IN><PRP.?>?|<IN>?)(<DT>?<JJ.?>)+<NN.?>}
                             {<VB.?>?(<DT>?<JJ.?>)+(<IN><PRP.?>?|<IN>?)(<DT>?<JJ.?>)*<NN.?>}
                             {<VB.?>?(<DT>?<JJ.?>)*(<IN><PRP.?>?|<IN>?)(<DT>?<JJ.?>)+<NN.?>}
                             {<RB.?>+<VB.?>?(<DT>?<JJ.?>)*(<IN><PRP.?>?|<IN>?)(<DT>?<JJ.?>)+}
                             {<JJ.?>}'''
        chunker = nltk.RegexpParser(grammer)
        chunked = chunker.parse(tagged)
        chunks = [np.array(subtree)[:,0] for subtree in chunked.subtrees() if subtree.label() == 'Chunk']

        return list(set([self.__process_chunks(chunk) for chunk in chunks]))

    def bag_of_words(self, review):
        # Tokenize into sentences
        sentences = nltk.sent_tokenize(review)
        return [x for sentence in sentences for x in self.__preprocess(sentence)]

    def __make_feature(self, word_list):
        return dict([(word, True) for word in word_list])

    def __train_data(self, pos=None, neg=None):
        if pos == None:
            p = self.__pos
        else:
            p = pos
        if neg == None:
            n = self.__neg
        else:
            n = neg
        return [(self.__make_feature(self.bag_of_words(review)), 'positive') for review in p] + [(self.__make_feature(self.bag_of_words(review)), 'negative') for review in n]

    def train(self, pos=None, neg=None, save=False):
        if pos == None and neg != None:
            raise Exception('pos is None and neg is not None')
        if pos != None and neg == None:
            raise Exception('pos is not None and neg is None')
        if pos != None and neg != None:
            if type(pos) != list and type(neg) != list:
                raise Exception('pos and neg aren\'t of type list')
            if len(pos) != len(neg):
                raise Exception('Unequal number of positive and negative reviews')

        # Train a data set
        self.__classifier = nltk.NaiveBayesClassifier.train(self.__train_data(pos, neg))

        if save:
            # Cache the features for faster predictions
            with open('classifier.pickle', 'wb') as f:
                pickle.dump(self.__classifier, f)

    def get_positive_data(self):
        return random.choice(self.__pos)

    def get_negative_data(self):
        return random.choice(self.__neg)

    def predict(self, sentences):
        if type(sentences) is str:
            return self.__classifier.classify(self.__make_feature(self.bag_of_words(sentences)))
        else:
            return [self.__classifier.classify(self.__make_feature(self.bag_of_words(sentence))) for sentence in sentences]

    def main(self):
        # Train data sets
        # self.train(pos=self.__pos, neg=self.__neg, save=True)

        # Test data sets
        # print(self.predict(self.get_positive_data()))
        # print(self.predict(self.get_negative_data()))
        # print(self.predict(self.get_positive_data()))
        # print(self.predict(self.get_negative_data()))
        # print(self.predict(self.get_positive_data()))
        # print(self.predict(self.get_negative_data()))
        # print(self.predict(self.get_positive_data()))
        # print(self.predict(self.get_negative_data()))

        # pos = self.__pos[7]
        # print('Filtered Words:\n', self.bag_of_words(pos))
        # print('\nPositive Review:\n', pos)
        # neg = self.__neg[39]
        # print('Filtered Words:\n', self.bag_of_words(neg))
        # print('\nNegative Review:\n', neg)
        # print('\nPrediction: ', self.predict([pos, neg]))

        for i in range(40,50):
            print(i, self.bag_of_words(self.__pos[i]))
        for i in range(40,50):
            print(i, self.bag_of_words(self.__neg[i]))
            
        # print(self.bag_of_words('not that bad'))
        # print(self.bag_of_words('very bad'))
        # print(self.bag_of_words('not very bad'))
        # print(self.bag_of_words('boring'))

if __name__ == '__main__':
    myObj = Sentiment()
    myObj.main()