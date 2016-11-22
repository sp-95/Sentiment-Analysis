import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from glob import glob
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


    def __get_pos(self, tag):
        if tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('R'):
            return wordnet.ADV
        return None


    def __combine_adv(self, x, y):
        return (' '.join([x[0], y[0]]), y[1]) if x[1] == wordnet.ADV else x

    def bag_of_words(self, review):
        if not review:
            raise Exception('Empty review')

        words = nltk.word_tokenize(review)

        # Parts of Speech tagging
        tagged = nltk.pos_tag(words)
        words = [(w[0], self.__get_pos(w[1])) for w in tagged]

        # Convert to base words
        lemmatizer = WordNetLemmatizer()
        lemmas = [(lemmatizer.lemmatize(*w), w[1]) if w[1] else w for w in words]

        # Combine adverbs with the next words
        converted_words = np.array(lemmas)
        while wordnet.ADV in converted_words[:,1]:
            indices = list(np.where(converted_words==wordnet.ADV)[0])
            for i, e in enumerate(indices):
                if i is not len(indices)-1 and e+1 == indices[i+1]:
                    del(indices[i+1])
            converted_words = np.array(list(map(self.__combine_adv, np.delete(converted_words[:len(converted_words)-1], np.add(indices, 1), axis=0), np.delete(converted_words[1:], np.add(indices, 1), axis=0))))
            if len(converted_words) > 1:
                if converted_words[-2, 1] != wordnet.ADV:
                    np.vstack((converted_words, converted_words[-1]))

        # Remove words that are not verb/adjective/adverb
        filtered_words = [w[0] for w in converted_words if w[1]]
        
        # Remove stop words
        r = re.compile("^'")
        stop_words = list(filter(r.match, filtered_words))
        stop_words += stopwords.words("english")
        filtered_words = [w for w in filtered_words if w not in stop_words]

        # Replace "n't" with 'not'
        result = [x.replace("n't", 'not') if "n't" in x else x for x in filtered_words]

        return list(set((result)))


    def get_positive_data(self):
        return random.choice(self.__pos)


    def get_negative_data(self):
        return random.choice(self.__neg)


    def main(self):
        print('\nPositive Review:\n', self.__pos[1])
        print('Filtered Words:\n', self.bag_of_words(self.__pos[1]))
        print()
        print()
        print()
        print('\nNegative Review:\n', self.__neg[1])
        print('Filtered Words:\n', self.bag_of_words(self.__neg[1]))


if __name__ == '__main__':
    myObj = Sentiment()
    myObj.main()
