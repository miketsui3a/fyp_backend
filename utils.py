from nltk.stem.porter import PorterStemmer
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.RegexpTokenizer(r"\w+").tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())

def removeStopword(words):
    return [word for word in words if not word in stopwords.words()]



def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0
    return bag
