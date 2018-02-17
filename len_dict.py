import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from collections import Counter
import re
import nltk
from nltk import wordnet, WordNetLemmatizer


def get_wordnet_pos(treebank_tag):
    my_switch = {'J': wordnet.wordnet.ADJ,
                'V': wordnet.wordnet.VERB,
                'N': wordnet.wordnet.NOUN,
                'R': wordnet.wordnet.ADV}
    for key, item in my_switch.items():
        if treebank_tag.startswith(key):
            return item
    return wordnet.wordnet.NOUN
	

def simple_lemmatizer(word):
    lemmatizer = WordNetLemmatizer()
    word, tag = nltk.pos_tag([word])[0]
    tag = get_wordnet_pos(tag)
    return lemmatizer.lemmatize(word, tag)


train_df = pd.read_csv("train.csv")
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
regex = re.compile("[^a-z' ]")
regex1 = re.compile("[\n]")
tmp = tknzr.tokenize(regex.sub('', regex1.sub(' ', train_df.iloc[0]['comment_text'].lower())))
for i in range(1, len(train_df)):
    tmp.extend(tknzr.tokenize(regex.sub('', regex1.sub(' ', train_df.iloc[i]['comment_text'].lower()))))
for i in range(len(tmp)):
    tmp[i] = simple_lemmatizer(tmp[i])
words_counter = Counter(tmp)
