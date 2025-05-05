import random, nltk
from nltk.corpus import wordnet, gutenberg

def synonym(word:str)->str:
    syns = {l.name().replace('_',' ') for s in wordnet.synsets(word) for l in s.lemmas()}
    syns.discard(word)
    return random.choice(list(syns)) if syns else word

def random_quote()->str:
    txt = gutenberg.raw("bible-kjv.txt")
    return random.choice(nltk.sent_tokenize(txt)[:5000])
