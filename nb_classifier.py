import json, re, nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

stop = set(stopwords.words("english"))
def clean(s:str):
    return " ".join(w for w in re.sub("[^a-z ]"," ",s.lower()).split() if w not in stop)

DATA = Path(__file__).with_suffix('').parent / "data" / "nb_dataset.json"
items = json.loads(DATA.read_text())
X_text, y = zip(*[(clean(d["text"]), d["label"]) for d in items])

vec = CountVectorizer()
clf = MultinomialNB().fit(vec.fit_transform(X_text), y)

def classify(txt:str) -> str:
    return clf.predict(vec.transform([clean(txt)]))[0]
