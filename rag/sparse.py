"""Corpus-fitted BM25 vectors; vocabulary is persisted with each snapshot."""
import math
import re
from collections import Counter

def tokens(text):
    return re.findall(r'[a-z0-9]+(?:[-_][a-z0-9]+)*', text.lower())

class BM25:
    def __init__(self, texts):
        documents = [tokens(t) for t in texts]
        self.vocabulary = {t: i for i, t in enumerate(sorted({t for d in documents for t in d}))}
        self.avg_length = sum(map(len, documents)) / max(len(documents), 1)
        df = Counter(t for d in documents for t in set(d))
        self.idf = {t: math.log(1 + (len(documents) - f + .5) / (f + .5)) for t, f in df.items()}

    def document(self, text):
        terms = tokens(text)
        counts = Counter(terms)
        weights = {self.vocabulary[t]: f * 2.2 / (f + 1.2 * (.25 + .75 * len(terms) / self.avg_length)) for t, f in counts.items() if t in self.vocabulary}
        return {'indices': list(weights), 'values': list(weights.values())}

    def query(self, text):
        terms = sorted(set(tokens(text)) & self.vocabulary.keys())
        return {'indices': [self.vocabulary[t] for t in terms], 'values': [self.idf[t] for t in terms]}
