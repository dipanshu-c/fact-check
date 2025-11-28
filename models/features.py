# models/features.py
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class EngineeredFeatures(BaseEstimator, TransformerMixin):
    """Compute a fixed set of engineered numeric features from (headline, content)."""
    def __init__(self):
        self.controversial = ['conspiracy', 'hoax', 'cover-up', 'fake']
        self.emotional = ['shocking', 'unbelievable', 'outrageous', 'disgusting', 'amazing']
        self.expert = ['expert', 'scientist', 'doctor', 'professor', 'analyst']
        self.source_kw = ['study', 'research', 'university', 'confirmed', 'verified']
        self.source_phrases = ['according to', 'reports say', 'sources claim', 'authorities say']
        self.passive_indicators = ['was', 'were', 'been', 'being']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for headline, content in X:
            if headline is None: headline = ''
            if content is None: content = ''
            text = (headline + " " + content).lower()
            words = text.split()
            nwords = max(len(words), 1)

            sensationalism = (text.count('!') + text.count('?')) / nwords
            source_cred = sum(1 for kw in self.source_kw if kw in text) / max(len(self.source_kw), 1)
            numbers = len(re.findall(r'\d+', text))
            dates = len(re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text))
            specificity = (numbers + dates) / nwords
            emotional = sum(1 for w in self.emotional if w in text) / max(len(self.emotional), 1)
            passive = sum(1 for w in self.passive_indicators if w in words) / nwords
            quote_factor = min((text.count('"') + text.count("'")) / 4, 1.0)
            expert_ref = sum(1 for kw in self.expert if kw in text) / max(len(self.expert), 1)
            controversy = sum(1 for kw in self.controversial if kw in text) / max(len(self.controversial), 1)
            multi_source = sum(1 for kw in self.source_phrases if kw in text) / max(len(self.source_phrases), 1)
            headline_words = set(headline.lower().split())
            content_words = set(content.lower().split())
            coherence = len(headline_words.intersection(content_words)) / max(len(headline_words), 1)

            rows.append([
                min(sensationalism, 1.0),
                min(source_cred, 1.0),
                min(specificity, 1.0),
                min(emotional, 1.0),
                min(passive, 1.0),
                quote_factor,
                min(expert_ref, 1.0),
                min(controversy, 1.0),
                min(multi_source, 1.0),
                min(coherence, 1.0),
            ])
        return np.array(rows, dtype=float)
