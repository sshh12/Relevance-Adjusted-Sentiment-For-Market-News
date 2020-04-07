from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import os

class AbstractSentiment:

    TAG = 'abs'

    def __init__(self, name, docs, ds_name=None):
        if ds_name is None:
            ds_name = str(len(docs))
        self.docs = docs
        self.exp_id = 'article-sentiment-{}-{}-{}'.format(ds_name, self.TAG, name)
        self.exp_name = 'Article Sentiment ({} on {})'.format(self.TAG, name)
        self.pickles = []
        self.figs = {}

    def prep(self):
        pass

    def bake_sentiment(self):
        raise NotImplementedError()
    
    def plot(self):
        df = pd.DataFrame({
            'score': self.doc_sent,
            'doc': [d[:50] for d in self.docs]
        })
        self.figs["hist"] = px.histogram(df, x="score", title=self.exp_name)
        self.figs["hist"].show()

    def save_all(self, folder='data'):
        fn_sent = os.path.join(folder, '{}.npy'.format(self.exp_id))
        np.save(fn_sent, self.doc_sent)
        if len(self.figs) > 0:
            for name, fig in self.figs.items():
                fn_fig = os.path.join(folder, '{}-{}.png'.format(self.exp_id, name))
                fig.write_image(fn_fig)


class TextBlobSentiment(AbstractSentiment):

    TAG = 'textblob'

    def _score(self, doc):
        return TextBlob(doc).sentiment.polarity

    def bake_sentiment(self):
        self.doc_sent = np.array([self._score(doc) for doc in self.docs])


class VADERSentiment(AbstractSentiment):

    TAG = 'vader'

    def prep(self):
        self.sia = SentimentIntensityAnalyzer()
        with open(os.path.join('data', 'vader_lexicon.pkl'), 'rb') as lex_file:
            self.sia.lexicon = pickle.load(lex_file)

    def _score(self, doc):
        return self.sia.polarity_scores(doc)['compound']

    def bake_sentiment(self):
        self.doc_sent = np.array([self._score(doc) for doc in self.docs])


SENTIMENT_ALGOS = [
    TextBlobSentiment,
    VADERSentiment
]