from nltk.sentiment.vader import SentimentIntensityAnalyzer
from allennlp.predictors.predictor import Predictor
from google.cloud import language_v1 as language
from google.cloud.language_v1 import enums
from textblob import TextBlob
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import time
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


class AllenNLPGlove(AbstractSentiment):

    TAG = 'allenglove'

    def prep(self):
        self.model = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/sst-2-basic-classifier-glove-2019.06.27.tar.gz")

    def _score(self, doc):
        return self.model.predict(sentence=doc)['probs'][0] - 0.5

    def bake_sentiment(self):
        self.doc_sent = np.array([self._score(doc) for doc in self.docs])


class GoogleCloud(AbstractSentiment):

    TAG = 'gcp'

    def prep(self):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join('data', 'gcp.json')
        self.gcp_client = language.LanguageServiceClient()

    def _score(self, doc):
        document = {'content': doc, 'type': enums.Document.Type.PLAIN_TEXT, 'language': 'en'}
        try:
            doc_resp = self.gcp_client.analyze_sentiment(document, encoding_type=enums.EncodingType.UTF8)
            polarity = doc_resp.document_sentiment.score
            mag = doc_resp.document_sentiment.magnitude
            time.sleep(0.2)  # just to be safe
            return polarity
        except Exception as e:
            print(e)
            return -1000

    def bake_sentiment(self):
        self.doc_sent = np.array([self._score(doc) for doc in self.docs])


def load_sentiment(fn, standardize=True):
    data = np.load(fn)
    if standardize:
        if '-gcp-' in fn:
            clean_idxs = (data != -1000)
            data = (data - data[clean_idxs].mean()) / data[clean_idxs].std()
            data[~clean_idxs] = -1000
        elif '-vader-content' in fn:
            temp = np.abs(data / 2)
            temp[temp == 0] = 0.001
            data = -np.sign(data) * (np.log(temp) + np.log(2))
            data = (data - data.mean()) / data.std()
        elif '-allenglove-' in fn:
            data = -np.sign(data) * (np.log(np.abs(data)) + np.log(2))
            data = (data - data.mean()) / data.std()
        else:
            data = (data - data.mean()) / data.std()
        data = np.clip(data, -3, 3)
    return data


SENTIMENT_ALGOS = [
    TextBlobSentiment,
    VADERSentiment,
    AllenNLPGlove,
    GoogleCloud
]