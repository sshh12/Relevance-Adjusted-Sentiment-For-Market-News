from dataset.util import sql_read_articles
from sentiment.articles import load_sentiment
from keras.models import load_model
from collections import defaultdict
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import tqdm
import glob
import os


RELV_MODELS = glob.glob(os.path.join('data', 'company-embs-*-article-embs-*-*-*-*-*-*.h5'))
COMP_MAP_FN = glob.glob(os.path.join('data', 'company-embs-*-map.pkl'))[0]
SENTIMENT_FNS = glob.glob(os.path.join('data', 'article-sentiment-*-*-*.npy'))


class RSentimentScore:

    def __init__(self, sym_to_idx, relv_model_fn, sent_fn):
        self.sym_to_idx = sym_to_idx
        self.relv_model_fn = relv_model_fn
        self.sent_fn = sent_fn
        self.art_embs_fn = os.path.join('data', '-'.join(
            os.path.basename(relv_model_fn).split('-')[3:-5]) + '.npy')
        self.art_embs = np.load(self.art_embs_fn)
        self.sent = load_sentiment(self.sent_fn)
        self.sent = self.sent
        self.model = load_model(self.relv_model_fn)

    def score(self, symbol, art_idxs):

        sentiment = self.sent[art_idxs]
        article_embs = self.art_embs[art_idxs]

        valid_sent_idxs = (sentiment != -1000)  # b/c gcp sent somewhat broken
        sentiment = sentiment[valid_sent_idxs]
        article_embs = article_embs[valid_sent_idxs]

        symbol_idxs = [self.sym_to_idx[symbol]] * len(article_embs)

        relv = self.model.predict([symbol_idxs, article_embs])
        relv_sentiment = sentiment * relv
        return np.mean(relv_sentiment)

    def get_id(self):
        return os.path.splitext(os.path.basename(self.relv_model_fn))[0] \
            + '-' + os.path.splitext(os.path.basename(self.sent_fn))[0]


def _dl_prices(symb, key='KOZNM03XM806URDU'):
    fn = os.path.join('data', 'PRICE_' + symb + '.csv')
    if not os.path.exists(fn):
        df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol={}&apikey={}&datatype=csv'.format(symb, key))
        df = df.rename(columns={'timestamp': 'date'})
        df.sort_values('date', ascending=True, inplace=True)
        df.to_csv(fn, index=False)
    df = pd.read_csv(fn)
    df['log_close'] = df['close'].apply(np.log)
    df['log_open'] = df['open'].apply(np.log)
    df['log_open_to_close'] = df['log_close'] - df['log_open']
    df['log_close_close'] = df['log_close'] - df['log_close'].shift(-1)
    return df


def main(symbol, price_column, plot=True):

    articles = sql_read_articles(only_labeled=True)
    article_idxs_by_date = defaultdict(list)
    for i, article in enumerate(articles):
        article_idxs_by_date[article[3]].append(i)
    dates = sorted(article_idxs_by_date)
    dates = [d for d in dates if d.startswith('2019') or d.startswith('2020')]

    with open(COMP_MAP_FN, 'rb') as f:
        sym_to_idx = pickle.load(f)

    plot_data = defaultdict(list)
    plot_data['date'] = dates
    names = []
    for relv_model_fn in RELV_MODELS:
        for sentiment_fn in SENTIMENT_FNS:
            RS = RSentimentScore(sym_to_idx, relv_model_fn, sentiment_fn)
            name = RS.get_id()
            names.append(name)
            for date in tqdm.tqdm(dates):
                score = RS.score(symbol, article_idxs_by_date[date])
                plot_data[name].append(score)

    prices = _dl_prices(symbol)
    df = pd.DataFrame(plot_data)
    for name in names:
        df[name + '_emw'] = df[name].ewm(span=5).mean()
        df[name + '_emw20'] = df[name].ewm(span=20).mean()
        df[name + '_sum'] = df[name].cumsum()
    df = df.merge(prices[['date', 'log_open_to_close']], on='date')
    print(df.corr()['log_open_to_close'])

    if plot:
        df_plot = df.melt(id_vars=['date'], var_name='method', value_name='score')
        fig = px.line(df_plot, x='date', y='score', color='method')
        fig.show()


if __name__ == "__main__":
    main('NFLX', 'log_open_to_close', plot=False)