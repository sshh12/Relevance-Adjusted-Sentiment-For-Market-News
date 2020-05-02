from dataset.util import sql_read_articles, mkdir, download_prices
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

    def load(self):
        self.model = load_model(self.relv_model_fn)
        self.art_embs = np.load(self.art_embs_fn)
        self.sent = load_sentiment(self.sent_fn)

    def score(self, symbol, art_idxs):

        sentiment = self.sent[art_idxs]
        article_embs = self.art_embs[art_idxs]

        # b/c gcp sent scores somewhat broken
        valid_sent_idxs = (sentiment != -1000)  
        sentiment = sentiment[valid_sent_idxs]
        article_embs = article_embs[valid_sent_idxs]

        symbol_idxs = [self.sym_to_idx[symbol]] * len(article_embs)

        relv = self.model.predict([symbol_idxs, article_embs])
        relv_sentiment = sentiment * relv
        return np.mean(relv_sentiment)

    def get_id(self):
        return os.path.splitext(os.path.basename(self.relv_model_fn))[0] \
            + '-' + os.path.splitext(os.path.basename(self.sent_fn))[0]


def main(symbol, plot=True):

    mkdir(os.path.join('data', 'plot_ckpt'))

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
            ckpt_fn = os.path.join('data', 'plot_ckpt', symbol + '-' + name + '.npy')
            if not os.path.exists(ckpt_fn):
                print('Computing', name)
                RS.load()
                scores = []
                for date in tqdm.tqdm(dates):
                    score = RS.score(symbol, article_idxs_by_date[date])
                    scores.append(score)
                np.save(ckpt_fn, scores)
            else:
                print('Already computed...skipping.')
                scores = list(np.load(ckpt_fn))
            plot_data[name] = scores

    prices = download_prices(symbol)
    df = pd.DataFrame(plot_data)
    for name in names:
        for win in [5, 7, 10, 30]:
            df[name + '_emw' + str(win)] = df[name].ewm(span=win).mean()
            df[name + '_ma' + str(win)] = df[name].rolling(win).mean()
        df[name + '_cumsum'] = df[name].cumsum()

    df_corr = df.merge(prices, on='date')
    df_corr.to_csv(os.path.join('data', 'prices-by-date-' + symbol + '.csv'))

    price_cols = [c for c in prices.columns if c != 'date']
    corr_table = df_corr.corr()
    corr_table[price_cols].to_csv(os.path.join('data', 'price-corr-' + symbol + '.csv'))

    if plot:
        df_plot = df.merge(prices[['date', 'lg_topen_to_tclose']], on='date')
        df_plot = df.melt(id_vars=['date'], var_name='method', value_name='score')
        fig = px.line(df_plot, x='date', y='score', color='method')
        fig.show()


if __name__ == "__main__":
    main('GE', plot=False)
