from dataset.util import download_prices
from collections import defaultdict
import plotly.express as px
from scipy import spatial
import pandas as pd
import numpy as np
import pickle
import tqdm
import glob
import os


COMP_MAP_FN = glob.glob(os.path.join('data', 'company-embs-*-map.pkl'))[0]
COMP_EMBS_FNS = glob.glob(os.path.join('data', 'company-embs-*-article-embs-*-*-*-keras-*-*.npy'))


def _load_price_data(symbols, price_col):
    df_all = None
    for sym in symbols:
        df = download_prices(sym).set_index('date')
        df = df[[price_col]]
        df = df.rename(columns={price_col: sym})
        if df_all is None:
            df_all = df
        else:
            merged_df_all = pd.merge(df_all, df, left_index=True, right_index=True)
            if len(merged_df_all.index) >= 365:
                df_all = merged_df_all
    dates = list(df_all.index)
    return dates, df_all


def plot_price_corr_vs_emb_dist(embs_substring, price_col='lg_close'):

    comp_embs_fn = None
    for fn in COMP_EMBS_FNS:
        if embs_substring in fn:
            comb_embs_fn = fn
            break

    embs = np.load(comb_embs_fn)
    with open(COMP_MAP_FN, 'rb') as f:
        sym_to_idx = pickle.load(f)

    symbols = []
    for sym in sym_to_idx:
        # see what data is available
        try:
            download_prices(sym)
            symbols.append(sym)
        except:
            pass

    dates, df = _load_price_data(symbols, price_col)
    # b/c some were removed
    symbols = list(df.columns)

    print('Loaded', len(sym_to_idx), 'companies, using', len(symbols))
    print('From', dates[0], 'to', dates[-1])

    plot_data = defaultdict(list)
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            sym_a = symbols[i]
            sym_b = symbols[j]
            corr_a_b = float(df[[sym_a]].corrwith(df[sym_b]).to_numpy().squeeze())
            emb_sim = float(np.dot(embs[sym_to_idx[sym_a]], embs[sym_to_idx[sym_b]]))
            if corr_a_b != corr_a_b or emb_sim != emb_sim:
                # nan-ness check
                continue
            plot_data['Price Correlation'].append(float(corr_a_b))
            plot_data['Emb. Similarity'].append(float(emb_sim))
            plot_data['name'].append(sym_a + ' ' + sym_b)

    plot_df = pd.DataFrame(plot_data)
    fig = px.scatter(plot_df, x='Emb. Similarity', y='Price Correlation', 
        hover_name='name', trendline='ols', 
        title='Company Pairs: Price Correlation vs. Embedding Similarity')
    fig.show()

    print(plot_df.corr())


if __name__ == '__main__':
    plot_price_corr_vs_emb_dist('counts-content-keras-1024-3', 'lg_topen_to_tclose')