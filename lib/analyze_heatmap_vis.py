from dataset.util import mkdir, download_prices, reduce_embs
from scipy.ndimage.filters import gaussian_filter
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import tqdm
import glob
import os


COMP_MAP_FN = glob.glob(os.path.join('data', 'company-embs-*-map.pkl'))[0]
COMP_EMBS_FNS = glob.glob(os.path.join('data', 'company-embs-*-article-embs-*-*-*-keras-*-*.npy'))
SAVE_DIR = os.path.join('data', 'heat-vis')


class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def _load_price_data(symbols):
    df_all = None
    for sym in symbols:
        if sym in ['LYFT', 'ONEM', 'WORK']:
            # omit late ipos
            continue
        df = download_prices(sym).set_index('date')
        df = df[['lg_tclose_tmclose']]
        df = df.rename(columns={'lg_yclose_tclose': sym})
        if df_all is None:
            df_all = df
        else:
            df_all = pd.merge(df_all, df, left_index=True, right_index=True)
    dates = list(df_all.index)
    return dates, df_all


def _gen_frame(date_idx, date, prices, rembs, blur=25):

    # bounds for centering
    xmin, xmax = rembs[:, 0].min(), rembs[:, 0].max()
    ymin, ymax = rembs[:, 1].min(), rembs[:, 1].max()
    xr = xmax - xmin
    yr = ymax - ymin

    grid = np.zeros((500, 500))
    price_day = prices[date_idx]
    if date_idx >= 3:
        # smoothing
        price_day = (
            prices[date_idx-2] * 0.2 + 
            prices[date_idx-1] * 0.3 + 
            prices[date_idx] * 0.5
        )
    for i, price in enumerate(price_day):
        if price != price:
            # nans
            continue
        x = int((rembs[i, 0] - xmin) / xr * 400 + 50)
        y = int((rembs[i, 1] - ymin) / yr * 400 + 50)
        grid[y, x] = np.square(price) * np.sign(price)

    # cool effect
    grid = gaussian_filter(grid, sigma=blur)

    plt.rcParams["figure.figsize"] = (20, 20)
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    plt.title(date)
    plt.imshow(grid, cmap=plt.cm.RdBu, interpolation='nearest', norm=MidpointNormalize(midpoint=0.))
    plt.savefig(os.path.join(SAVE_DIR, '{:05d}.png'.format(date_idx)))


def heatmap_vis(embs_substring, **kwargs):

    comp_embs_fn = None
    for fn in COMP_EMBS_FNS:
        if embs_substring in fn:
            comb_embs_fn = fn
            break

    embs = np.load(comb_embs_fn)
    rembs = reduce_embs(embs)[1]
    with open(COMP_MAP_FN, 'rb') as f:
        sym_to_idx = pickle.load(f)

    symbols = []
    for sym in sym_to_idx:
        # see what data is available
        try:
            download_prices(sym)
        except:
            break
        else:
            symbols.append(sym)

    print('Loaded', len(sym_to_idx), 'companies, using', len(symbols))
    print('Saving frames to', SAVE_DIR)

    dates, df = _load_price_data(symbols)

    prices = df.to_numpy()
    prices = (prices - prices.mean(axis=0)) / prices.std(axis=0)

    mkdir(SAVE_DIR)

    for date_idx, date in tqdm.tqdm(enumerate(dates), total=len(dates)):
        _gen_frame(date_idx, date, prices, rembs, **kwargs)

    print('Done. Combine with $ ffmpeg -r 10 -i \'%5d.png\' -c:v libx264 -vf format=yuv420p heatmap.mp4')


if __name__ == '__main__':
    heatmap_vis('counts-content-keras-1024-3')