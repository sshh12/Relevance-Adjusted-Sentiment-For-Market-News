import backtrader as bt
import pandas as pd
import numpy as np
import random
import tqdm
import os

from collections import defaultdict
from datetime import datetime


def _parse_rsentiment_data_and_make_strat(symbol, adjusted=True):

    data_fn = 'prices-by-date-{}.csv'.format(symbol)
    if not adjusted:
        data_fn = 'unadjusted-' + data_fn

    rs_df = pd.read_csv(os.path.join('data', data_fn))
    drop_cols = [
        'Unnamed: 0', 'low', 'close', 'volume', 'lg_close', 'lg_open', 'lg_yopen_to_yclose',
        'lg_topen_to_tclose', 'lg_tmopen_to_tmclose', 'lg_yclose_tclose',
        'lg_tclose_tmclose', 'open', 'high'
    ]
    for c in rs_df.columns:
        if '_cumsum' in c:
            drop_cols.append(c)
    rs_df = rs_df.drop(columns=drop_cols).set_index('date')

    class RSSignal(bt.Indicator):
    
        lines = ('rs',)
        params = (('method', None),)
            
        def next(self):
            ticker = self.data._name
            date = self.data.datetime.date()
            date_formatted = date.isoformat()[:10]
            try:
                val = rs_df.loc[date_formatted][self.p.method]
                self.l.rs[0] = val
            except Exception as e:
                self.l.rs[0] = 0
                
        def _plotinit(self):
            self.plotinfo.plotyhlines = [0]

    class RSStrat(bt.Strategy):
    
        params = (('thresh', 0.0), ('multiplier', 1), ('method', None))
        
        def __init__(self):
            self.signal = RSSignal(self.data0, method=self.p.method)
                
        def next(self):
            target = 0
            if self.signal[0] * self.p.multiplier > self.p.thresh:
                target = 1
            self.order_target_percent(data=self.data0, target=target)

    return rs_df, list(rs_df.columns), RSSignal, RSStrat


class LongStrat(bt.Strategy):
    
    params = (('thresh', 0.0), ('multiplier', 1), ('method', 'long'))
    
    def __init__(self):
        pass
            
    def next(self):
        self.order_target_percent(data=self.data0, target=1)


class RandomStrat(bt.Strategy):
    
    params = (('thresh', 0.0), ('multiplier', 1), ('method', 'rand'))
    
    def __init__(self):
        pass
            
    def next(self):
        target = random.choice([0, 1])
        self.order_target_percent(data=self.data0, target=target)


def _simulate_return(sym, strat, start, end, show=False, **kwargs):
    cerebro = bt.Cerebro()
    feed = bt.feeds.GenericCSVData(dataname='data/PRICE_{}.csv'.format(sym),
        dtformat="%Y-%m-%d", openinterest=-1,
        fromdate=start, todate=end)
    cerebro.adddata(feed, name=sym)
    cerebro.addstrategy(strat, **kwargs)
    cerebro.addanalyzer(bt.analyzers.Returns)
    run = cerebro.run()[0]
    if show:
        cerebro.plot()
    data = run.analyzers[0].get_analysis()
    return data['rtot']


def analyze_returns(symbol, start, end, adjusted=True, show=False):

    print('Simulating', symbol, start, end)

    rs_df, methods, RSSignal, RSStrat = \
        _parse_rsentiment_data_and_make_strat(symbol, adjusted=adjusted)
    
    data = {}
    mx = -9e9
    for method in tqdm.tqdm(methods):
        for thresh in [-0.075, 0.0, 0.075]:
            for multi in [-1, 1]:
                exp_id = (method, thresh, multi)
                if exp_id in data:
                    continue
                ret = _simulate_return(symbol, RSStrat, start, end, 
                    method=method, multiplier=multi, thresh=thresh)
                mx = max(mx, ret)
                data[exp_id] = ret

    results = defaultdict(list)
    best_result = list(data.keys())[0]

    print('Aggregating', len(data), 'results...')

    for (name, thresh, multi), ret in data.items():
        temp, mod = (name.split('_') + [""])[:2]
        ids = temp.split('-')
        if adjusted:
            a_method = ids[6]
            a_type = ids[7]
            c_num_emb = int(ids[9])
            c_num_hidden = int(ids[10])
            c_acc = float(ids[12])
            s_method = ids[16]
            s_type = ids[17]
        else:
            a_method = 'none'
            a_type = 'none'
            c_num_emb = 0
            c_num_hidden = 0
            c_acc = 0
            s_method = ids[4]
            s_type = ids[5]
        results['0-a_method-' + a_method].append(ret)
        results['1-s_method-' + s_method].append(ret)
        results['2-a_type-' + a_type].append(ret)
        results['3-s_type-' + s_type].append(ret)
        results['4-c_num_emb-' + str(c_num_emb)].append(ret)
        if ret > data[best_result]:
            best_result = (name, thresh, multi)
    results['5-long'] = _simulate_return(symbol, LongStrat, start, end)
    results['6-rand'] = np.median([
        _simulate_return(symbol, RandomStrat, start, end)
        for _ in range(20) 
    ])

    print(symbol)
    print('-'*20)
    for k in sorted(results):
        print(k, int(round(np.max(results[k]) * 100)))
    print('-'*20)

    print('Best', best_result)
    _simulate_return(symbol, RSStrat, start, end, show=show, 
        method=best_result[0], 
        thresh=best_result[1], 
        multiplier=best_result[2])


if __name__ == '__main__':
    analyze_returns('NFLX', datetime(2019, 1, 2), datetime(2020, 4, 9), adjusted=True, show=True)