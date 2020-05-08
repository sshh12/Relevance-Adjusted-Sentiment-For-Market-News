from multiprocessing import Pool
import pandas as pd
import numpy as np
import random
import sqlite3
import signal
import glob
import umap
import re
import os

from .config import DATABASE_URI, MAX_PROCS


IGNORE_TEXT = [
    'Read:',
    'Now read:',
    'See:',
    'And see:',
    'Read more:',
    'Check out:',
    'Related: ',
    'An expanded version of this',
    'Also:',
    'See now:',
    'Don\'t miss:',
    'See also:',
    'For more news:',
    'Full coverage at ',
    'Additional reporting by ',
    'Sign up for ',
    'This story has ',
    'contributed to this',
    'Read this:',
    'This report originally',
    'click on this',
    'you understand and agree that we',
    'Recommended:',
    'Related:',
    'Below is a snapshot of',
    ', go here.',
    'Full details at http'
]

SALPHA_IGNORE_HEADLINE = [
    'on the hour',
    'beats on',
    ' misses on revenue'
    'equity offering',
    'Notable earnings',
    ' dividend'
    'leads after hour',
    'Gainers: ',
    ' beats by ',
    ' reports Q'
]


SAPLHA_IGNORE_TEXT = [
    'Scorecard, Yield Chart',
    'click here',
    'Press Release',
    'ETFs:',
    'See all stocks',
    'now read:',
    'Shelf registration',
    'call starts at',
    'debt offering',
    'Forward yield',
    'for shareholders of record',
    ' principal amount of'
]


def clean_html_text(html):
    html = html.replace('&rsquo;', '\'').replace('&lsquo;', '\'')
    html = html.replace('&ldquo;', '"').replace('&rdquo;', '"').replace('&quot;', '"')
    html = html.replace('&amp;', '&')
    html = html.replace('&copy;', '')
    html = html.replace('&nbsp;', ' ')
    html = html.replace('&lt;', '<').replace('&gt;', '>')
    html = html.replace('•', '*').replace('●', '* ')
    html = html.replace('\r', '')
    html = html.replace('—', '-').replace('&ndash;', '-').replace('&mdash;', '-')
    html = html.replace('‘', '\'').replace('’', '\'')
    html = html.replace('“', '').replace('”', '')
    html = re.sub(r'<style[\s\w=":/\.\-,\'!%&+@\|{}\(\);#~\?]*>([\s\S]+?)<\/style>', '', html)
    html = re.sub(r'<script[\s\w=":/\.\-,\'!%&+@\|{}\(\);#~\?]*>([\s\S]+?)<\/script>', '', html)
    html = re.sub(r'<\w+[\s\w=":/\.\-,\'!%&+@\|#~{}\(\);\?]*>', '', html)
    html = re.sub(r'<\/?[\w\-]+>', '', html)
    html = re.sub(r'<!-*[^>]+>', '', html)
    html = re.sub(r'&#[\w\d]+;', '', html)
    html = re.sub(r'\s{3,}', ' ', html)
    html = re.sub('([a-z])\s{2,}([A-Z])', '\\1 \\2', html)
    return html.strip()


def string_contains(text, items):
    text = text.lower()
    for item in items:
        item_lower = item.lower()
        if item_lower in text:
            return True
    return False


def ignore_this_text(text, mode='normal'):
    if mode == 'normal':
        return string_contains(text, IGNORE_TEXT) or len(text) < 30
    elif mode == 'salpha-headline':
        return string_contains(text, SALPHA_IGNORE_HEADLINE)
    elif mode == 'salpha':
        return string_contains(text, SAPLHA_IGNORE_TEXT) or len(text) < 5


def mw_format_date(date):
    formatted = date.strftime('%I:%M+%p+%b.+%d,+%Y')\
        .replace('PM', 'p.m.')\
        .replace('AM', 'a.m.')
    parts = formatted.split('+')
    if parts[0].startswith("0"):
        parts[0] = parts[0][1:]
    if parts[3].startswith("0"):
        parts[3] = parts[3][1:]
    return '+'.join(parts)


def reut_format_date(date):
    return str(int(date.timestamp() * 1e9))


def salpha_format_date(date):
    return str(int(date.timestamp()))


def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass


def download_prices(symb, key='KOZNM03XM806URDU', refresh=False):
    fn = os.path.join('data', 'PRICE_' + symb + '.csv')
    if not os.path.exists(fn) or refresh:
        df = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol={}&apikey={}&datatype=csv'.format(symb, key))
        try:
            df = df.rename(columns={'timestamp': 'date'})
            df.sort_values('date', ascending=True, inplace=True)
        except KeyError:
            raise Exception('Rate limited.')
        df.to_csv(fn, index=False)
    df = pd.read_csv(fn)
    df['lg_close'] = df['close'].apply(np.log)
    df['lg_open'] = df['open'].apply(np.log)
    df['lg_yopen_to_yclose'] = df['lg_close'].shift(1) - df['lg_open'].shift(1)
    df['lg_topen_to_tclose'] = df['lg_close'] - df['lg_open']
    df['lg_tmopen_to_tmclose'] = df['lg_close'].shift(-1) - df['lg_open'].shift(-1)
    df['lg_yclose_tclose'] = df['lg_close'] - df['lg_close'].shift(1)
    df['lg_tclose_tmclose'] = df['lg_close'].shift(-1) - df['lg_close']
    return df


def reduce_embs(embs):
    reducer = umap.UMAP()
    rembs = reducer.fit_transform(embs)
    return reducer, rembs


def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_multi(func, params, shuffle=False):
    params = list(params)
    random.shuffle(params)
    pool = Pool(MAX_PROCS, initializer=_init_worker)
    try:
        pool.starmap(func, params)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        print('Interrupted!')


def sql_attempt(conn, cur, sql):
    try:
        cur.execute(sql)
        conn.commit()
        return True
    except sqlite3.OperationalError:
        pass
    return False


def sql_connect(group=''):
    mkdir('data')
    actual_uri = DATABASE_URI
    if group:
        b, a = DATABASE_URI.split('.')
        actual_uri = b + '-' + group + '.' + a
    conn = sqlite3.connect(os.path.join('data', actual_uri))
    conn.execute('PRAGMA busy_timeout = 120000')
    cur = conn.cursor()
    sql_attempt(conn, cur, """
    CREATE TABLE articles (
        article_id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol VARCHAR(10),
        headline VARCHAR(255),
        date VARCHAR(10),
        content TEXT,
        url VARCHAR(255),
        UNIQUE(symbol, url)
    )""")
    sql_attempt(conn, cur, """
    CREATE TABLE companies (
        company_id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol VARCHAR(10) UNIQUE,
        name VARCHAR(255),
        industry VARCHAR(255),
        sector VARCHAR(255),
        desc TEXT
    )""")
    if sql_attempt(conn, cur, "ALTER TABLE articles ADD source VARCHAR(20)"):
        cur.execute("UPDATE articles SET source=?", ('marketwatch',))
        conn.commit()
    return (conn, cur)


def sql_add_article(cur, params):
    assert len(params) == 6, 'Bad Article'
    cur.execute("""
    INSERT OR IGNORE INTO articles
        (symbol, headline, date, content, url, source)
        VALUES
        (?,?,?,?,?,?)
    """, params)


def sql_add_company(cur, params):
    assert len(params) == 5, 'Bad Company'
    cur.execute("""
    INSERT OR IGNORE INTO companies
        (symbol, name, industry, sector, desc)
        VALUES
        (?,?,?,?,?)
    """, params)


def sql_merge(groups=None, delete=False):
    if groups is None:
        groups = [os.path.splitext(os.path.basename(fn))[0].replace('db-', '') 
            for fn in glob.glob(os.path.join('data', 'db-*.sqlite'))]
    (conn, cur) = sql_connect()
    for group in groups:
        (conn2, cur2) = sql_connect(group=group)
        cur2.execute('SELECT * FROM companies') 
        for row in cur2:
            sql_add_company(cur, row[1:])
        cur2.execute('SELECT * FROM articles') 
        for row in cur2:
            sql_add_article(cur, row[1:])
        conn2.close()
        if delete:
            os.remove(group + '-' + DATABASE_URI)
    conn.commit()


def sql_read_articles(only_labeled=False):
    (conn, cur) = sql_connect()
    cmd = 'SELECT article_id, symbol, headline, date, content, url FROM articles'
    if only_labeled:
        cmd += ' WHERE symbol != \'????\''
    cmd += ' ORDER BY article_id ASC'
    articles = cur.execute(cmd).fetchall()
    conn.close()
    return articles


def sql_read_companies_dict():
    (conn, cur) = sql_connect()
    companies = cur.execute('SELECT company_id, symbol, name, industry, sector, desc FROM companies ORDER BY company_id ASC').fetchall()
    conn.close()
    return {c[1]: c for c in companies}