from datetime import datetime, timedelta
from multiprocessing import Pool
import requests
import sqlite3
import re

from util import mw_format_date, clean_html_text, ignore_this_text


def fetch_meta(symbol):
    url = 'https://www.marketwatch.com/investing/stock/{}/profile'.format(symbol)
    html = requests.get(url).text
    desc = re.search(r'<div class="full">\s+<p>([^<]+?)<\/p>', html).group(1).strip()
    return (symbol.upper(), desc)


def fetch_iter_news(symbol, date=None):

    if date is None:
        date = datetime.now()

    while True:
        form_date = mw_format_date(date)
        url = 'https://www.marketwatch.com/news/headline/getheadlines?ticker={0}&dateTime={1}&countryCode=US&count=16&channelName=%2Fnews%2Flatest%2Fcompany%2Fus%{0}'.format(symbol, form_date)
        resp = requests.get(url).json()
        for art in resp:
            art_data = (
                date.strftime('%Y-%m-%d'),
                "https://www.marketwatch.com/story" + art['SeoHeadlineFragment']
            )
            yield art_data
        date = date - timedelta(days=1)


def fetch_article(url):
    
    article_html = requests.get(url).text

    headline_match = re.search(r'itemprop="headline">([\s\S]+?)<\/h1>', article_html)
    if not headline_match:
        return (None, "")
    headline = clean_html_text(headline_match.group(1))

    text = []

    start_idx = article_html.index('articleBody')
    try:
        end_idx = article_html.index('author-commentPromo')
    except ValueError:
        end_idx = len(article_html)

    content_html = article_html[start_idx:end_idx]
    for paragraph_match in re.finditer(r'<p>([\s\S]+?)<\/p>', content_html):
        p = clean_html_text(paragraph_match.group(1))
        if not ignore_this_text(p):
            text.append(p)

    return (headline, "\n\n\n".join(text))


def dl_data_for_symbol(symbol, limit=float('inf')):
    (conn, cur) = sql_connect()
    _, desc = fetch_meta(symbol)
    for i, (date, url) in enumerate(fetch_iter_news(symbol)):
        (headline, content) = fetch_article(url)
        if headline is None:
            continue
        cur.execute("""
        INSERT OR IGNORE INTO articles
         (headline, date, content, url)
        VALUES
         (?,?,?,?)
        """, (headline, date, content, url))
        if i % 10 == 0:
            (conn, cur) = sql_save(conn, cur, restart=True)
        if i > limit:
            break
    sql_save(conn, cur)


def sql_connect(try_init=True):
    conn = sqlite3.connect('db.sqlite')
    conn.execute("PRAGMA busy_timeout = 60000")
    cur = conn.cursor()
    try:
        cur.execute("""
        CREATE TABLE articles (
            article_id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline VARCHAR(255),
            date VARCHAR(10),
            content TEXT,
            url VARCHAR(255) UNIQUE
        )
        """)
        conn.commit()
    except sqlite3.OperationalError:
        pass
    return (conn, cur)


def sql_save(conn, cur, restart=False):
    conn.commit()
    conn.close()
    if restart:
        return sql_connect(try_init=False)
    return (None, None)


def main():

    tickers = ['AAPL', 'AMD']
    pool = Pool(processes=4)
    pool.map(dl_data_for_symbol, tickers)

    (conn, cur) = sql_connect()
    print(cur.execute('SELECT COUNT(article_id) FROM articles').fetchall())


if __name__ == "__main__":
    main()
