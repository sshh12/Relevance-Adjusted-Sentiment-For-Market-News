from datetime import datetime, timedelta
from multiprocessing import Pool
import requests
import sqlite3
import re

from util import mw_format_date, clean_html_text, ignore_this_text
from config import SYMBOLS


def fetch_meta(symbol):
    url = 'https://www.marketwatch.com/investing/stock/{}/profile'.format(symbol)
    html = requests.get(url).text
    try:
        name = re.search(r'<p class="companyname">([^<]+?)<\/p>', html).group(1).strip()
        desc = re.search(r'<div class="full">\s+<p>([^<]+?)<\/p>', html).group(1).strip()
        industry = re.search(r'<p class="column">Industry<\/p>\s+<p class="data lastcolumn">([^<]+?)<\/p>', html).group(1).strip()
        sector = re.search(r'<p class="column">Sector<\/p>\s+<p class="data lastcolumn">([^<]+?)<\/p>', html).group(1).strip()
        return (symbol.upper(), name, industry, sector, desc)
    except AttributeError:
        return (symbol.upper(), None, None, None, None)


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


def dl_data_for_symbol(symbol, limit=10000):
    (conn, cur) = sql_connect()
    symb, name, industry, sector, desc = fetch_meta(symbol)
    if name is None:
        print('No data for:', symbol)
        return
    else:
        print('Scraping:', symbol)
    cur.execute("""
        INSERT OR IGNORE INTO companys
         (symbol, name, industry, sector, desc)
        VALUES
         (?,?,?,?,?)
    """, (symb, name, industry, sector, desc))
    for i, (date, url) in enumerate(fetch_iter_news(symbol)):
        exists = (cur.execute("SELECT COUNT(url) FROM articles WHERE url = ?", (url,)).fetchone()[0] == 1)
        if exists:
            continue
        (headline, content) = fetch_article(url)
        if headline is None:
            continue
        cur.execute("""
        INSERT OR IGNORE INTO articles
         (symbol, headline, date, content, url)
        VALUES
         (?,?,?,?,?)
        """, (symb, headline, date, content, url))
        (conn, cur) = sql_save(conn, cur, restart=True)
        if i > limit:
            break
    sql_save(conn, cur)


def sql_connect(try_init=True):
    conn = sqlite3.connect('db.sqlite')
    conn.execute("PRAGMA busy_timeout = 120000")
    cur = conn.cursor()
    try:
        cur.execute("""
        CREATE TABLE articles (
            article_id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline VARCHAR(255),
            date VARCHAR(10),
            content TEXT,
            url VARCHAR(255) UNIQUE
        )""")
        cur.execute("""
        CREATE TABLE companys (
            company_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol VARCHAR(10) UNIQUE,
            name VARCHAR(255),
            industry VARCHAR(255),
            sector VARCHAR(255),
            desc TEXT
        )""")
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

    def print_num_articles():
        (conn, cur) = sql_connect()
        print('Articles cnt:', cur.execute('SELECT COUNT(article_id) FROM articles').fetchone()[0])
        sql_save(conn, cur)

    print_num_articles()

    pool = Pool(processes=4)
    pool.map(dl_data_for_symbol, SYMBOLS)

    print_num_articles()


if __name__ == "__main__":
    main()
