from datetime import datetime, timedelta
from multiprocessing import Pool
import requests
import re

from util import (
    mw_format_date, clean_html_text, ignore_this_text,
    sql_connect, sql_merge, sql_add_company, sql_add_article
)
from config import SYMBOLS, MAX_PROCS


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


def dl_data_for_symbol(symbol, limit=5, batch_size=50):

    (conn, cur) = sql_connect(prefix=symbol)
    symb, name, industry, sector, desc = fetch_meta(symbol)

    if name is None:
        print('No data for:', symbol)
        return
    else:
        print('Scraping:', symbol)

    sql_add_company(cur, (symb, name, industry, sector, desc))

    batch = []
    found = 0
    try:
        for date, url in fetch_iter_news(symbol):
            exists = (cur.execute("SELECT COUNT(url) FROM articles WHERE url = ?", (url,)).fetchone()[0] == 1)
            if not exists:
                (headline, content) = fetch_article(url)
                if headline is None:
                    continue
                batch.append((symbol, headline, date, content, url))
                found += 1
            if len(batch) == batch_size or found > limit:
                for item in batch:
                    sql_add_article(cur, item)
                batch = []
                conn.commit()
                if found > limit:
                    break
    except KeyboardInterrupt:
        print('Interrupted!')
    conn.close()





def main():

    def print_db_info():
        (conn, cur) = sql_connect()
        print('Articles:', cur.execute('SELECT COUNT(*) FROM articles').fetchone()[0])
        print('Companies:', cur.execute('SELECT COUNT(*) FROM companies').fetchone()[0])
        conn.close()

    print_db_info()

    pool = Pool(processes=MAX_PROCS)
    pool.map(dl_data_for_symbol, ['AAPL', 'AMD'])

    print('Merging...')
    sql_merge(['AAPL', 'AMD'])

    print_db_info()


if __name__ == "__main__":
    main()
