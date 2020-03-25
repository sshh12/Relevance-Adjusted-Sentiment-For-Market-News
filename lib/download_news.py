from datetime import datetime, timedelta
from newspaper import Article
import requests
import pendulum
import time
import re

from dataset.util import (
    mw_format_date, reut_format_date, clean_html_text, ignore_this_text,
    sql_connect, sql_merge, sql_add_company, sql_add_article, run_multi,
    salpha_format_date
)
from dataset.config import SYMBOLS, MAX_PROCS


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


def mw_fetch_iter_news(symbol, date=None):

    if date is None:
        date = datetime.now()

    bad_attempts = 0

    while bad_attempts < 365:
        form_date = mw_format_date(date)
        url = 'https://www.marketwatch.com/news/headline/getheadlines?'\
            + 'ticker={0}&dateTime={1}&countryCode=US&count=16&channelName=%2Fnews%2Flatest%2Fcompany%2Fus%{0}'.format(symbol, form_date)
        resp = requests.get(url).json()
        for art in resp:
            art_data = (
                date.strftime('%Y-%m-%d'),
                'https://www.marketwatch.com/story' + art['SeoHeadlineFragment']
            )
            yield art_data
        date = date - timedelta(days=1)
        if len(resp) > 0:
            bad_attempts = 0
        else:
            bad_attempts += 1


def reut_fetch_iter_news(symbol, date=None):

    if date is None:
        date = datetime.now()

    bad_attempts = 0

    while bad_attempts < 365:
        form_date = mw_format_date(date)
        url = 'https://wireapi.reuters.com/v8/feed/rcom/us/marketnews/ric:{}.OQ?until={}'.format(symbol, form_date)
        resp = requests.get(url).json()
        arts = resp.get('wireitems', [])
        for art in arts:
            date_id = art['wireitem_id']
            action = None
            for template in art['templates']:
                if 'template_action' in template:
                    action = template['template_action']
                    break
            if action is None:
                continue
            art_date = datetime.fromtimestamp(int(date_id) / 1e9)
            art_data = (
                art_date.strftime('%Y-%m-%d'),
                action['url']
            )
            yield art_data
        if len(arts) > 0:
            bad_attempts = 0
            date = date - timedelta(days=2)
        else:
            bad_attempts += 1


def salpha_fetch_iter_news(symbol, date=None):

    if date is None:
        date = datetime.now()

    bad_attempts = 0

    UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'
    sess = requests.Session()
    def get_content(url):
        return sess.get(url, headers={'user-agent': UA}).text

    get_content('https://seekingalpha.com/symbol/{}'.format(symbol))

    while bad_attempts < 365:

        form_date = salpha_format_date(date)
        url = 'https://seekingalpha.com/symbol/{}/news/more_latest_news?page={}&new_layout=true'.format(symbol, form_date)
        resp = get_content(url)

        if 'Access to this page has been denied' in resp:
            time.sleep(5)
            bad_attempts += 1
            continue

        art_matches = re.findall(r'<div class=\\"symbol_article\\" time=\\"(\d+)\\"><a href=\\"([^"]+?)\\" sasource=\\"\w+?\\">([^<]+?)<\/a><\/div>', resp)
        for art_m in art_matches:
            date = art_m[0]
            path = art_m[1]
            art_date = datetime.fromtimestamp(int(date))
            art_data = (
                art_date.strftime('%Y-%m-%d'),
                'https://seekingalpha.com' + path
            )
            yield art_data
        if len(art_matches) > 0:
            bad_attempts = 0
        else:
            bad_attempts += 1
        date = date - timedelta(days=3)


def bensinga_fetch_iter_news(symbol, date=None):

    if date is None:
        date = datetime.now()

    stock_page = requests.get('https://www.benzinga.com/stock/{}/'.format(symbol.lower()))
    tid_match = re.search(r'"tids":"(\d+)"', stock_page.text)
    if not tid_match:
        return
    tid = tid_match.group(1)

    bad_attempts = 0

    while bad_attempts < 365:

        form_date = int(date.timestamp() / 100)
        url = 'https://www.benzinga.com/services/webapps/content?lastnid={}&parameters[tids]={}&parameters[type]=story,scoutfin_realtimebriefs,press_releases'.format(form_date, tid)
        resp = requests.get(url).json()

        for article in resp:
            date = pendulum.from_format(article['created'], 'ddd, D MMM YYYY HH:mm:ss ZZ')
            url = article['url']
            yield date.strftime('%Y-%m-%d'), url
        if len(resp) > 0:
            bad_attempts = 0
        else:
            bad_attempts += 1
        date = date - timedelta(days=3)


def mw_fetch_article(url):
    
    article_html = requests.get(url).text

    headline_match = re.search(r'itemprop="headline">([\s\S]+?)<\/h1>', article_html)
    if not headline_match:
        return (None, "")
    headline = clean_html_text(headline_match.group(1))

    text = []

    try:
        start_idx = article_html.index('articleBody')
    except ValueError:
        return (None, "")
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


def reut_fetch_article(url):
    
    article_html = requests.get(url).text

    headline_match = re.search(r'ArticleHeader_headline">([^<]+)<\/h1>', article_html)
    if headline_match is None:
        return (None, "")
    headline = clean_html_text(headline_match.group(1))

    text = []
    try:
        start_idx = article_html.index('StandardArticleBody_body')
    except ValueError:
        return (None, "")
    try:
        end_idx = article_html.index('Attribution_container')
    except ValueError:
        end_idx = len(article_html)
    content_html = article_html[start_idx:end_idx]
    for paragraph_match in re.finditer(r'<p>([^<]+)<\/p>', content_html):
        paragraph = clean_html_text(paragraph_match.group(1))
        if not ignore_this_text(paragraph):
            text.append(paragraph)

    if len(text) == 0:
        return (None, "")

    return (headline, "\n\n\n".join(text))


def sa_fetch_article(url):

    UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'
    
    article_html = requests.get(url, headers={
        'User-Agent': UA
    }).text
    if 'Access to this page has been denied' in article_html:
        time.sleep(5)
        return (None, "")

    headline_match = re.search(r'itemprop="headline">([^<]+)<', article_html)
    if not headline_match:
        return (None, "")
    headline = clean_html_text(headline_match.group(1))

    if ignore_this_text(headline, mode='salpha-headline'):
        return (None, "")

    text = []
    for bullet_match in re.finditer(r'<p class="bullets_li">([\s\S]+?)<\/p>', article_html):
        bullet_text = clean_html_text(bullet_match.group(1))
        if ignore_this_text(bullet_text, mode='salpha'):
            continue
        text.append(bullet_text)

    if len(text) < 2:
        return (None, "")

    return (headline, "\n\n\n".join(text))


def benzinga_fetch_article(url):

    try:
        art = Article(url)
        art.download()
        art.parse()
        headline = clean_html_text(art.title)
        text = clean_html_text(art.text)
        text = '\n'.join([t for t in text.split('\n') if not ignore_this_text(t)])
        assert len(text) > 30
    except:
        return (None, "")

    return (headline, text)


def dl_data_for_symbol(symbol, source, limit=5000, batch_size=50):

    (iter_news, fetch_article) = {
        'marketwatch': (mw_fetch_iter_news, mw_fetch_article),
        'reuters': (reut_fetch_iter_news, reut_fetch_article),
        'seekingalpha': (salpha_fetch_iter_news, sa_fetch_article),
        'benzinga': (bensinga_fetch_iter_news, benzinga_fetch_article)
    }[source]

    (conn, cur) = sql_connect(group=symbol)
    symb, name, industry, sector, desc = fetch_meta(symbol)

    if name is None:
        print('No data for:', symbol)
        return
    else:
        print('Scraping:', symbol, 'from', source)

    sql_add_company(cur, (symb, name, industry, sector, desc))

    batch = []
    recents = set()
    found = 0
    for date, url in iter_news(symbol):
        recent_key = symbol + '_' + url
        exists = (cur.execute('SELECT COUNT(url) FROM articles WHERE url = ? AND symbol = ?', 
            (url, symbol)).fetchone()[0] == 1)
        if not exists and recent_key not in recents:
            (headline, content) = fetch_article(url)
            if headline is None:
                continue
            batch.append((symbol, headline, date, content, url, source))
            recents.add(recent_key)
            print(symbol, url)
            found += 1
        if len(batch) == batch_size or found > limit:
            for item in batch:
                sql_add_article(cur, item)
            batch = []
            conn.commit()
            if found > limit:
                break
    conn.close()


def print_stats():
    (conn, cur) = sql_connect()
    print('Articles:', cur.execute('SELECT COUNT(*) FROM articles').fetchone()[0])
    print('Companies:', cur.execute('SELECT COUNT(*) FROM companies').fetchone()[0])
    conn.close()


def main():

    print_stats()

    n = len(SYMBOLS)
    runs = zip(
        SYMBOLS + SYMBOLS + SYMBOLS,
        ['reuters'] * n + ['marketwatch'] * n + ['seekingalpha'] * n + ['benzinga'] * n
    )
    run_multi(dl_data_for_symbol, runs, shuffle=True)

    print('Merging...')
    sql_merge()

    print_stats()


if __name__ == "__main__":
    main()
