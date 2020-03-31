import pendulum
import json
import os
import re

from dataset.util import sql_connect, sql_add_article


def iter_dump(fn='news.data'):
    with open(os.path.join('data', fn), encoding='utf-8') as fp:
        for line in fp:
            article = json.loads(line)['_source']
            if article['source'] == 'twitter':
                continue
            date = pendulum.parse(article['date']).in_tz('America/Chicago')
            art_tup = (
                None,
                article['headline'],
                date.to_date_string(),
                article['content'],
                article['url'],
                'esdump-' + article['source']
            )
            yield art_tup


def _strip_name(name):
    name = name.replace(' Inc.', '').replace(' & Co.', '').replace('Co.', '')\
        .replace('Corp.', '').replace('Ltd.', '').replace(',', '')\
        .replace(' LP', '')
    name = re.sub(r' \([ \w]+\)', '', name)
    name = re.sub(r' \/[ \w]+\/', '', name)
    name = name.strip()
    name = re.sub(r' Group$', '', name)
    name = re.sub(r' Holdings$', '', name)
    name = re.sub(r'^The ', '', name)
    return name.strip()


def _str_includes(text, test):
    return re.search(r'\b' + test + r'\b', text) is not None


def find_obvious_companies(companies, article):
    _, headline, _, text, _, _ = article
    comps = set()
    for sym, name, desc in companies:
        sym = sym.upper()
        name = _strip_name(name)
        if _str_includes(text, sym) or _str_includes(headline, sym):
            comps.add(sym)
        elif _str_includes(text, name) or _str_includes(headline, name):
            comps.add(sym)
    return comps


def main():
    (conn, cur) = sql_connect()
    print('Articles:', cur.execute('SELECT COUNT(*) FROM articles').fetchone()[0])
    companies = cur.execute('SELECT symbol, name, desc FROM companies').fetchall()
    for i, article in enumerate(iter_dump()):
        comps = find_obvious_companies(companies, article)
        comps.add('????')
        for comp in comps:
            new_art = tuple([comp, *article[1:]])
            sql_add_article(cur, new_art)
        if i % 100 == 0:
            print('Processed:', i)
    print('Articles:', cur.execute('SELECT COUNT(*) FROM articles').fetchone()[0])
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
