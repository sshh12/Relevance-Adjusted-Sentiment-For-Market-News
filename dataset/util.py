import sqlite3
import glob
import re
import os

from config import DATABASE_URI


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
    'Related:'
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


def ignore_this_text(text):
    return string_contains(text, IGNORE_TEXT) or len(text) < 30


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


def sql_attempt(conn, cur, sql):
    try:
        cur.execute(sql)
        conn.commit()
        return True
    except sqlite3.OperationalError:
        pass
    return False


def sql_connect(group=''):
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
