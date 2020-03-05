import re


IGNORE_TEXT = [
    'Read: ',
    'Now read: ',
    'See: ',
    'And see: ',
    'Read more: ',
    'Check out: ',
    'Related: ',
    'An expanded version of this',
    'Also: ',
    'See now: ',
    'Don\'t miss: ',
    'See also: ',
    'For more news: ',
    'Full coverage at ',
    'Additional reporting by ',
    'Sign up for ',
    'This story has ',
    'contributed to this',
    'Read this:',
    'This report originally',
    'click on this',
    'you understand and agree that we',
    'Recommended: '
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