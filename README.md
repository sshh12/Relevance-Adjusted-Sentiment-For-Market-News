# Relevance-Adjusted Sentiment For Market News

> TODO Abstract

![srelv_scores_overtime](https://user-images.githubusercontent.com/6625384/81112379-2dc82300-8ee4-11ea-8a60-68ee80046bb4.png)

## Full Usage

#### Generate Dataset

Add company symbols to `SYMBOLS` in `lib/dataset/config.py`.

Webscrape news and clean (headline, content, company labels, date) and store it in a local sqlite database.

`$ python lib\download_news.py`

#### Embeddings and Sentiment

Generate embeddings for articles, companies, and sentiment. Some methods may require additional dependencies.

Adjust embedding methods by editing `EMBEDDINGS` in `lib\embs\*.py`.

1. `$ python lib\gen_article_embs.py`
2. `$ python lib\gen_symbol_embs.py`
3. `$ python lib\gen_sentiment.py`

#### Generate Adjusted Sentiment Scores

Compute the historical daily adjusted sentiment for a company.

`$ python lib\analyze_sent_with_price.py`

#### Misc Scripts

* `$ python lib\analyze_heatmap.py`
* `$ python lib\analyze_returns.py`

## Data

The original dataset is available on request.

Misc project files and historical daily relevance scores can be found on [google drive](https://drive.google.com/drive/folders/1st5B8ytoQ0DovkvXo09HK-m3ass-Yb9r?usp=sharing).

## Help

This project is minimally documented, so fill free to create an issue for help, issues, etc.