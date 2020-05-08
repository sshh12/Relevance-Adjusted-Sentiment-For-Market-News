# Relevance-Adjusted Sentiment For Market News

> Media sentiment has been an important tool for investing over the last few decades. This paper addresses the issue of sentiment relevance when trying to measure attitudes with respect to a specific company. We analyze various methods for computing embeddings and sentiment using both classical and deep learning techniques to produce a more viable relevance-adjusted metric. Experimental results show intuitive changes in sentiment for the same news but different companies, interpretable company embeddings, as well as correlations with a company's stock price.

![srelv_scores_overtime](https://user-images.githubusercontent.com/6625384/81112379-2dc82300-8ee4-11ea-8a60-68ee80046bb4.png)

## Full Usage

#### Generate a Dataset

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
* `$ python lib\analyze_corr_and_comp_embs.py`

## Data

The original dataset is available on request.

Misc. project files and historical relavance-adjusted sentiment scores can be found on [google drive](https://drive.google.com/drive/folders/1st5B8ytoQ0DovkvXo09HK-m3ass-Yb9r?usp=sharing).

## Help

This project is minimally documented, so fill free to create an issue for help, issues, etc.