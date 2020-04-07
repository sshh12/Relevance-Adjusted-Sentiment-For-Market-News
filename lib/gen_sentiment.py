from dataset.util import sql_read_articles
from sentiment.articles import SENTIMENT_ALGOS
import pickle


def main():

    articles = sql_read_articles(only_labeled=True)
    ids = [a[0] for a in articles]
    headlines = [a[2] for a in articles]
    content = [a[2] + '\n\n' + a[4] for a in articles]

    with open('data/article-sentiment-{}-ids.pkl'.format(len(articles)), 'wb') as pkl_file:
        pickle.dump(ids, pkl_file)

    tests = []

    for SentAlgo in SENTIMENT_ALGOS:
        tests.append(SentAlgo('headlines', headlines))
        tests.append(SentAlgo('content', content))

    for test in tests:
        test.prep()
        test.bake_sentiment()
        test.plot()
        test.save_all()


if __name__ == "__main__":
    main()
