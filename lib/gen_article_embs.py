

from dataset.util import sql_read_articles, sql_read_companies_dict
from embs.articles import EMBEDDINGS
import pickle


def main():

    comps = sql_read_companies_dict()
    articles = sql_read_articles(only_labeled=True)
    sectors = [comps[a[1]][4] for a in articles]
    ids = [a[0] for a in articles]
    headlines = [a[2] for a in articles]
    content = [a[2] + '\n\n' + a[4] for a in articles]

    with open('data/article-embs-{}-ids.pkl'.format(len(articles)), 'wb') as pkl_file:
        pickle.dump(ids, pkl_file)

    tests = []

    for Emb in EMBEDDINGS:
        tests.append(Emb('headlines', headlines))
        tests.append(Emb('content', content))

    for test in tests:
        test.prep()
        test.bake_embs()
        test.plot('Sector', sectors)
        test.save_all()


if __name__ == "__main__":
    main()
