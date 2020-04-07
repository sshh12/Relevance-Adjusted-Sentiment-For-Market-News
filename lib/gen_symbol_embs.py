

from dataset.util import sql_read_articles, sql_read_companies_dict
from embs.articles import load_embs_from_exp_id
from embs.companies import KerasDeep
import random
import pickle


EXP_IDS = [
    'article-embs-40-doc2vec-content'
]


def main():

    comps = sql_read_companies_dict()
    articles = sql_read_articles(only_labeled=True)
    sectors = [comps[c][4] for c in comps]
    names = [comps[c][2] + ' (' + comps[c][1] + ')' for c in comps]

    ids = [a[0] for a in articles]
    with open('data/article-embs-{}-ids.pkl'.format(len(articles)), 'rb') as pkl_file:
        assert ids == pickle.load(pkl_file)

    sym_to_idx = {sym: i for i, sym in enumerate(comps)}
    with open('data/company-embs-{}-map.pkl'.format(len(comps)), 'wb') as pkl_file:
        pickle.dump(sym_to_idx, pkl_file)

    sym_to_art_idxs = {}
    for sym in comps:
        art_idxs = [i for i, a in enumerate(articles) if a[1] == sym]
        random.shuffle(art_idxs)
        sym_to_art_idxs[sym] = art_idxs

    tests = []

    for art_exp_id in EXP_IDS:
        art_embs = load_embs_from_exp_id(art_exp_id)
        tests.append(KerasDeep(art_exp_id, art_embs, sym_to_idx, sym_to_art_idxs, 
            latent_size=1024, post_emb_layers=1))
        tests.append(KerasDeep(art_exp_id, art_embs, sym_to_idx, sym_to_art_idxs, 
            latent_size=1024, post_emb_layers=2))
        tests.append(KerasDeep(art_exp_id, art_embs, sym_to_idx, sym_to_art_idxs, 
            latent_size=512, post_emb_layers=1))
        tests.append(KerasDeep(art_exp_id, art_embs, sym_to_idx, sym_to_art_idxs, 
            latent_size=512, post_emb_layers=2))

    for test in tests:
        test.prep()
        test.bake_embs()
        test.plot('Sector', sectors, names)
        test.save_all()


if __name__ == "__main__":
    main()
