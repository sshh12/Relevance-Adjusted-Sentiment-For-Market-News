

from dataset.util import sql_read_articles, sql_read_companies_dict
from embs.articles import load_embs_from_exp_id
from embs.companies import KerasDeep
import numpy as np
import random
import pickle
import glob
import os


EXP_IDS = [
    os.path.splitext(os.path.basename(fn))[0] 
    for fn in glob.iglob(os.path.join('data', 'article-embs-*-*-*.npy'))
]


def _make_dataset(art_embs, sym_to_idx, sym_to_art_idxs):
    S = []
    A = []
    Y = []
    all_ids = list(range(len(art_embs)))
    for sym, sym_idx in sym_to_idx.items():
        sym_arts = sym_to_art_idxs[sym]
        for art_id in sym_arts:
            S.append(sym_idx)
            A.append(art_embs[art_id])
            Y.append(1)
        nsym_arts = []
        for _ in sym_arts:
            nart_id = sym_arts[0]
            while nart_id in sym_arts or nart_id in nsym_arts:
                nart_id = random.choice(all_ids)
            nsym_arts.append(nart_id)
            S.append(sym_idx)
            A.append(art_embs[nart_id])
            Y.append(0)
    S = np.array(S)
    A = np.array(A)
    Y = np.array(Y)
    rand_ord = np.random.permutation(S.shape[0])
    S = S[rand_ord]
    A = A[rand_ord]
    Y = Y[rand_ord]
    return S, A, Y


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

        art_emb_tests = []
        for ls in [2024, 1024, 512, 65]:
            for layers in [0, 1, 3]:
                art_emb_tests.append(KerasDeep(art_exp_id, art_embs, 
                    sym_to_idx, sym_to_art_idxs, 
                    latent_size=ls, post_emb_layers=layers
                ))

        dataset = _make_dataset(art_embs, sym_to_idx, sym_to_art_idxs)
        for test in art_emb_tests:
            test.dataset = dataset
        tests.extend(art_emb_tests)

    for test in tests:
        print(test.exp_id)
        test.prep()
        test.bake_embs()
        test.plot('Sector', sectors, names)
        test.save_all()


if __name__ == "__main__":
    main()
