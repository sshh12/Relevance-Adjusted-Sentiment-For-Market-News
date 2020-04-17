from gensim.models.doc2vec import Doc2Vec as Doc2VecModel, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from bert_serving.client import BertClient
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import umap
import nltk
import os


def load_embs_from_exp_id(exp_id):
    fn = os.path.join('data', exp_id + '.npy')
    return np.load(fn)


class AbstractEmb:

    TAG = 'abs'

    def __init__(self, name, docs, ds_name=None):
        if ds_name is None:
            ds_name = str(len(docs))
        self.docs = docs
        self.exp_id = 'article-embs-{}-{}-{}'.format(ds_name, self.TAG, name)
        self.exp_name = 'Article Embeddings ({} on {})'.format(self.TAG, name)
        self.pickles = []
        self.figs = {}

    def prep(self):
        pass

    def bake_embs(self):
        raise NotImplementedError()

    def plot(self, label_name, labels):
        assert len(labels) == len(self.docs)
        reducer = umap.UMAP()
        docs_rembs = reducer.fit_transform(self.doc_embs)
        df = pd.DataFrame({
            'x': docs_rembs[:, 0], 'y': docs_rembs[:, 1],
            label_name: labels,
            'doc': [d[:50] for d in self.docs]
        })
        self.figs[label_name] = px.scatter(df, x="x", y="y", color=label_name, hover_data=['doc'], title=self.exp_name)
        self.figs[label_name].show()

    def save_all(self, folder='data'):
        fn_docs = os.path.join(folder, '{}.npy'.format(self.exp_id))
        np.save(fn_docs, self.doc_embs)
        if len(self.figs) > 0:
            for name, fig in self.figs.items():
                fn_fig = os.path.join(folder, '{}-{}.png'.format(self.exp_id, name.lower()))
                fig.write_image(fn_fig)
        if len(self.pickles) > 0:
            fn_pkl = os.path.join(folder, '{}.pkl'.format(self.exp_id))
            with open(fn_pkl, 'wb') as pkl_file:
                pickle.dump(self.pickles, pkl_file)


class PretrainedBERT(AbstractEmb):

    TAG = 'pretrainedbert'

    def prep(self):
        print('$ bert-serving-start -model_dir data/uncased_L-24_H-1024_A-16 -num_worker=1 -max_seq_len=512 -max_batch_size 64')
        self.bc = BertClient(check_length=False)

    def bake_embs(self):
        self.doc_embs = self.bc.encode(self.docs)


class FinetunedBERT(AbstractEmb):

    TAG = 'finetunedbert'

    def prep(self):
        print('$ bert-serving-start -model_dir data/uncased_L-12_H-768_A-12 -tuned_model_dir data/uncased_L-12_H-768_A-12_ft_symbol_pairs -ckpt_name=model.ckpt-100775 -num_worker=1 -max_seq_len=256 -max_batch_size 64')
        self.bc = BertClient(check_length=False)

    def bake_embs(self):
        self.doc_embs = self.bc.encode(self.docs)


class Doc2Vec(AbstractEmb):

    TAG = 'doc2vec'

    def prep(self):
        self.docs_token = [nltk.word_tokenize(d) for d in self.docs]
        self.docs_tagged = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.docs_token)]

    def bake_embs(self):
        model = Doc2VecModel(self.docs_tagged, vector_size=128, window=2, min_count=2, workers=4)
        self.doc_embs = np.array([model.infer_vector(dt) for dt in self.docs_token])
        model.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=False)
        self.pickles.extend([model])


class CountVec(AbstractEmb):

    TAG = 'counts'

    def bake_embs(self):
        count_model = CountVectorizer(max_features=2048, stop_words='english', lowercase=True, ngram_range=(1, 2))
        doc_counts = count_model.fit_transform(self.docs)
        freq_model = TfidfTransformer()
        doc_freqs = freq_model.fit_transform(doc_counts).toarray()
        mean, std = doc_freqs.mean(), doc_freqs.std()
        self.doc_embs = (doc_freqs - mean) / std
        self.pickles.extend([count_model, freq_model, mean, std])


EMBEDDINGS = [
    PretrainedBERT,
    FinetunedBERT,
    Doc2Vec,
    CountVec
]